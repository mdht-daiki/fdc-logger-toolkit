"""FastAPI ベースの DB API エントリポイント。

集約結果の書き込み系エンドポイントを提供し、アプリ単位で
`DBTaskRunner` を初期化・再利用・停止する。
"""

from __future__ import annotations

import json
import logging
import math
import os as _os
import pathlib
import sqlite3
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from dataclasses import asdict
from datetime import UTC, datetime, timedelta
from email.utils import format_datetime
from threading import Lock
from typing import Annotated, NoReturn, cast
from urllib.parse import quote

from fastapi import Depends, FastAPI, HTTPException, Path, Query, Request
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from .aggregate_repository import (
    delete_process,
    write_aggregate_atomic,
    write_parameters_bulk,
    write_process,
    write_step_windows_bulk,
)
from .audit_event_writer import AuditEventWriter
from .chart_repository import (
    ActiveChartsQueryCriteria,
    ChartRepository,
    ChartsHistoryQueryCriteria,
    ChartsQueryCriteria,
)
from .datetime_util import to_utc_millis
from .db import MAIN_DB, TEMP_DB, _connect, _connect_readonly, _init_schema
from .governance_repository import (
    GovernanceApprovalsRepository,
    GovernanceChangeRequestRepository,
    GovernanceNotFoundError,
)
from .judge_repository import (
    JudgeDataCorruptionError,
    JudgeRepository,
    JudgeResultsQueryCriteria,
)
from .schemas import (
    AggregateWriteIn,
    ChangeRequestApplyIn,
    ChangeRequestApproveIn,
    ChangeRequestIn,
    ChangeRequestsQuery,
    GovernanceAuditEventsQuery,
    ParameterIn,
    ProcessDeleteIn,
    ProcessInfoIn,
    StepWindowIn,
    validate_timestamp_range,
)
from .task_runner import DBTaskRunner

logger = logging.getLogger(__name__)
_runner_lock = Lock()
DATA_ROOT = "DATA_ROOT"


# パストラバーサル対策: raw CSV ファイルの許可ベースディレクトリ
# 環境変数 DATA_ROOT で指定可能（未設定時はカレントワーキングディレクトリ）
def _get_allowed_base_dirs() -> list[pathlib.Path]:
    """許可されたベースディレクトリリストを取得。"""
    data_root = _os.environ.get(DATA_ROOT)
    if data_root:
        return [pathlib.Path(data_root).resolve()]
    return [pathlib.Path.cwd().resolve()]


LEGACY_DELETE_PROCESSES_SUNSET_AT = datetime(2026, 6, 30, 23, 59, 59, tzinfo=UTC)
LEGACY_DELETE_PROCESSES_SUNSET = format_datetime(LEGACY_DELETE_PROCESSES_SUNSET_AT, usegmt=True)
CHARTS_FILTER_PATTERN = r"^[A-Za-z0-9_./:-]+$"
CHARTS_FILTER_MAX_LENGTH = 128
CHART_ID_PATTERN = r"^CHART_[0-9]+$"
JUDGE_LEVEL_PATTERN = r"^(OK|WARN|NG)$"
RESULT_ID_PATTERN = r"^JR_[0-9]+$"
MAX_NOTIFICATION_RETRY_COUNT = 3
NOTIFICATION_RETRY_BACKOFF_MINUTES = {1: 1, 2: 5, 3: 30}


def _legacy_delete_headers(process_id: str | None) -> dict[str, str]:
    """旧 DELETE `/processes` の移行ヘッダを生成する。"""
    if process_id is None:
        link_target = "/processes"
    else:
        link_target = f"/processes/{quote(process_id, safe='')}"
    return {
        "Deprecation": "true",
        "Sunset": LEGACY_DELETE_PROCESSES_SUNSET,
        "Link": f'<{link_target}>; rel="successor-version"',
    }


def _get_or_create_runner(app: FastAPI) -> DBTaskRunner:
    """`app.state.runner` から実行中ランナーを取得し、未作成なら生成する。"""
    with _runner_lock:
        existing_runner = getattr(app.state, "runner", None)
        if isinstance(existing_runner, DBTaskRunner):
            return existing_runner

        if existing_runner is not None and hasattr(existing_runner, "stop"):
            try:
                existing_runner.stop()
            except RuntimeError:
                logger.exception("Failed to stop existing DBTaskRunner before replacement")

        runner = DBTaskRunner(main_db=MAIN_DB, temp_db=TEMP_DB)
        app.state.runner = runner
        return runner


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """FastAPI の起動/終了時に DBTaskRunner のライフサイクルを管理する。"""
    try:
        _init_schema(MAIN_DB)
        yield
    finally:
        runner: DBTaskRunner | None = None
        with _runner_lock:
            if hasattr(app.state, "runner"):
                runner = cast(DBTaskRunner, app.state.runner)

        if runner is not None:
            try:
                runner.stop()
            except RuntimeError:
                logger.exception("Failed to stop DBTaskRunner during shutdown")
            else:
                with _runner_lock:
                    if getattr(app.state, "runner", None) is runner:
                        del app.state.runner


app = FastAPI(title="db_api", version="0.1.0", lifespan=lifespan)


@app.exception_handler(RequestValidationError)
async def handle_request_validation_error(request: Request, exc: RequestValidationError):
    """FastAPI の入力バリデーション例外を共通エラーフォーマットへ変換する。"""
    logger.warning("Validation error on %s %s", request.method, request.url.path)
    issues = jsonable_encoder(exc.errors())
    return JSONResponse(
        status_code=422,
        content={
            "ok": False,
            "error": {
                "code": "VALIDATION_ERROR",
                "message": "Validation error",
                "details": {"issues": issues},
            },
        },
    )


@app.middleware("http")
async def add_legacy_delete_migration_headers(request: Request, call_next):
    """`DELETE /processes` の全レスポンスに移行ヘッダを付与する。"""
    response = await call_next(request)
    if request.method == "DELETE" and request.url.path == "/processes":
        process_id = getattr(request.state, "legacy_delete_process_id", None)
        response.headers.update(_legacy_delete_headers(process_id))
    return response


def _runner_from_request(request: Request) -> DBTaskRunner:
    """リクエストコンテキストから DBTaskRunner を取得する。"""
    return _get_or_create_runner(request.app)


def get_runner(request: Request) -> DBTaskRunner:
    """FastAPI Depends 経由で遅延初期化された DBTaskRunner を提供する。"""
    return _runner_from_request(request)


def _is_runner_unavailable_error(error: Exception) -> bool:
    """DBTaskRunner 停止/タイムアウト起因の一時的障害かを判定する。"""
    if isinstance(error, TimeoutError):
        return True
    if not isinstance(error, RuntimeError):
        return False
    return str(error).startswith("DBTaskRunner")


def _is_transient_operational_error(error: sqlite3.OperationalError) -> bool:
    """OperationalError が一時的な DB 障害かどうかを判定する。"""
    message = str(error).lower()

    # 恒久的な設定/SQL 不整合は 500 として扱う。
    non_transient_markers = (
        "no such table",
        "no such column",
        "syntax error",
        "malformed",
    )
    if any(marker in message for marker in non_transient_markers):
        return False

    transient_markers = (
        "database is locked",
        "database is busy",
        "busy",
        "unable to open database file",
        "disk i/o error",
        "readonly database",
    )
    return any(marker in message for marker in transient_markers)


def _raise_api_error(
    *,
    operation: str,
    error: Exception,
    headers: dict[str, str] | None = None,
) -> NoReturn:
    """内部例外をログに残しつつ、クライアント向けには安全なエラーを返す。"""
    logger.exception("%s failed: %s", operation, type(error).__name__)

    if _is_runner_unavailable_error(error):
        raise HTTPException(
            status_code=503,
            detail="Service temporarily unavailable",
            headers=headers,
        ) from error

    if isinstance(error, sqlite3.OperationalError):
        if _is_transient_operational_error(error):
            raise HTTPException(
                status_code=503,
                detail="Database temporarily unavailable",
                headers=headers,
            ) from error
        raise HTTPException(
            status_code=500,
            detail="Database operation failed",
            headers=headers,
        ) from error

    if isinstance(error, sqlite3.DatabaseError):
        raise HTTPException(
            status_code=500,
            detail="Database operation failed",
            headers=headers,
        ) from error

    raise HTTPException(
        status_code=500,
        detail="Internal server error",
        headers=headers,
    ) from error


RunnerDep = Annotated[DBTaskRunner, Depends(get_runner)]
_chart_repository = ChartRepository()
_judge_repository = JudgeRepository()
_governance_change_request_repository = GovernanceChangeRequestRepository()
_governance_approvals_repository = GovernanceApprovalsRepository()
_audit_event_writer = AuditEventWriter()


@app.get("/charts")
def get_charts(
    tool_id: str | None = Query(
        default=None,
        min_length=1,
        max_length=CHARTS_FILTER_MAX_LENGTH,
        pattern=CHARTS_FILTER_PATTERN,
    ),
    chamber_id: str | None = Query(
        default=None,
        min_length=1,
        max_length=CHARTS_FILTER_MAX_LENGTH,
        pattern=CHARTS_FILTER_PATTERN,
    ),
    recipe_id: str | None = Query(
        default=None,
        min_length=1,
        max_length=CHARTS_FILTER_MAX_LENGTH,
        pattern=CHARTS_FILTER_PATTERN,
    ),
    parameter: str | None = Query(
        default=None,
        min_length=1,
        max_length=CHARTS_FILTER_MAX_LENGTH,
        pattern=CHARTS_FILTER_PATTERN,
    ),
    step_no: int | None = Query(default=None, ge=0),
    feature_type: str | None = Query(
        default=None,
        min_length=1,
        max_length=CHARTS_FILTER_MAX_LENGTH,
        pattern=CHARTS_FILTER_PATTERN,
    ),
    active_only: bool = False,
):
    """Chart 定義一覧を返す。"""
    criteria = ChartsQueryCriteria(
        tool_id=tool_id,
        chamber_id=chamber_id,
        recipe_id=recipe_id,
        parameter=parameter,
        step_no=step_no,
        feature_type=feature_type,
        active_only=active_only,
    )
    try:
        rows = _chart_repository.find_charts(criteria)
        return {"ok": True, "data": [asdict(row) for row in rows]}
    except Exception as e:
        _raise_api_error(operation="GET /charts", error=e)


@app.get("/charts/active")
def get_active_charts(
    tool_id: str | None = Query(
        default=None,
        min_length=1,
        max_length=CHARTS_FILTER_MAX_LENGTH,
        pattern=CHARTS_FILTER_PATTERN,
    ),
    chamber_id: str | None = Query(
        default=None,
        min_length=1,
        max_length=CHARTS_FILTER_MAX_LENGTH,
        pattern=CHARTS_FILTER_PATTERN,
    ),
    recipe_id: str | None = Query(
        default=None,
        min_length=1,
        max_length=CHARTS_FILTER_MAX_LENGTH,
        pattern=CHARTS_FILTER_PATTERN,
    ),
):
    """active chart set と有効閾値一覧を返す。"""
    criteria = ActiveChartsQueryCriteria(
        tool_id=tool_id,
        chamber_id=chamber_id,
        recipe_id=recipe_id,
    )
    try:
        data = _chart_repository.find_active_chart_set(criteria)
        return {"ok": True, "data": asdict(data)}
    except Exception as e:
        _raise_api_error(operation="GET /charts/active", error=e)


def _normalize_query_datetime(raw: datetime | None) -> str | None:
    """履歴検索用の datetime クエリを SQLite 比較用 ISO 文字列へ変換する。"""
    if raw is None:
        return None
    if raw.tzinfo is None:
        raise HTTPException(
            status_code=400,
            detail="from_ts and to_ts must be timezone-aware datetimes",
        )
    return to_utc_millis(raw.isoformat())


def _validate_query_datetime_range(
    from_ts: datetime | None,
    to_ts: datetime | None,
    *,
    require_pair: bool,
) -> None:
    """from_ts/to_ts の指定整合と範囲整合を検証する。"""
    if require_pair and (from_ts is None) != (to_ts is None):
        raise HTTPException(
            status_code=400,
            detail="from_ts and to_ts must be specified together",
        )

    if from_ts is not None and to_ts is None:
        if from_ts.tzinfo is None:
            raise HTTPException(
                status_code=400,
                detail="from_ts and to_ts must be timezone-aware datetimes",
            )
        return

    if to_ts is not None and from_ts is None:
        if to_ts.tzinfo is None:
            raise HTTPException(
                status_code=400,
                detail="from_ts and to_ts must be timezone-aware datetimes",
            )
        return

    if from_ts is None or to_ts is None:
        return

    try:
        validate_timestamp_range(from_ts, to_ts)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


def _parse_chart_pk(chart_id: str | None) -> int | None:
    """`CHART_<id>` 形式の chart_id を int PK へ変換する。"""
    if chart_id is None:
        return None

    try:
        numeric_part = chart_id.split("_", maxsplit=1)[1]
        if not numeric_part.isdigit():
            raise ValueError("chart_id numeric part must contain only digits")
        chart_pk = int(numeric_part)
        if chart_pk < 1:
            raise ValueError("chart_id must be greater than or equal to 1")
        if not (-(2**63) <= chart_pk <= 2**63 - 1):
            raise ValueError("chart_id out of int64 range")
        return chart_pk
    except (ValueError, OverflowError, IndexError) as exc:
        raise HTTPException(status_code=400, detail="Invalid chart_id") from exc


def _parse_result_pk(result_id: str) -> int:
    """`JR_<id>` 形式の result_id を int PK へ変換する。"""
    try:
        numeric_part = result_id.split("_", maxsplit=1)[1]
        if not numeric_part.isdigit():
            raise ValueError("result_id numeric part must contain only digits")
        result_pk = int(numeric_part)
        if result_pk < 1:
            raise ValueError("result_id must be greater than or equal to 1")
        if not (-(2**63) <= result_pk <= 2**63 - 1):
            raise ValueError("result_id out of int64 range")
        return result_pk
    except (ValueError, OverflowError, IndexError) as exc:
        raise HTTPException(status_code=400, detail="Invalid result_id") from exc


def _not_found_error_response(*, message: str, details: dict[str, str]) -> JSONResponse:
    """契約準拠の 404 error envelope を返す。"""
    return JSONResponse(
        status_code=404,
        content={
            "ok": False,
            "error": {
                "code": "NOT_FOUND",
                "message": message,
                "details": details,
            },
        },
    )


def _duplicate_idempotency_error_response(*, idempotency_key: str) -> JSONResponse:
    """重複 idempotency_key への 409 error envelope を返す。"""
    return JSONResponse(
        status_code=409,
        content={
            "ok": False,
            "error": {
                "code": "DUPLICATE_IDEMPOTENCY_KEY",
                "message": "idempotency_key already exists",
                "details": {"idempotency_key": idempotency_key},
            },
        },
    )


def _conflict_error_response(*, code: str, message: str, details: dict[str, str]) -> JSONResponse:
    """契約準拠の 409 error envelope を返す。"""
    return JSONResponse(
        status_code=409,
        content={
            "ok": False,
            "error": {
                "code": code,
                "message": message,
                "details": details,
            },
        },
    )


def _bad_request_error_response(
    *, code: str, message: str, details: dict[str, str]
) -> JSONResponse:
    """契約準拠の 400 error envelope を返す。"""
    return JSONResponse(
        status_code=400,
        content={
            "ok": False,
            "error": {
                "code": code,
                "message": message,
                "details": details,
            },
        },
    )


def _validation_error_response(*, issues: list[dict[str, object]]) -> JSONResponse:
    """契約準拠の 422 validation error envelope を返す。"""
    return JSONResponse(
        status_code=422,
        content={
            "ok": False,
            "error": {
                "code": "VALIDATION_ERROR",
                "message": "Validation error",
                "details": {"issues": issues},
            },
        },
    )


def _is_duplicate_change_request_idempotency_error(error: sqlite3.IntegrityError) -> bool:
    """GovernanceChangeRequests.idempotency_key の UNIQUE 違反かを判定する。"""
    message = str(error)
    return (
        "GovernanceChangeRequests.idempotency_key" in message
        or "idx_change_requests_idempotency" in message
    )


def _is_governance_change_request_chart_fk_error(error: sqlite3.IntegrityError) -> bool:
    """GovernanceChangeRequests.chart_id の外部キー制約違反かを判定する。"""
    return "foreign key constraint failed" in str(error).lower()


class _GovernanceChangeRequestIdempotencyConflict(Exception):
    """change-request 作成時の重複 idempotency_key を表す内部例外。"""


class _GovernanceChangeRequestChartFkViolation(Exception):
    """change-request 作成時の chart_id 外部キー違反を表す内部例外。"""


class _GovernanceApproveAlreadyApproved(Exception):
    """approve 対象が既に approved の場合に送出する内部例外。"""


class _GovernanceApproveInvalidState(Exception):
    """approve 対象の status が pending 以外の場合に送出する内部例外。"""


class _GovernanceApplyInvalidState(Exception):
    """apply 対象の status が approved 以外の場合に送出する内部例外。"""


class _GovernanceApplyChartNotFound(Exception):
    """apply 対象 chart が存在しない場合に送出する内部例外。"""


class _GovernanceApplyVersionConflict(Exception):
    """expected_version と current.version が不一致の場合に送出する内部例外。"""

    def __init__(self, *, current_version: int, current_updated_at: str, chart_id: int) -> None:
        self.current_version = current_version
        self.current_updated_at = current_updated_at
        self.chart_id = chart_id


class _GovernanceApplyValidationError(Exception):
    """apply payload がしきい値整合性を満たさない場合に送出する内部例外。"""

    def __init__(self, *, message: str) -> None:
        self.message = message


class _GovernanceNotificationNotFound(Exception):
    """notification outbox が存在しない場合に送出する内部例外。"""


class _GovernanceNotificationInvalidState(Exception):
    """retry 対象の status が failed 以外の場合に送出する内部例外。"""

    def __init__(self, *, status: str) -> None:
        self.status = status


class _GovernanceNotificationRetryLimitExceeded(Exception):
    """retry_count が上限に達している場合に送出する内部例外。"""

    def __init__(self, *, retry_count: int) -> None:
        self.retry_count = retry_count


class _GovernanceNotificationConcurrentModification(Exception):
    """conditional UPDATE が 0 件更新になった場合に送出する内部例外。"""


def _compute_notification_next_retry_at(*, base_time: datetime, retry_count: int) -> str:
    """retry_count に応じた次回試行時刻を UTC ミリ秒 ISO 文字列で返す。"""
    minutes = NOTIFICATION_RETRY_BACKOFF_MINUTES[retry_count]
    return to_utc_millis((base_time + timedelta(minutes=minutes)).isoformat())


def _parse_threshold_patch(change_payload: str) -> dict[str, float | None]:
    """change_payload からしきい値更新パッチを抽出する。"""
    payload = json.loads(change_payload)
    if not isinstance(payload, dict):
        raise _GovernanceApplyValidationError(message="change_payload must be an object")

    aliases = {
        "warn_low": "warn_low",
        "warning_lcl": "warn_low",
        "warn_high": "warn_high",
        "warning_ucl": "warn_high",
        "crit_low": "crit_low",
        "critical_lcl": "crit_low",
        "crit_high": "crit_high",
        "critical_ucl": "crit_high",
    }
    patch: dict[str, float | None] = {}
    for key, value in payload.items():
        mapped = aliases.get(str(key))
        if mapped is None:
            continue
        if value is None:
            patch[mapped] = None
            continue
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise _GovernanceApplyValidationError(message=f"{key} must be a number or null")
        numeric = float(value)
        if numeric != numeric or numeric in (float("inf"), float("-inf")):
            raise _GovernanceApplyValidationError(message=f"{key} must be finite")
        patch[mapped] = numeric
    return patch


def _validate_threshold_consistency(
    *,
    warn_low: float | None,
    warn_high: float | None,
    crit_low: float | None,
    crit_high: float | None,
) -> None:
    """しきい値の大小関係を検証する。"""
    if warn_low is not None and warn_high is not None and warn_low > warn_high:
        raise _GovernanceApplyValidationError(
            message="warn_low must be less than or equal to warn_high"
        )
    if crit_low is not None and crit_high is not None and crit_low > crit_high:
        raise _GovernanceApplyValidationError(
            message="crit_low must be less than or equal to crit_high"
        )
    if crit_low is not None and warn_low is not None and crit_low > warn_low:
        raise _GovernanceApplyValidationError(
            message="crit_low must be less than or equal to warn_low"
        )
    if warn_high is not None and crit_high is not None and warn_high > crit_high:
        raise _GovernanceApplyValidationError(
            message="warn_high must be less than or equal to crit_high"
        )
    if crit_low is not None and warn_high is not None and crit_low > warn_high:
        raise _GovernanceApplyValidationError(
            message="crit_low must be less than or equal to warn_high"
        )


def _build_waveform_preview(process_id: str, limit: int) -> dict[str, object]:
    """ProcessInfo.raw_csv_path からドリルダウン表示用の波形プレビューを返す。

    READ-ONLY 接続を使用することで、書き込み作業中の lock 競合を回避する。
    """
    con = _connect_readonly(MAIN_DB)
    try:
        row = con.execute(
            "SELECT raw_csv_path FROM ProcessInfo WHERE process_id = ?",
            (process_id,),
        ).fetchone()
    finally:
        con.close()

    if row is None:
        raise HTTPException(status_code=404, detail="process not found")

    raw_path = row[0]
    if raw_path is None:
        return {
            "process_id": process_id,
            "source_path": None,
            "points": [],
        }

    src = pathlib.Path(str(raw_path))
    if not src.is_absolute():
        src = pathlib.Path.cwd() / src

    # パストラバーサル対策: ベースディレクトリ外のアクセスを拒否
    src_resolved = src.resolve()
    allowed_base_dirs = _get_allowed_base_dirs()
    is_allowed = any(
        src_resolved == allowed_dir or str(src_resolved).startswith(str(allowed_dir) + _os.sep)
        for allowed_dir in allowed_base_dirs
    )
    if not is_allowed:
        src_name = src_resolved.name
        logger.warning(
            "Access attempt outside allowed directories: process_id=%s, path=%s, allowed_dirs=%s",
            process_id,
            src_resolved.as_posix(),
            ", ".join(d.as_posix() for d in allowed_base_dirs),
        )
        raise HTTPException(
            status_code=403,
            detail=(
                "Access to files outside the allowed base directories is forbidden: "
                f"process_id={process_id}, file={src_name}"
            ),
        )

    if not src_resolved.exists():
        return {
            "process_id": process_id,
            "source_path": src_resolved.as_posix(),
            "points": [],
        }

    try:
        import pandas as pd  # Local import to avoid global import cost.

        with src_resolved.open("r", encoding="utf-8", errors="ignore") as f:
            start_row = 0
            for idx, line in enumerate(f):
                if idx >= 200:
                    break
                if line.strip().upper().startswith("DATA"):
                    start_row = idx + 1
                    break
            # DATA行の直後からpandasで読み込む
            f.seek(0)
            for _ in range(start_row):
                next(f)
            try:
                frame = pd.read_csv(f)
            except pd.errors.ParserError as e:
                logger.error(
                    "CSV parse error in waveform preview for process_id=%s source_path=%s: %s",
                    process_id,
                    src_resolved.as_posix(),
                    str(e),
                )
                return {
                    "process_id": process_id,
                    "source_path": src_resolved.as_posix(),
                    "points": [],
                }
        if frame.empty:
            return {
                "process_id": process_id,
                "source_path": src_resolved.as_posix(),
                "points": [],
            }

        x_col = frame.columns[0]
        y_col = None
        for col in frame.columns[1:]:
            if pd.api.types.is_numeric_dtype(frame[col]):
                y_col = col
                break
        if y_col is None:
            return {
                "process_id": process_id,
                "source_path": src_resolved.as_posix(),
                "points": [],
            }

        sample = frame[[x_col, y_col]].tail(limit)
        points = [
            {
                "x": str(x),
                "y": None if pd.isna(y) else float(y),
            }
            for x, y in sample.to_records(index=False)
        ]
        return {
            "process_id": process_id,
            "source_path": src_resolved.as_posix(),
            "points": points,
        }
    except Exception:
        logger.exception(
            "Failed to build waveform preview for process_id=%s source_path=%s",
            process_id,
            src_resolved.as_posix(),
        )
        return {
            "process_id": process_id,
            "source_path": src_resolved.as_posix(),
            "points": [],
        }


@app.get("/charts/history")
def get_charts_history(
    chart_id: str | None = Query(
        default=None,
        min_length=1,
        max_length=64,
        pattern=CHART_ID_PATTERN,
    ),
    chart_set_id: int | None = Query(default=None, ge=1),
    change_source: str | None = Query(
        default=None,
        min_length=1,
        max_length=CHARTS_FILTER_MAX_LENGTH,
        pattern=CHARTS_FILTER_PATTERN,
    ),
    from_ts: Annotated[datetime | None, Query()] = None,
    to_ts: Annotated[datetime | None, Query()] = None,
    limit: int = Query(default=100, ge=1, le=500),
    offset: int = Query(default=0, ge=0),
):
    """Chart 閾値変更履歴を返す。"""
    _validate_query_datetime_range(from_ts, to_ts, require_pair=False)

    chart_pk = _parse_chart_pk(chart_id)
    criteria = ChartsHistoryQueryCriteria(
        chart_pk=chart_pk,
        chart_set_id=chart_set_id,
        change_source=change_source,
        from_ts=_normalize_query_datetime(from_ts),
        to_ts=_normalize_query_datetime(to_ts),
        limit=limit,
        offset=offset,
    )

    try:
        rows = _chart_repository.find_chart_history(criteria)
        return {"ok": True, "data": [asdict(row) for row in rows]}
    except Exception as e:
        _raise_api_error(operation="GET /charts/history", error=e)


@app.get("/charts/{chart_id}/points")
def get_chart_points(
    chart_id: str = Path(
        min_length=1,
        max_length=64,
        pattern=CHART_ID_PATTERN,
    ),
    limit: int = Query(default=50, ge=1, le=500),
):
    """指定 chart に対応する最新特徴量点を返す。"""
    chart_pk = _parse_chart_pk(chart_id)
    if chart_pk is None:
        raise HTTPException(status_code=400, detail="Invalid chart_id")

    try:
        rows = _chart_repository.find_chart_points(chart_pk, limit)
        return {"ok": True, "data": [asdict(row) for row in rows]}
    except Exception as e:
        _raise_api_error(operation="GET /charts/{chart_id}/points", error=e)


@app.get("/processes/{process_id}/waveform-preview")
def get_process_waveform_preview(
    process_id: str = Path(
        min_length=1, max_length=CHARTS_FILTER_MAX_LENGTH, pattern=CHARTS_FILTER_PATTERN
    ),
    limit: int = Query(default=300, ge=10, le=2000),
):
    """process_id に紐づく元波形（raw_csv_path）のプレビューを返す。"""
    try:
        data = _build_waveform_preview(process_id, limit)
        return {"ok": True, "data": data}
    except HTTPException:
        raise
    except Exception as e:
        _raise_api_error(operation="GET /processes/{process_id}/waveform-preview", error=e)


@app.get("/judge/results")
def get_judge_results(
    chart_id: str | None = Query(
        default=None,
        min_length=1,
        max_length=64,
        pattern=CHART_ID_PATTERN,
    ),
    process_id: str | None = Query(
        default=None,
        min_length=1,
        max_length=CHARTS_FILTER_MAX_LENGTH,
        pattern=CHARTS_FILTER_PATTERN,
    ),
    lot_id: str | None = Query(
        default=None,
        min_length=1,
        max_length=CHARTS_FILTER_MAX_LENGTH,
        pattern=CHARTS_FILTER_PATTERN,
    ),
    recipe_id: str | None = Query(
        default=None,
        min_length=1,
        max_length=CHARTS_FILTER_MAX_LENGTH,
        pattern=CHARTS_FILTER_PATTERN,
    ),
    level: str | None = Query(default=None, pattern=JUDGE_LEVEL_PATTERN),
    from_ts: Annotated[datetime | None, Query()] = None,
    to_ts: Annotated[datetime | None, Query()] = None,
    limit: int = Query(default=200, ge=1, le=1000),
    offset: int = Query(default=0, ge=0),
):
    """判定結果一覧を返す。"""
    _validate_query_datetime_range(from_ts, to_ts, require_pair=True)

    criteria = JudgeResultsQueryCriteria(
        chart_id=chart_id,
        process_id=process_id,
        lot_id=lot_id,
        recipe_id=recipe_id,
        level=level,
        from_ts=_normalize_query_datetime(from_ts),
        to_ts=_normalize_query_datetime(to_ts),
        limit=limit,
        offset=offset,
    )

    try:
        rows = _judge_repository.find_results(criteria)
        return {"ok": True, "data": [asdict(row) for row in rows]}
    except Exception as e:
        _raise_api_error(operation="GET /judge/results", error=e)


@app.get("/judge/results/{result_id}")
def get_judge_result_by_id(
    result_id: str = Path(
        min_length=1,
        max_length=64,
        pattern=RESULT_ID_PATTERN,
    ),
):
    """判定結果詳細を返す。"""
    result_pk = _parse_result_pk(result_id)

    try:
        row = _judge_repository.find_result_by_id(result_pk)
        if row is None:
            return _not_found_error_response(
                message="judge result not found",
                details={"result_id": result_id},
            )
        return {"ok": True, "data": asdict(row)}
    except JudgeDataCorruptionError as e:
        logger.error(
            "JUDGE_DATA_CORRUPTION: GET /judge/results/{result_id} failed "
            "(requested_result_id=%s, result_pk=%s)",
            result_id,
            result_pk,
            exc_info=True,
        )
        raise HTTPException(status_code=500, detail="Internal server error") from e
    except Exception as e:
        _raise_api_error(operation="GET /judge/results/{result_id}", error=e)


@app.get("/governance/change-requests")
def get_governance_change_requests(
    query: Annotated[ChangeRequestsQuery, Depends()],
):
    """ガバナンス変更申請一覧を返す。"""
    con = _connect_readonly(MAIN_DB)
    try:
        rows = _governance_change_request_repository.list(
            con,
            status=query.status,
            chart_id=query.chart_id,
            from_ts=_normalize_query_datetime(query.from_ts),
            to_ts=_normalize_query_datetime(query.to_ts),
            limit=query.limit,
            offset=query.offset,
        )
        return {"ok": True, "data": [asdict(row) for row in rows]}
    except Exception as e:
        _raise_api_error(operation="GET /governance/change-requests", error=e)
    finally:
        con.close()


@app.get("/governance/audit-events")
def get_governance_audit_events(
    query: Annotated[GovernanceAuditEventsQuery, Depends()],
):
    """ガバナンス監査イベント一覧を返す。"""
    con = _connect_readonly(MAIN_DB)
    try:
        _validate_query_datetime_range(query.from_ts, query.to_ts, require_pair=False)

        sql = """
            SELECT
                id, event_type, actor, actor_role, target_type, target_id,
                occurred_at, before_json, after_json, correlation_id
            FROM GovernanceAuditEvents
        """
        where_clauses: list[str] = []
        params: list[object] = []

        if query.event_type is not None:
            where_clauses.append("event_type = ?")
            params.append(query.event_type)
        if query.target_type is not None:
            where_clauses.append("target_type = ?")
            params.append(query.target_type)
        if query.target_id is not None:
            where_clauses.append("target_id = ?")
            params.append(query.target_id)
        if query.from_ts is not None:
            where_clauses.append("occurred_at >= ?")
            params.append(_normalize_query_datetime(query.from_ts))
        if query.to_ts is not None:
            where_clauses.append("occurred_at <= ?")
            params.append(_normalize_query_datetime(query.to_ts))

        if where_clauses:
            sql += " WHERE " + " AND ".join(where_clauses)

        sql += " ORDER BY occurred_at DESC, id DESC LIMIT ? OFFSET ?"
        params.extend((query.limit, query.offset))

        rows = con.execute(sql, params).fetchall()
        data = [
            {
                "id": int(row[0]),
                "event_type": str(row[1]),
                "actor": str(row[2]),
                "actor_role": str(row[3]),
                "target_type": str(row[4]),
                "target_id": int(row[5]),
                "occurred_at": str(row[6]),
                "before_json": row[7],
                "after_json": row[8],
                "correlation_id": row[9],
            }
            for row in rows
        ]
        return {"ok": True, "data": data}
    except HTTPException:
        raise
    except Exception as e:
        _raise_api_error(operation="GET /governance/audit-events", error=e)
    finally:
        con.close()


@app.post("/governance/notifications/{event_id}/retry")
def retry_governance_notification(
    runner: RunnerDep,
    event_id: int = Path(ge=1),
):
    """通知 outbox の failed レコードを再送キューへ戻す。"""

    def _write() -> dict[str, object]:
        con = _connect(MAIN_DB)
        try:
            con.execute("BEGIN")
            row = con.execute(
                """
                SELECT id, status, retry_count
                FROM GovernanceNotificationOutbox
                WHERE event_id = ?
                """,
                (event_id,),
            ).fetchone()
            if row is None:
                raise _GovernanceNotificationNotFound

            outbox_id = int(row[0])
            status = str(row[1])
            retry_count = int(row[2])

            if status != "failed":
                raise _GovernanceNotificationInvalidState(status=status)
            if retry_count >= MAX_NOTIFICATION_RETRY_COUNT:
                raise _GovernanceNotificationRetryLimitExceeded(retry_count=retry_count)

            now = datetime.now(UTC)
            now_iso = to_utc_millis(now.isoformat())
            next_retry_at = _compute_notification_next_retry_at(
                base_time=now,
                retry_count=retry_count + 1,
            )

            update_cur = con.execute(
                """
                UPDATE GovernanceNotificationOutbox
                SET status = 'pending',
                    retry_count = retry_count + 1,
                    next_retry_at = ?,
                    last_attempt_at = ?,
                    last_error = NULL
                WHERE id = ?
                  AND status = 'failed'
                  AND retry_count = ?
                """,
                (next_retry_at, now_iso, outbox_id, retry_count),
            )
            if update_cur.rowcount == 0:
                latest = con.execute(
                    """
                    SELECT status, retry_count
                    FROM GovernanceNotificationOutbox
                    WHERE id = ?
                    """,
                    (outbox_id,),
                ).fetchone()
                if latest is None:
                    raise _GovernanceNotificationNotFound

                latest_status = str(latest[0])
                latest_retry_count = int(latest[1])
                if latest_status != "failed":
                    raise _GovernanceNotificationInvalidState(status=latest_status)
                if latest_retry_count >= MAX_NOTIFICATION_RETRY_COUNT:
                    raise _GovernanceNotificationRetryLimitExceeded(retry_count=latest_retry_count)
                raise _GovernanceNotificationConcurrentModification

            _audit_event_writer.write(
                con,
                event_type="notification_queued",
                actor="ops",
                actor_role="ops",
                target_type="notification",
                target_id=outbox_id,
                occurred_at=now_iso,
                correlation_id=f"event:{event_id}",
            )
            con.commit()
            return {
                "event_id": event_id,
                "status": "pending",
                "retry_count": retry_count + 1,
                "next_retry_at": next_retry_at,
            }
        except Exception:
            con.rollback()
            raise
        finally:
            con.close()

    try:
        data = runner.submit("write", _write)
        return {"ok": True, "data": data}
    except _GovernanceNotificationNotFound:
        return _not_found_error_response(
            message="notification outbox not found",
            details={"event_id": str(event_id)},
        )
    except _GovernanceNotificationInvalidState as e:
        return _bad_request_error_response(
            code="INVALID_RETRY_TARGET",
            message="only failed notification can be retried",
            details={"event_id": str(event_id), "current_status": e.status},
        )
    except _GovernanceNotificationRetryLimitExceeded as e:
        return _conflict_error_response(
            code="RETRY_LIMIT_EXCEEDED",
            message="retry_count has reached the maximum",
            details={
                "event_id": str(event_id),
                "retry_count": str(e.retry_count),
                "max_retry_count": str(MAX_NOTIFICATION_RETRY_COUNT),
            },
        )
    except _GovernanceNotificationConcurrentModification:
        return _conflict_error_response(
            code="CONCURRENT_MODIFICATION",
            message="notification outbox was modified concurrently",
            details={"event_id": str(event_id)},
        )
    except Exception as e:
        _raise_api_error(operation="POST /governance/notifications/{event_id}/retry", error=e)


@app.post("/governance/change-requests")
def create_governance_change_request(payload: ChangeRequestIn, runner: RunnerDep):
    """ガバナンス変更申請を作成する。"""

    proposed_at = to_utc_millis(datetime.now(UTC).isoformat())

    def _write() -> dict[str, int | str]:
        con = _connect(MAIN_DB)
        try:
            con.execute("BEGIN")
            try:
                request_id = _governance_change_request_repository.create(
                    con,
                    chart_id=payload.chart_id,
                    proposed_by=payload.proposed_by,
                    proposed_at=proposed_at,
                    change_payload=payload.change_payload,
                    expected_version=payload.expected_version,
                    idempotency_key=payload.idempotency_key,
                )
            except sqlite3.IntegrityError as e:
                if _is_duplicate_change_request_idempotency_error(e):
                    raise _GovernanceChangeRequestIdempotencyConflict from e
                if _is_governance_change_request_chart_fk_error(e):
                    raise _GovernanceChangeRequestChartFkViolation from e
                raise

            _audit_event_writer.write(
                con,
                event_type="change_requested",
                actor=payload.proposed_by,
                actor_role="requester",
                target_type="change_request",
                target_id=request_id,
                occurred_at=proposed_at,
                correlation_id=payload.idempotency_key,
            )

            row = _governance_change_request_repository.find_by_id(con, request_id)
            con.commit()
            return {"request_id": request_id, "status": row.status}
        except Exception:
            con.rollback()
            raise
        finally:
            con.close()

    try:
        data = runner.submit("write", _write)
        return {"ok": True, "data": data}
    except _GovernanceChangeRequestIdempotencyConflict:
        return _duplicate_idempotency_error_response(
            idempotency_key=payload.idempotency_key,
        )
    except _GovernanceChangeRequestChartFkViolation:
        return _validation_error_response(
            issues=[
                {
                    "loc": ["body", "chart_id"],
                    "msg": "chart_id must reference an existing chart",
                    "type": "value_error",
                }
            ]
        )
    except Exception as e:
        _raise_api_error(operation="POST /governance/change-requests", error=e)


@app.post("/governance/change-requests/{request_id}/approve")
def approve_governance_change_request(
    payload: ChangeRequestApproveIn,
    runner: RunnerDep,
    request_id: int = Path(ge=1),
):
    """ガバナンス変更申請を承認する。"""

    approved_at = to_utc_millis(datetime.now(UTC).isoformat())

    def _write() -> dict[str, int | str]:
        con = _connect(MAIN_DB)
        try:
            con.execute("BEGIN")
            row = _governance_change_request_repository.find_by_id(con, request_id)

            if row.status == "approved":
                raise _GovernanceApproveAlreadyApproved
            if row.status != "pending":
                raise _GovernanceApproveInvalidState

            _governance_approvals_repository.create(
                con,
                request_id=request_id,
                approved_by=payload.approved_by,
                approved_by_role=payload.approved_by_role,
                approved_at=approved_at,
                comment=payload.comment,
            )
            _governance_change_request_repository.update_status(
                con,
                record_id=request_id,
                new_status="approved",
            )
            _audit_event_writer.write(
                con,
                event_type="change_approved",
                actor=payload.approved_by,
                actor_role=payload.approved_by_role,
                target_type="change_request",
                target_id=request_id,
                occurred_at=approved_at,
                correlation_id=f"request:{request_id}",
            )
            con.commit()
            return {"request_id": request_id, "status": "approved"}
        except Exception:
            con.rollback()
            raise
        finally:
            con.close()

    try:
        data = runner.submit("write", _write)
        return {"ok": True, "data": data}
    except GovernanceNotFoundError:
        return _not_found_error_response(
            message="change request not found",
            details={"request_id": str(request_id)},
        )
    except _GovernanceApproveAlreadyApproved:
        return _conflict_error_response(
            code="ALREADY_APPROVED",
            message="change request is already approved",
            details={"request_id": str(request_id)},
        )
    except _GovernanceApproveInvalidState:
        return _conflict_error_response(
            code="INVALID_STATUS_TRANSITION",
            message="only pending change request can be approved",
            details={"request_id": str(request_id)},
        )
    except Exception as e:
        _raise_api_error(
            operation="POST /governance/change-requests/{request_id}/approve",
            error=e,
        )


@app.post("/governance/change-requests/{request_id}/apply")
def apply_governance_change_request(
    payload: ChangeRequestApplyIn,
    runner: RunnerDep,
    request_id: int = Path(ge=1),
):
    """承認済みのガバナンス変更申請を反映する。"""

    applied_at = to_utc_millis(datetime.now(UTC).isoformat())

    def _write() -> dict[str, object]:
        con = _connect(MAIN_DB)
        try:
            con.execute("BEGIN")
            req = _governance_change_request_repository.find_by_id(con, request_id)
            if req.status != "approved":
                raise _GovernanceApplyInvalidState

            chart_row = con.execute(
                """
                SELECT
                    id, chart_set_id, tool_id, chamber_id, recipe_id, parameter,
                    step_no, feature_type, warn_low, warn_high, crit_low, crit_high,
                    version, updated_at
                FROM ChartsV2
                WHERE id = ?
                """,
                (req.chart_id,),
            ).fetchone()
            if chart_row is None:
                raise _GovernanceApplyChartNotFound

            (
                chart_id,
                chart_set_id,
                tool_id,
                chamber_id,
                recipe_id,
                parameter,
                step_no,
                feature_type,
                old_warn_low,
                old_warn_high,
                old_crit_low,
                old_crit_high,
                current_version,
                current_updated_at,
            ) = chart_row

            if int(current_version) != int(req.expected_version):
                raise _GovernanceApplyVersionConflict(
                    current_version=int(current_version),
                    current_updated_at=str(current_updated_at),
                    chart_id=int(chart_id),
                )

            patch = _parse_threshold_patch(req.change_payload)
            new_warn_low = patch.get("warn_low", old_warn_low)
            new_warn_high = patch.get("warn_high", old_warn_high)
            new_crit_low = patch.get("crit_low", old_crit_low)
            new_crit_high = patch.get("crit_high", old_crit_high)

            _validate_threshold_consistency(
                warn_low=new_warn_low,
                warn_high=new_warn_high,
                crit_low=new_crit_low,
                crit_high=new_crit_high,
            )

            def _thresh_eq(a: float | None, b: float | None) -> bool:
                if a is None and b is None:
                    return True
                if a is None or b is None:
                    return False
                return math.isclose(a, b, rel_tol=1e-9, abs_tol=1e-12)

            is_noop = (
                _thresh_eq(old_warn_low, new_warn_low)
                and _thresh_eq(old_warn_high, new_warn_high)
                and _thresh_eq(old_crit_low, new_crit_low)
                and _thresh_eq(old_crit_high, new_crit_high)
            )

            resulting_version = int(current_version)
            if not is_noop:
                resulting_version = int(current_version) + 1
                con.execute(
                    """
                    UPDATE ChartsV2
                    SET warn_low = ?, warn_high = ?, crit_low = ?, crit_high = ?,
                        version = ?, updated_at = ?, updated_by = ?,
                        update_reason = ?, update_source = ?
                    WHERE id = ?
                    """,
                    (
                        new_warn_low,
                        new_warn_high,
                        new_crit_low,
                        new_crit_high,
                        resulting_version,
                        applied_at,
                        payload.applied_by,
                        payload.reason,
                        "governance_apply",
                        chart_id,
                    ),
                )
                con.execute(
                    """
                    INSERT INTO ChartsHistory(
                        chart_set_id, tool_id, chamber_id, recipe_id, parameter,
                        step_no, feature_type,
                        old_warn_low, old_warn_high, old_crit_low, old_crit_high,
                        new_warn_low, new_warn_high, new_crit_low, new_crit_high,
                        changed_at, changed_by, change_reason, change_source, chart_id
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        chart_set_id,
                        tool_id,
                        chamber_id,
                        recipe_id,
                        parameter,
                        step_no,
                        feature_type,
                        old_warn_low,
                        old_warn_high,
                        old_crit_low,
                        old_crit_high,
                        new_warn_low,
                        new_warn_high,
                        new_crit_low,
                        new_crit_high,
                        applied_at,
                        payload.applied_by,
                        payload.reason,
                        "governance_apply",
                        chart_id,
                    ),
                )

            con.execute(
                """
                INSERT OR REPLACE INTO GovernanceApplyResults(
                    request_id, applied_at, success, resulting_version,
                    error_code, error_message
                ) VALUES (?, ?, 1, ?, NULL, NULL)
                """,
                (request_id, applied_at, resulting_version),
            )
            _governance_change_request_repository.update_status(
                con,
                record_id=request_id,
                new_status="applied",
            )
            _audit_event_writer.write(
                con,
                event_type="change_applied",
                actor=payload.applied_by,
                actor_role=payload.applied_by_role,
                target_type="change_request",
                target_id=request_id,
                occurred_at=applied_at,
                correlation_id=f"request:{request_id}",
            )
            con.commit()
            return {
                "request_id": request_id,
                "status": "applied",
                "resulting_version": resulting_version,
                "noop": is_noop,
            }
        except Exception:
            con.rollback()
            raise
        finally:
            con.close()

    try:
        data = runner.submit("write", _write)
        return {"ok": True, "data": data}
    except GovernanceNotFoundError:
        return _not_found_error_response(
            message="change request not found",
            details={"request_id": str(request_id)},
        )
    except _GovernanceApplyInvalidState:
        return _conflict_error_response(
            code="INVALID_STATUS_TRANSITION",
            message="only approved change request can be applied",
            details={"request_id": str(request_id)},
        )
    except _GovernanceApplyChartNotFound:
        return _bad_request_error_response(
            code="CHART_NOT_FOUND",
            message="target chart not found",
            details={"request_id": str(request_id)},
        )
    except _GovernanceApplyVersionConflict as e:
        return JSONResponse(
            status_code=409,
            content={
                "ok": False,
                "error": {
                    "code": "STALE_EXPECTED_VERSION",
                    "message": "expected_version does not match current version",
                    "details": {
                        "request_id": str(request_id),
                        "current": {
                            "chart_id": e.chart_id,
                            "version": e.current_version,
                            "updated_at": e.current_updated_at,
                        },
                    },
                },
            },
        )
    except _GovernanceApplyValidationError as e:
        return _validation_error_response(
            issues=[
                {
                    "loc": ["body", "change_payload"],
                    "msg": e.message,
                    "type": "value_error",
                }
            ]
        )
    except Exception as e:
        _raise_api_error(
            operation="POST /governance/change-requests/{request_id}/apply",
            error=e,
        )


@app.post("/processes")
def create_process(p: ProcessInfoIn, runner: RunnerDep):
    """1 件の ProcessInfo をキュー経由で保存する。"""
    try:
        runner.submit("write", lambda: write_process(p))
        return {"ok": True}
    except Exception as e:
        _raise_api_error(operation="POST /processes", error=e)


@app.delete("/processes/{process_id:path}")
def remove_process_by_path(process_id: str, runner: RunnerDep):
    """指定 process_id の ProcessInfo を削除する（推奨エンドポイント）。"""
    try:
        deleted = runner.submit("write", lambda: delete_process(process_id))
        return {"ok": True, "deleted": deleted}
    except Exception as e:
        _raise_api_error(operation="DELETE /processes/{process_id}", error=e)


@app.delete("/processes")
def remove_process_legacy(
    request: Request,
    req: ProcessDeleteIn,
    runner: RunnerDep,
):
    """互換用の旧削除 API。廃止予定日まで `/processes/{process_id}` と併存する。"""
    request.state.legacy_delete_process_id = req.process_id
    headers = _legacy_delete_headers(req.process_id)
    try:
        deleted = runner.submit("write", lambda: delete_process(req.process_id))
        return {"ok": True, "deleted": deleted}
    except Exception as e:
        _raise_api_error(operation="DELETE /processes", error=e, headers=headers)


@app.post("/step_windows/bulk")
def create_step_windows_bulk(items: list[StepWindowIn], runner: RunnerDep):
    """StepWindow レコードをまとめて保存する。"""
    try:
        inserted = runner.submit("write", lambda: write_step_windows_bulk(items))
        return {"ok": True, "inserted": inserted}
    except Exception as e:
        _raise_api_error(operation="POST /step_windows/bulk", error=e)


@app.post("/parameters/bulk")
def create_parameters_bulk(params: list[ParameterIn], runner: RunnerDep):
    """Parameter レコードをまとめて保存する。"""
    try:
        n = runner.submit("write", lambda: write_parameters_bulk(params))
        return {"ok": True, "inserted": n}
    except Exception as e:
        _raise_api_error(operation="POST /parameters/bulk", error=e)


@app.post("/aggregate/write")
def create_aggregate_write(payload: AggregateWriteIn, runner: RunnerDep):
    """Process/StepWindow/Parameter を 1 API・1 トランザクションで保存する。"""
    try:
        result = runner.submit("write", lambda: write_aggregate_atomic(payload))
        return result
    except Exception as e:
        _raise_api_error(operation="POST /aggregate/write", error=e)
