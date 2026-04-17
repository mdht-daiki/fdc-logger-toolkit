"""FastAPI ベースの DB API エントリポイント。

集約結果の書き込み系エンドポイントを提供し、アプリ単位で
`DBTaskRunner` を初期化・再利用・停止する。
"""

from __future__ import annotations

import logging
import sqlite3
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from dataclasses import asdict
from datetime import UTC, datetime
from email.utils import format_datetime
from threading import Lock
from typing import Annotated, NoReturn, cast
from urllib.parse import quote

from fastapi import Depends, FastAPI, HTTPException, Query, Request
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
from .chart_repository import (
    ActiveChartsQueryCriteria,
    ChartRepository,
    ChartsHistoryQueryCriteria,
    ChartsQueryCriteria,
)
from .db import MAIN_DB, TEMP_DB, _init_schema
from .judge_repository import JudgeRepository, JudgeResultsQueryCriteria
from .schemas import (
    AggregateWriteIn,
    ParameterIn,
    ProcessDeleteIn,
    ProcessInfoIn,
    StepWindowIn,
    validate_timestamp_range,
)
from .task_runner import DBTaskRunner

logger = logging.getLogger(__name__)
_runner_lock = Lock()
LEGACY_DELETE_PROCESSES_SUNSET_AT = datetime(2026, 6, 30, 23, 59, 59, tzinfo=UTC)
LEGACY_DELETE_PROCESSES_SUNSET = format_datetime(LEGACY_DELETE_PROCESSES_SUNSET_AT, usegmt=True)
CHARTS_FILTER_PATTERN = r"^[A-Za-z0-9_./:-]+$"
CHARTS_FILTER_MAX_LENGTH = 128
CHART_ID_PATTERN = r"^CHART_[0-9]+$"
JUDGE_LEVEL_PATTERN = r"^(OK|WARN|NG)$"


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
    return raw.astimezone(UTC).isoformat()


def _validate_query_datetime_range(from_ts: datetime | None, to_ts: datetime | None) -> None:
    """from_ts/to_ts の指定整合と範囲整合を検証する。"""
    if (from_ts is None) != (to_ts is None):
        raise HTTPException(
            status_code=400,
            detail="from_ts and to_ts must be specified together",
        )
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
    _validate_query_datetime_range(from_ts, to_ts)

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
    _validate_query_datetime_range(from_ts, to_ts)

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
