from __future__ import annotations

import json
import logging
import math
import sqlite3
from datetime import datetime, timedelta
from typing import NoReturn

from fastapi import HTTPException
from fastapi.responses import JSONResponse

from .datetime_util import to_utc_millis
from .schemas import validate_timestamp_range

logger = logging.getLogger("portfolio_fdc.db_api.app")

CHARTS_FILTER_PATTERN = r"^[A-Za-z0-9_./:-]+$"
CHARTS_FILTER_MAX_LENGTH = 128
CHART_ID_PATTERN = r"^CHART_[0-9]+$"
JUDGE_LEVEL_PATTERN = r"^(OK|WARN|NG)$"
RESULT_ID_PATTERN = r"^JR_[0-9]+$"
NOTIFICATION_RETRY_BACKOFF_MINUTES = {1: 1, 2: 5, 3: 30}


class GovernanceApplyValidationError(Exception):
    """apply payload がしきい値整合性を満たさない場合に送出する内部例外。"""

    def __init__(self, *, message: str) -> None:
        self.message = message


def is_runner_unavailable_error(error: Exception) -> bool:
    """DBTaskRunner 停止/タイムアウト起因の一時的障害かを判定する。"""
    if isinstance(error, TimeoutError):
        return True
    if not isinstance(error, RuntimeError):
        return False
    return str(error).startswith("DBTaskRunner")


def is_transient_operational_error(error: sqlite3.OperationalError) -> bool:
    """OperationalError が一時的な DB 障害かどうかを判定する。"""
    message = str(error).lower()

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


def raise_api_error(
    *,
    operation: str,
    error: Exception,
    headers: dict[str, str] | None = None,
) -> NoReturn:
    """内部例外をログに残しつつ、クライアント向けには安全なエラーを返す。"""
    logger.exception("%s failed: %s", operation, type(error).__name__)

    if is_runner_unavailable_error(error):
        raise HTTPException(
            status_code=503,
            detail="Service temporarily unavailable",
            headers=headers,
        ) from error

    if isinstance(error, sqlite3.OperationalError):
        if is_transient_operational_error(error):
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


def normalize_query_datetime(raw: datetime | None) -> str | None:
    """履歴検索用の datetime クエリを SQLite 比較用 ISO 文字列へ変換する。"""
    if raw is None:
        return None
    if raw.tzinfo is None:
        raise HTTPException(
            status_code=400,
            detail="from_ts and to_ts must be timezone-aware datetimes",
        )
    return to_utc_millis(raw.isoformat())


def validate_query_datetime_range(
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


def parse_chart_pk(chart_id: str | None) -> int | None:
    """CHART_<id> 形式の chart_id を int PK へ変換する。"""
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


def parse_result_pk(result_id: str) -> int:
    """JR_<id> 形式の result_id を int PK へ変換する。"""
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


def not_found_error_response(
    *, code: str = "NOT_FOUND", message: str, details: dict[str, str]
) -> JSONResponse:
    """404 error envelope を返す。"""
    return JSONResponse(
        status_code=404,
        content={
            "ok": False,
            "error": {
                "code": code,
                "message": message,
                "details": details,
            },
        },
    )


def duplicate_idempotency_error_response(*, idempotency_key: str) -> JSONResponse:
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


def conflict_error_response(*, code: str, message: str, details: dict[str, str]) -> JSONResponse:
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


def bad_request_error_response(*, code: str, message: str, details: dict[str, str]) -> JSONResponse:
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


def validation_error_response(*, issues: list[dict[str, object]]) -> JSONResponse:
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


def compute_notification_next_retry_at(*, base_time: datetime, retry_count: int) -> str:
    """retry_count に応じた次回試行時刻を UTC ミリ秒 ISO 文字列で返す。"""
    minutes = NOTIFICATION_RETRY_BACKOFF_MINUTES.get(retry_count)
    if minutes is None:
        max_retries = len(NOTIFICATION_RETRY_BACKOFF_MINUTES)
        raise ValueError(f"retry_count must be between 1 and {max_retries}")
    return to_utc_millis((base_time + timedelta(minutes=minutes)).isoformat())


def parse_threshold_patch(change_payload: str) -> dict[str, float | None]:
    """change_payload からしきい値更新パッチを抽出する。"""
    payload = json.loads(change_payload)
    if not isinstance(payload, dict):
        raise GovernanceApplyValidationError(message="change_payload must be an object")

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
            raise GovernanceApplyValidationError(message=f"{key} must be a number or null")
        numeric = float(value)
        if numeric != numeric or numeric in (float("inf"), float("-inf")):
            raise GovernanceApplyValidationError(message=f"{key} must be finite")
        patch[mapped] = numeric
    return patch


def validate_threshold_consistency(
    *,
    warn_low: float | None,
    warn_high: float | None,
    crit_low: float | None,
    crit_high: float | None,
) -> None:
    """しきい値の大小関係を検証する。"""
    if warn_low is not None and warn_high is not None and warn_low > warn_high:
        raise GovernanceApplyValidationError(
            message="warn_low must be less than or equal to warn_high"
        )
    if crit_low is not None and crit_high is not None and crit_low > crit_high:
        raise GovernanceApplyValidationError(
            message="crit_low must be less than or equal to crit_high"
        )
    if crit_low is not None and warn_low is not None and crit_low > warn_low:
        raise GovernanceApplyValidationError(
            message="crit_low must be less than or equal to warn_low"
        )
    if warn_high is not None and crit_high is not None and warn_high > crit_high:
        raise GovernanceApplyValidationError(
            message="warn_high must be less than or equal to crit_high"
        )
    if crit_low is not None and warn_high is not None and crit_low > warn_high:
        raise GovernanceApplyValidationError(
            message="crit_low must be less than or equal to warn_high"
        )


def thresholds_equal(a: float | None, b: float | None) -> bool:
    """閾値の同値判定。None 同士は等値、数値は許容誤差で比較する。"""
    if a is None and b is None:
        return True
    if a is None or b is None:
        return False
    return math.isclose(a, b, rel_tol=1e-9, abs_tol=1e-12)
