"""FastAPI ベースの DB API エントリポイント。"""

from __future__ import annotations

import logging
import os as _os
import pathlib
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from email.utils import format_datetime
from threading import Lock
from typing import cast
from urllib.parse import quote

from fastapi import FastAPI, HTTPException, Request
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from .api_common import (
    compute_notification_next_retry_at,
    parse_chart_pk,
    raise_api_error,
)
from .audit_event_writer import AuditEventWriter
from .chart_repository import ChartRepository
from .db import MAIN_DB, TEMP_DB, _connect_readonly, _init_schema
from .governance_repository import (
    GovernanceApprovalsRepository,
    GovernanceChangeRequestRepository,
    GovernanceEmergencyChangesRepository,
    GovernanceRatificationsRepository,
)
from .judge_repository import JudgeRepository
from .routers.governance_router import GovernanceRouter
from .routers.ingest_router import IngestRouter
from .routers.query_router import QueryRouter
from .task_runner import DBTaskRunner

logger = logging.getLogger(__name__)
_runner_lock = Lock()
DATA_ROOT = "DATA_ROOT"

LEGACY_DELETE_PROCESSES_SUNSET_AT = datetime(2026, 6, 30, 23, 59, 59, tzinfo=UTC)
LEGACY_DELETE_PROCESSES_SUNSET = format_datetime(LEGACY_DELETE_PROCESSES_SUNSET_AT, usegmt=True)


# Backward-compatible alias for tests importing from this module.
_compute_notification_next_retry_at = compute_notification_next_retry_at
_parse_chart_pk = parse_chart_pk


def _get_allowed_base_dirs() -> list[pathlib.Path]:
    """許可されたベースディレクトリリストを取得。"""
    data_root = _os.environ.get(DATA_ROOT)
    if data_root:
        return [pathlib.Path(data_root).resolve()]
    return [pathlib.Path.cwd().resolve()]


def _build_waveform_preview(process_id: str, limit: int) -> dict[str, object]:
    """ProcessInfo.raw_csv_path からドリルダウン表示用の波形プレビューを返す。"""
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
        return {"process_id": process_id, "source_path": None, "points": []}

    src = pathlib.Path(str(raw_path))
    if not src.is_absolute():
        src = pathlib.Path.cwd() / src

    src_resolved = src.resolve()
    allowed_base_dirs = _get_allowed_base_dirs()
    is_allowed = any(
        src_resolved == allowed_dir or str(src_resolved).startswith(str(allowed_dir) + _os.sep)
        for allowed_dir in allowed_base_dirs
    )
    if not is_allowed:
        raise HTTPException(
            status_code=403,
            detail=(
                "Access to files outside the allowed base directories is forbidden: "
                f"process_id={process_id}, file={src_resolved.name}"
            ),
        )

    if not src_resolved.exists():
        return {
            "process_id": process_id,
            "source_path": src_resolved.as_posix(),
            "points": [],
        }

    try:
        import pandas as pd

        with src_resolved.open("r", encoding="utf-8", errors="ignore") as f:
            start_row = 0
            for idx, line in enumerate(f):
                if idx >= 200:
                    break
                if line.strip().upper().startswith("DATA"):
                    start_row = idx + 1
                    break
            f.seek(0)
            for _ in range(start_row):
                next(f)
            try:
                frame = pd.read_csv(f)
            except pd.errors.ParserError:
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
            {"x": str(x), "y": None if pd.isna(y) else float(y)}
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


# Repository singletons are shared across routers.
_chart_repository = ChartRepository()
_judge_repository = JudgeRepository()
_governance_change_request_repository = GovernanceChangeRequestRepository()
_governance_approvals_repository = GovernanceApprovalsRepository()
_governance_emergency_changes_repository = GovernanceEmergencyChangesRepository()
_governance_ratifications_repository = GovernanceRatificationsRepository()
_audit_event_writer = AuditEventWriter()

query_router = QueryRouter(
    chart_repository=_chart_repository,
    judge_repository=_judge_repository,
    governance_change_request_repository=_governance_change_request_repository,
    get_runner=_runner_from_request,
    build_waveform_preview=lambda process_id, limit: _build_waveform_preview(process_id, limit),
)

governance_router = GovernanceRouter(
    governance_change_request_repository=_governance_change_request_repository,
    governance_approvals_repository=_governance_approvals_repository,
    governance_emergency_changes_repository=_governance_emergency_changes_repository,
    governance_ratifications_repository=_governance_ratifications_repository,
    audit_event_writer=_audit_event_writer,
    get_runner=_runner_from_request,
)

ingest_router = IngestRouter(
    get_runner=_runner_from_request,
    legacy_headers=_legacy_delete_headers,
)

app.include_router(query_router.router)
app.include_router(governance_router.router)
app.include_router(ingest_router.router)


def _raise_api_error(*, operation: str, error: Exception, headers: dict[str, str] | None = None):
    """後方互換のため app モジュールからも共通エラーハンドラを公開する。"""
    raise_api_error(operation=operation, error=error, headers=headers)
