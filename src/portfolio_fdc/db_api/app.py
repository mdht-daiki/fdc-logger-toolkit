"""FastAPI ベースの DB API エントリポイント。

集約結果の書き込み系エンドポイントを提供し、アプリ単位で
`DBTaskRunner` を初期化・再利用・停止する。
"""

from __future__ import annotations

import logging
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from dataclasses import asdict
from datetime import UTC, datetime
from email.utils import format_datetime
from threading import Lock
from typing import Annotated, cast
from urllib.parse import quote

from fastapi import Depends, FastAPI, HTTPException, Query, Request

from .aggregate_repository import (
    delete_process,
    write_aggregate_atomic,
    write_parameters_bulk,
    write_process,
    write_step_windows_bulk,
)
from .chart_repository import ChartRepository, ChartsQueryCriteria
from .db import MAIN_DB, TEMP_DB, _init_schema
from .schemas import (
    AggregateWriteIn,
    ParameterIn,
    ProcessDeleteIn,
    ProcessInfoIn,
    StepWindowIn,
)
from .task_runner import DBTaskRunner

logger = logging.getLogger(__name__)
_runner_lock = Lock()
LEGACY_DELETE_PROCESSES_SUNSET_AT = datetime(2026, 6, 30, 23, 59, 59, tzinfo=UTC)
LEGACY_DELETE_PROCESSES_SUNSET = format_datetime(LEGACY_DELETE_PROCESSES_SUNSET_AT, usegmt=True)


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


RunnerDep = Annotated[DBTaskRunner, Depends(get_runner)]
_chart_repository = ChartRepository()


@app.get("/charts")
def get_charts(
    tool_id: str | None = None,
    chamber_id: str | None = None,
    recipe_id: str | None = None,
    parameter: str | None = None,
    step_no: int | None = Query(default=None, ge=0),
    feature_type: str | None = None,
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
        logger.exception("Failed to fetch charts")
        raise HTTPException(status_code=500, detail="Internal server error") from e


@app.post("/processes")
def create_process(p: ProcessInfoIn, runner: RunnerDep):
    """1 件の ProcessInfo をキュー経由で保存する。"""
    try:
        runner.submit("write", lambda: write_process(p))
        return {"ok": True}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e


@app.delete("/processes/{process_id:path}")
def remove_process_by_path(process_id: str, runner: RunnerDep):
    """指定 process_id の ProcessInfo を削除する（推奨エンドポイント）。"""
    try:
        deleted = runner.submit("write", lambda: delete_process(process_id))
        return {"ok": True, "deleted": deleted}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e


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
        raise HTTPException(status_code=500, detail=str(e), headers=headers) from e


@app.post("/step_windows/bulk")
def create_step_windows_bulk(items: list[StepWindowIn], runner: RunnerDep):
    """StepWindow レコードをまとめて保存する。"""
    try:
        inserted = runner.submit("write", lambda: write_step_windows_bulk(items))
        return {"ok": True, "inserted": inserted}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e


@app.post("/parameters/bulk")
def create_parameters_bulk(params: list[ParameterIn], runner: RunnerDep):
    """Parameter レコードをまとめて保存する。"""
    try:
        n = runner.submit("write", lambda: write_parameters_bulk(params))
        return {"ok": True, "inserted": n}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e


@app.post("/aggregate/write")
def create_aggregate_write(payload: AggregateWriteIn, runner: RunnerDep):
    """Process/StepWindow/Parameter を 1 API・1 トランザクションで保存する。"""
    try:
        result = runner.submit("write", lambda: write_aggregate_atomic(payload))
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e
