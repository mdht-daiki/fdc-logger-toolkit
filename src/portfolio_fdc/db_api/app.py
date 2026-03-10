"""FastAPI ベースの DB API エントリポイント。

集約結果の書き込み系エンドポイントを提供し、アプリ単位で
`DBTaskRunner` を初期化・再利用・停止する。
"""

from __future__ import annotations

import logging
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from threading import Lock
from typing import cast

from fastapi import FastAPI, HTTPException, Request, Response

from .aggregate_repository import (
    delete_process,
    write_aggregate_atomic,
    write_parameters_bulk,
    write_process,
    write_step_windows_bulk,
)
from .db import MAIN_DB, TEMP_DB
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
LEGACY_DELETE_PROCESSES_SUNSET = "2026-06-30"


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
    _get_or_create_runner(app)
    try:
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


def _runner_from_request(request: Request) -> DBTaskRunner:
    """リクエストコンテキストから DBTaskRunner を取得する。"""
    return _get_or_create_runner(request.app)


@app.post("/processes")
def create_process(request: Request, p: ProcessInfoIn):
    """1 件の ProcessInfo をキュー経由で保存する。"""
    try:
        _runner_from_request(request).submit("write", lambda: write_process(p))
        return {"ok": True}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e


@app.delete("/processes/{process_id}")
def remove_process_by_path(request: Request, process_id: str):
    """指定 process_id の ProcessInfo を削除する（推奨エンドポイント）。"""
    try:
        deleted = _runner_from_request(request).submit("write", lambda: delete_process(process_id))
        return {"ok": True, "deleted": deleted}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e


@app.delete("/processes")
def remove_process_legacy(request: Request, req: ProcessDeleteIn, response: Response):
    """互換用の旧削除 API。廃止予定日まで `/processes/{process_id}` と併存する。"""
    response.headers["Deprecation"] = "true"
    response.headers["Sunset"] = LEGACY_DELETE_PROCESSES_SUNSET
    response.headers["Link"] = '</processes/{process_id}>; rel="successor-version"'
    try:
        deleted = _runner_from_request(request).submit(
            "write", lambda: delete_process(req.process_id)
        )
        return {"ok": True, "deleted": deleted}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e


@app.post("/step_windows/bulk")
def create_step_windows_bulk(request: Request, items: list[StepWindowIn]):
    """StepWindow レコードをまとめて保存する。"""
    try:
        inserted = _runner_from_request(request).submit(
            "write", lambda: write_step_windows_bulk(items)
        )
        return {"ok": True, "inserted": inserted}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e


@app.post("/parameters/bulk")
def create_parameters_bulk(request: Request, params: list[ParameterIn]):
    """Parameter レコードをまとめて保存する。"""
    try:
        n = _runner_from_request(request).submit("write", lambda: write_parameters_bulk(params))
        return {"ok": True, "inserted": n}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e


@app.post("/aggregate/write")
def create_aggregate_write(request: Request, payload: AggregateWriteIn):
    """Process/StepWindow/Parameter を 1 API・1 トランザクションで保存する。"""
    try:
        result = _runner_from_request(request).submit(
            "write", lambda: write_aggregate_atomic(payload)
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e
