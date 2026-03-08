from __future__ import annotations

import logging
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from threading import Lock
from typing import cast

from fastapi import FastAPI, HTTPException, Request

from .aggregate_repository import (
    delete_process,
    write_parameters_bulk,
    write_process,
    write_step_windows_bulk,
)
from .db import MAIN_DB, TEMP_DB
from .schemas import (
    ParameterIn,
    ProcessDeleteIn,
    ProcessInfoIn,
    StepWindowIn,
)
from .task_runner import DBTaskRunner

logger = logging.getLogger(__name__)
_runner_lock = Lock()


def _get_or_create_runner(app: FastAPI) -> DBTaskRunner:
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
    _get_or_create_runner(app)
    try:
        yield
    finally:
        try:
            with _runner_lock:
                if hasattr(app.state, "runner"):
                    runner = cast(DBTaskRunner, app.state.runner)
                    runner.stop()
        except RuntimeError:
            logger.exception("Failed to stop DBTaskRunner during shutdown")
        finally:
            with _runner_lock:
                if hasattr(app.state, "runner"):
                    del app.state.runner


app = FastAPI(title="db_api", version="0.1.0", lifespan=lifespan)


def _runner_from_request(request: Request) -> DBTaskRunner:
    return _get_or_create_runner(request.app)


@app.post("/processes")
def create_process(request: Request, p: ProcessInfoIn):
    try:
        _runner_from_request(request).submit("write", lambda: write_process(p))
        return {"ok": True}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e


@app.delete("/processes")
def remove_process(request: Request, req: ProcessDeleteIn):
    try:
        deleted = _runner_from_request(request).submit(
            "write", lambda: delete_process(req.process_id)
        )
        return {"ok": True, "deleted": deleted}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e


@app.post("/step_windows/bulk")
def create_step_windows_bulk(request: Request, items: list[StepWindowIn]):
    try:
        inserted = _runner_from_request(request).submit(
            "write", lambda: write_step_windows_bulk(items)
        )
        return {"ok": True, "inserted": inserted}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e


@app.post("/parameters/bulk")
def create_parameters_bulk(request: Request, params: list[ParameterIn]):
    try:
        n = _runner_from_request(request).submit("write", lambda: write_parameters_bulk(params))
        return {"ok": True, "inserted": n}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e
