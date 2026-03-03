from __future__ import annotations

from fastapi import FastAPI, HTTPException

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

runner = DBTaskRunner(main_db=MAIN_DB, temp_db=TEMP_DB)
app = FastAPI(title="db_api", version="0.1.0")


@app.post("/processes")
def create_process(p: ProcessInfoIn):
    try:
        runner.submit("write", lambda: write_process(p))
        return {"ok": True}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e


@app.delete("/processes")
def remove_process(req: ProcessDeleteIn):
    try:
        deleted = runner.submit("write", lambda: delete_process(req.process_id))
        return {"ok": True, "deleted": deleted}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e


@app.post("/step_windows/bulk")
def create_step_windows_bulk(items: list[StepWindowIn]):
    try:
        inserted = runner.submit("write", lambda: write_step_windows_bulk(items))
        return {"ok": True, "inserted": inserted}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e


@app.post("/parameters/bulk")
def create_parameters_bulk(params: list[ParameterIn]):
    try:
        n = runner.submit("write", lambda: write_parameters_bulk(params))
        return {"ok": True, "inserted": n}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e
