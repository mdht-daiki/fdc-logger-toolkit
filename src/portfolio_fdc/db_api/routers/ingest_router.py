from __future__ import annotations

from collections.abc import Callable

from fastapi import APIRouter, Request

from ..aggregate_repository import (
    delete_process as _delete_process,
)
from ..aggregate_repository import (
    write_aggregate_atomic as _write_aggregate_atomic,
)
from ..aggregate_repository import (
    write_parameters_bulk,
    write_process,
    write_step_windows_bulk,
)
from ..api_common import raise_api_error
from ..schemas import AggregateWriteIn, ParameterIn, ProcessDeleteIn, ProcessInfoIn, StepWindowIn
from ..task_runner import DBTaskRunner


class IngestRouter:
    """ingest 書き込み系エンドポイントをまとめるルータークラス。"""

    def __init__(
        self,
        *,
        get_runner: Callable[[Request], DBTaskRunner],
        legacy_headers: Callable[[str | None], dict[str, str]],
        delete_process: Callable[..., int] = _delete_process,
        write_aggregate_atomic: Callable[..., dict[str, int | bool]] = _write_aggregate_atomic,
    ) -> None:
        self._get_runner = get_runner
        self._legacy_headers = legacy_headers
        self._delete_process = delete_process
        self._write_aggregate_atomic = write_aggregate_atomic
        self.router = APIRouter()
        self._register_routes()

    def _register_routes(self) -> None:
        @self.router.post("/processes")
        def create_process(p: ProcessInfoIn, request: Request):
            runner = self._get_runner(request)
            try:
                runner.submit("write", lambda: write_process(p))
                return {"ok": True}
            except Exception as e:
                raise_api_error(operation="POST /processes", error=e)

        @self.router.delete("/processes/{process_id:path}")
        def remove_process_by_path(process_id: str, request: Request):
            runner = self._get_runner(request)
            try:
                deleted = runner.submit("write", lambda: self._delete_process(process_id))
                return {"ok": True, "deleted": deleted}
            except Exception as e:
                raise_api_error(operation="DELETE /processes/{process_id}", error=e)

        @self.router.delete("/processes")
        def remove_process_legacy(request: Request, req: ProcessDeleteIn):
            runner = self._get_runner(request)
            request.state.legacy_delete_process_id = req.process_id
            headers = self._legacy_headers(req.process_id)
            try:
                deleted = runner.submit("write", lambda: self._delete_process(req.process_id))
                return {"ok": True, "deleted": deleted}
            except Exception as e:
                raise_api_error(operation="DELETE /processes", error=e, headers=headers)

        @self.router.post("/step_windows/bulk")
        def create_step_windows_bulk(items: list[StepWindowIn], request: Request):
            runner = self._get_runner(request)
            try:
                inserted = runner.submit("write", lambda: write_step_windows_bulk(items))
                return {"ok": True, "inserted": inserted}
            except Exception as e:
                raise_api_error(operation="POST /step_windows/bulk", error=e)

        @self.router.post("/parameters/bulk")
        def create_parameters_bulk(params: list[ParameterIn], request: Request):
            runner = self._get_runner(request)
            try:
                inserted = runner.submit("write", lambda: write_parameters_bulk(params))
                return {"ok": True, "inserted": inserted}
            except Exception as e:
                raise_api_error(operation="POST /parameters/bulk", error=e)

        @self.router.post("/aggregate/write")
        def create_aggregate_write(payload: AggregateWriteIn, request: Request):
            runner = self._get_runner(request)
            try:
                return runner.submit("write", lambda: self._write_aggregate_atomic(payload))
            except Exception as e:
                raise_api_error(operation="POST /aggregate/write", error=e)
