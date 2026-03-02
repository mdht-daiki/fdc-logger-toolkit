from __future__ import annotations

import shutil
import threading
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from queue import Empty, Queue
from typing import Any

from .db import _init_schema


@dataclass
class Task:
    kind: str
    fn: Callable[[], Any]
    done: threading.Event
    result: Any = None
    error: Exception | None = None


class DBTaskRunner:
    def __init__(self, main_db: Path, temp_db: Path):
        self.main_db = main_db
        self.temp_db = temp_db
        self.q: Queue[Task] = Queue()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._stop = threading.Event()

        self._write_in_progress = False
        self._temp_exists = False

        _init_schema(self.main_db)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        self._thread.join(timeout=2)

    def submit(self, kind: str, fn: Callable[[], Any], timeout: float | None = None) -> Any:
        if self._stop.is_set() or not self._thread.is_alive():
            raise RuntimeError("DBTaskRunner is stopped")
        task = Task(kind=kind, fn=fn, done=threading.Event())
        self.q.put(task)
        if not task.done.wait(timeout=timeout):
            raise TimeoutError(f"task timed out: {kind}")
        if task.error is not None:
            raise task.error
        return task.result

    def _ensure_temp_snapshot(self) -> None:
        if self._temp_exists:
            return
        if not self.main_db.exists():
            _init_schema(self.main_db)
        shutil.copy2(self.main_db.as_posix(), self.temp_db.as_posix())
        self._temp_exists = True

    def _delete_temp(self) -> None:
        if self.temp_db.exists():
            try:
                self.temp_db.unlink()
            except OSError:
                pass
        self._temp_exists = False

    def _loop(self) -> None:
        while not self._stop.is_set():
            try:
                task = self.q.get(timeout=0.2)
            except Empty:
                continue

            try:
                if task.kind == "write":
                    self._write_in_progress = True
                    self._ensure_temp_snapshot()
                    task.result = task.fn()
                    self._write_in_progress = False
                    if self.q.empty():
                        self.q.put(
                            Task(
                                kind="delete_temp",
                                fn=lambda: self._delete_temp(),
                                done=threading.Event(),
                            )
                        )
                elif task.kind == "delete_temp":
                    if not self._write_in_progress:
                        task.result = task.fn()
                else:
                    raise RuntimeError(f"unknown task kind: {task.kind}")
            except Exception as exc:
                task.error = exc
            finally:
                task.done.set()
                self.q.task_done()
