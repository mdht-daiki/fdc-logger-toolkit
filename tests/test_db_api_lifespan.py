from __future__ import annotations

import asyncio
import logging
from pathlib import Path

from portfolio_fdc.db_api import app as db_app


class _RecordingRunner:
    instances: list[_RecordingRunner] = []

    def __init__(self, main_db: Path, temp_db: Path) -> None:
        self.main_db = main_db
        self.temp_db = temp_db
        self.stopped = False
        self.__class__.instances.append(self)

    def stop(self) -> None:
        self.stopped = True


class _ErrorRunner:
    def __init__(self, main_db: Path, temp_db: Path) -> None:
        self.main_db = main_db
        self.temp_db = temp_db

    def stop(self) -> None:
        raise RuntimeError("stop failed")


class _SwapOnStopRunner:
    def __init__(self, main_db: Path, temp_db: Path) -> None:
        self.main_db = main_db
        self.temp_db = temp_db

    def stop(self) -> None:
        db_app.app.state.runner = object()


async def _run_lifespan_once() -> int:
    async with db_app.lifespan(db_app.app):
        return id(db_app.app.state.runner)


def test_lifespan_creates_fresh_runner_per_entry(monkeypatch) -> None:
    _RecordingRunner.instances.clear()
    monkeypatch.setattr(db_app, "DBTaskRunner", _RecordingRunner)

    first_runner_id = asyncio.run(_run_lifespan_once())
    assert not hasattr(db_app.app.state, "runner")

    second_runner_id = asyncio.run(_run_lifespan_once())
    assert not hasattr(db_app.app.state, "runner")

    assert len(_RecordingRunner.instances) == 2
    assert first_runner_id != second_runner_id
    assert all(instance.stopped for instance in _RecordingRunner.instances)


def test_lifespan_handles_stop_runtime_error(monkeypatch, caplog) -> None:
    monkeypatch.setattr(db_app, "DBTaskRunner", _ErrorRunner)

    with caplog.at_level(logging.ERROR):
        asyncio.run(_run_lifespan_once())

    assert "Failed to stop DBTaskRunner during shutdown" in caplog.text
    assert hasattr(db_app.app.state, "runner")
    del db_app.app.state.runner


def test_lifespan_reuses_existing_runner(monkeypatch) -> None:
    _RecordingRunner.instances.clear()
    monkeypatch.setattr(db_app, "DBTaskRunner", _RecordingRunner)

    existing_runner = _RecordingRunner(
        main_db=Path("existing-main.db"), temp_db=Path("existing-temp.db")
    )
    db_app.app.state.runner = existing_runner

    runner_id = asyncio.run(_run_lifespan_once())

    assert runner_id == id(existing_runner)
    assert len(_RecordingRunner.instances) == 1
    assert existing_runner.stopped is True
    assert not hasattr(db_app.app.state, "runner")


def test_lifespan_does_not_delete_replaced_runner_after_stop(monkeypatch) -> None:
    monkeypatch.setattr(db_app, "DBTaskRunner", _SwapOnStopRunner)

    asyncio.run(_run_lifespan_once())

    assert hasattr(db_app.app.state, "runner")
    del db_app.app.state.runner
