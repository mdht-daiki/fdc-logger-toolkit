from __future__ import annotations

import asyncio
import logging

from portfolio_fdc.db_api import app as db_app


class _SyncStopRunner:
    def __init__(self) -> None:
        self.stopped = False

    def stop(self) -> None:
        self.stopped = True


class _AsyncStopRunner:
    def __init__(self) -> None:
        self.stopped = False

    async def stop(self) -> None:
        self.stopped = True


class _ErrorStopRunner:
    def stop(self) -> None:
        raise RuntimeError("stop failed")


async def _run_lifespan_once() -> None:
    async with db_app.lifespan(db_app.app):
        return


def test_lifespan_calls_sync_runner_stop(monkeypatch) -> None:
    fake_runner = _SyncStopRunner()
    monkeypatch.setattr(db_app, "runner", fake_runner)

    asyncio.run(_run_lifespan_once())

    assert fake_runner.stopped is True


def test_lifespan_awaits_async_runner_stop(monkeypatch) -> None:
    fake_runner = _AsyncStopRunner()
    monkeypatch.setattr(db_app, "runner", fake_runner)

    asyncio.run(_run_lifespan_once())

    assert fake_runner.stopped is True


def test_lifespan_handles_stop_runtime_error(monkeypatch, caplog) -> None:
    fake_runner = _ErrorStopRunner()
    monkeypatch.setattr(db_app, "runner", fake_runner)

    with caplog.at_level(logging.ERROR):
        asyncio.run(_run_lifespan_once())

    assert "Failed to stop DBTaskRunner during shutdown" in caplog.text
