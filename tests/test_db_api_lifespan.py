from __future__ import annotations

import asyncio
import logging
from pathlib import Path

from portfolio_fdc.db_api import app as db_app


class _RecordingRunner:
    """停止呼び出しの有無を記録するテスト用ランナー。"""

    instances: list[_RecordingRunner] = []

    def __init__(self, main_db: Path, temp_db: Path) -> None:
        self.main_db = main_db
        self.temp_db = temp_db
        self.stopped = False
        self.__class__.instances.append(self)

    def stop(self) -> None:
        self.stopped = True


class _ErrorRunner:
    """stop() が常に失敗するテスト用ランナー。"""

    def __init__(self, main_db: Path, temp_db: Path) -> None:
        self.main_db = main_db
        self.temp_db = temp_db

    def stop(self) -> None:
        raise RuntimeError("stop failed")


class _SwapOnStopRunner:
    """stop() 中に app.state.runner を差し替えるテスト用ランナー。"""

    def __init__(self, main_db: Path, temp_db: Path) -> None:
        self.main_db = main_db
        self.temp_db = temp_db

    def stop(self) -> None:
        db_app.app.state.runner = object()


async def _run_lifespan_once(*, create_runner: bool = False) -> int | None:
    """lifespan を 1 回実行し、必要なら遅延初期化で runner を生成する。"""
    async with db_app.lifespan(db_app.app):
        if not create_runner:
            return None
        runner = db_app._get_or_create_runner(db_app.app)
        return id(runner)


def test_lifespan_lazy_init_does_not_create_runner_without_request(monkeypatch) -> None:
    """リクエストがなければ lifespan 開始時に runner を作らないことを確認する。"""
    _RecordingRunner.instances.clear()
    monkeypatch.setattr(db_app, "DBTaskRunner", _RecordingRunner)

    first_runner_id = asyncio.run(_run_lifespan_once())
    assert not hasattr(db_app.app.state, "runner")

    second_runner_id = asyncio.run(_run_lifespan_once())
    assert not hasattr(db_app.app.state, "runner")

    assert first_runner_id is None
    assert second_runner_id is None
    assert len(_RecordingRunner.instances) == 0


def test_lifespan_handles_stop_runtime_error(monkeypatch, caplog) -> None:
    """shutdown 時の stop 失敗がログに残り、例外送出しないことを確認する。"""
    monkeypatch.setattr(db_app, "DBTaskRunner", _ErrorRunner)

    with caplog.at_level(logging.ERROR):
        asyncio.run(_run_lifespan_once(create_runner=True))

    assert "Failed to stop DBTaskRunner during shutdown" in caplog.text
    assert hasattr(db_app.app.state, "runner")
    del db_app.app.state.runner


def test_lifespan_reuses_existing_runner(monkeypatch) -> None:
    """既存 runner がある場合は shutdown で同一インスタンスを停止する。"""
    _RecordingRunner.instances.clear()
    monkeypatch.setattr(db_app, "DBTaskRunner", _RecordingRunner)

    existing_runner = _RecordingRunner(
        main_db=Path("existing-main.db"), temp_db=Path("existing-temp.db")
    )
    db_app.app.state.runner = existing_runner

    runner_id = asyncio.run(_run_lifespan_once())

    assert runner_id is None
    assert len(_RecordingRunner.instances) == 1
    assert existing_runner.stopped is True
    assert not hasattr(db_app.app.state, "runner")


def test_lifespan_does_not_delete_replaced_runner_after_stop(monkeypatch) -> None:
    """stop 中に差し替えられた runner を shutdown で誤削除しないことを確認する。"""
    monkeypatch.setattr(db_app, "DBTaskRunner", _SwapOnStopRunner)

    db_app.app.state.runner = _SwapOnStopRunner(Path("main.db"), Path("temp.db"))
    asyncio.run(_run_lifespan_once())

    assert hasattr(db_app.app.state, "runner")
    del db_app.app.state.runner
