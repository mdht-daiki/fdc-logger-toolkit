from __future__ import annotations

import threading
import time
from pathlib import Path
from queue import Queue

import pytest

from portfolio_fdc.db_api.task_runner import DBTaskRunner, Task


class _AliveFlipThread:
    def __init__(self) -> None:
        # is_alive() の戻り値を1回目だけ True にするためのカウンタ。
        self._calls = 0

    def is_alive(self) -> bool:
        # submit() の事前チェックは通しつつ、待機ループでは停止扱いにする。
        self._calls += 1
        return self._calls == 1


def test_submit_raises_if_worker_stops_after_enqueue(tmp_path: Path) -> None:
    # ワーカー停止後に submit() された場合の防御挙動を確認する。
    runner = DBTaskRunner(main_db=tmp_path / "main.db", temp_db=tmp_path / "temp.db")
    runner.stop()

    # submit() の初回 is_alive 判定だけ通る疑似スレッドを差し込み、
    # done 未完了時に RuntimeError へ遷移することを検証する。
    runner._stop = threading.Event()
    runner._thread = _AliveFlipThread()  # type: ignore[assignment]

    with pytest.raises(RuntimeError, match="stopped before task completion"):
        runner.submit("write", lambda: "never", timeout=None)


class _StopOnPutQueue(Queue[Task]):
    def __init__(self, stop_callback) -> None:
        # 最初の put 時に stop() を挟み込むためのフックを保持する。
        super().__init__()
        self._stop_callback = stop_callback
        self._called = False

    def put(self, item: Task, block: bool = True, timeout: float | None = None) -> None:
        # 停止チェック後〜キュー投入直後のレース窓を再現する。
        if not self._called:
            self._called = True
            self._stop_callback()
        super().put(item, block=block, timeout=0 if timeout is None else timeout)


def test_submit_does_not_block_when_stopped_between_check_and_enqueue(tmp_path: Path) -> None:
    # submit() の停止チェック直後に stop() が発生しても無限待機しないことを確認する。
    runner = DBTaskRunner(main_db=tmp_path / "main.db", temp_db=tmp_path / "temp.db")
    runner.q = _StopOnPutQueue(runner.stop)

    start = time.monotonic()
    with pytest.raises(RuntimeError, match="stopped before task completion"):
        runner.submit("write", lambda: "never", timeout=None)

    assert time.monotonic() - start < 1.0


class _NeverStopThread:
    def join(self, timeout: float | None = None) -> None:
        # join() 呼び出しだけ受け取り、停止しないスレッドを模擬する。
        _ = timeout

    def is_alive(self) -> bool:
        # stop() 後も生存し続ける状態を返す。
        return True


def test_stop_raises_when_worker_does_not_terminate(tmp_path: Path) -> None:
    # join timeout 後もワーカー生存時に RuntimeError となることを確認する。
    runner = DBTaskRunner(main_db=tmp_path / "main.db", temp_db=tmp_path / "temp.db")
    runner.stop()

    runner._stop = threading.Event()
    runner._thread = _NeverStopThread()  # type: ignore[assignment]

    with pytest.raises(RuntimeError, match="failed to stop"):
        runner.stop()
