from __future__ import annotations

import threading
import time
from concurrent.futures import ThreadPoolExecutor
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

    with pytest.raises(RuntimeError, match="DBTaskRunner failed to stop within timeout"):
        runner.stop()


def test_submit_returns_before_timeout(tmp_path: Path) -> None:
    # timeout 指定ありでも、処理が期限内なら正常に結果を返すことを確認する。
    runner = DBTaskRunner(main_db=tmp_path / "main.db", temp_db=tmp_path / "temp.db")
    try:
        result = runner.submit("write", lambda: "ok", timeout=1.0)
        assert result == "ok"
    finally:
        runner.stop()


def test_submit_times_out_with_zero_timeout(tmp_path: Path) -> None:
    # timeout=0 では未完了タスクが即時タイムアウトすることを確認する。
    runner = DBTaskRunner(main_db=tmp_path / "main.db", temp_db=tmp_path / "temp.db")
    try:
        with pytest.raises(TimeoutError, match="task timed out"):
            runner.submit("write", lambda: time.sleep(0.2), timeout=0)
    finally:
        runner.stop()


def test_submit_times_out_with_very_short_timeout(tmp_path: Path) -> None:
    # 極短 timeout で長いタスクを投げると TimeoutError になることを確認する。
    runner = DBTaskRunner(main_db=tmp_path / "main.db", temp_db=tmp_path / "temp.db")
    start = time.monotonic()
    try:
        with pytest.raises(TimeoutError, match="task timed out"):
            runner.submit("write", lambda: time.sleep(0.2), timeout=0.01)
        assert time.monotonic() - start < 0.5
    finally:
        runner.stop()


def test_submit_succeeds_with_short_timeout_when_task_finishes(tmp_path: Path) -> None:
    # 短い timeout でもタスクが期限内完了なら正常に返ることを確認する。
    runner = DBTaskRunner(main_db=tmp_path / "main.db", temp_db=tmp_path / "temp.db")

    def _quick_task() -> str:
        time.sleep(0.01)
        return "ok"

    try:
        result = runner.submit("write", _quick_task, timeout=0.2)
        assert result == "ok"
    finally:
        runner.stop()


def test_submit_supports_concurrent_callers(tmp_path: Path) -> None:
    # 複数スレッドから同時に submit() しても全タスクが処理されることを確認する。
    runner = DBTaskRunner(main_db=tmp_path / "main.db", temp_db=tmp_path / "temp.db")
    counter = 0
    lock = threading.Lock()

    def _submit(i: int) -> int:
        def _task() -> int:
            nonlocal counter
            with lock:
                counter += 1
            return i * 2

        return runner.submit("write", _task, timeout=2.0)

    try:
        with ThreadPoolExecutor(max_workers=5) as executor:
            results = list(executor.map(_submit, range(20)))
        assert sorted(results) == [i * 2 for i in range(20)]
        assert counter == 20
    finally:
        runner.stop()


def test_write_task_eventually_deletes_temp_db(tmp_path: Path) -> None:
    # write 完了後に自動投入される delete_temp で temp DB が最終的に消えることを確認する。
    runner = DBTaskRunner(main_db=tmp_path / "main.db", temp_db=tmp_path / "temp.db")
    try:
        runner.submit("write", lambda: "ok", timeout=2.0)

        deadline = time.monotonic() + 1.0
        while runner.temp_db.exists() and time.monotonic() < deadline:
            time.sleep(0.02)

        assert not runner.temp_db.exists()
    finally:
        runner.stop()


def test_stop_can_be_called_multiple_times(tmp_path: Path) -> None:
    # stop() を複数回呼んでも安全に終了できることを確認する。
    runner = DBTaskRunner(main_db=tmp_path / "main.db", temp_db=tmp_path / "temp.db")
    runner.stop()
    runner.stop()


def test_stop_terminates_worker_thread(tmp_path: Path) -> None:
    # stop() 実行後にワーカースレッドが停止していることを確認する。
    runner = DBTaskRunner(main_db=tmp_path / "main.db", temp_db=tmp_path / "temp.db")
    runner.stop()
    assert not runner._thread.is_alive()
