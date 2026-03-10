"""StepBundle を固定長で保持するキュー。"""

from __future__ import annotations

from collections import deque

from .models import StepBundle


class StepPeakQueue:
    """最新のステップ束を最大件数まで保持する FIFO キュー。"""

    def __init__(self, maxlen: int = 4):
        """保持上限を指定してキューを初期化する。"""
        self._q: deque[StepBundle] = deque(maxlen=maxlen)

    def push(self, bundle: StepBundle) -> None:
        """新しいステップ束を末尾へ追加する。"""
        self._q.append(bundle)

    def is_full(self) -> bool:
        """キューが保持上限に達しているかを返す。"""
        return len(self._q) == self._q.maxlen

    def to_list(self) -> list[StepBundle]:
        """現在のキュー内容をリストとして返す。"""
        return list(self._q)

    def clear(self) -> None:
        """キュー内容をすべて破棄する。"""
        self._q.clear()
