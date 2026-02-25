from __future__ import annotations

from collections import deque

from .models import StepBundle


class StepPeakQueue:
    def __init__(self, maxlen: int = 4):
        self._q: deque[StepBundle] = deque(maxlen=maxlen)

    def push(self, bundle: StepBundle) -> None:
        self._q.append(bundle)

    def is_full(self) -> bool:
        return len(self._q) == self._q.maxlen

    def to_list(self) -> list[StepBundle]:
        return list(self._q)

    def clear(self) -> None:
        self._q.clear()
