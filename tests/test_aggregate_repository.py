from __future__ import annotations

from dataclasses import dataclass

import pytest

from portfolio_fdc.db_api import aggregate_repository as repo
from portfolio_fdc.db_api.schemas import AggregateWriteIn, ProcessInfoIn


@dataclass
class _FakeConnection:
    fail_on_execute_at: int | None = None
    execute_call_count: int = 0
    rollback_called: bool = False
    commit_called: bool = False
    close_called: bool = False

    def execute(self, sql: str, params=None):
        _ = sql, params
        self.execute_call_count += 1
        if (
            self.fail_on_execute_at is not None
            and self.execute_call_count == self.fail_on_execute_at
        ):
            raise RuntimeError("forced db failure")
        return self

    def executemany(self, sql: str, seq):
        _ = sql, seq
        return self

    def commit(self) -> None:
        self.commit_called = True

    def rollback(self) -> None:
        self.rollback_called = True

    def close(self) -> None:
        self.close_called = True


def test_write_aggregate_atomic_rolls_back_on_failure(monkeypatch) -> None:
    """write_aggregate_atomic の途中失敗時に rollback が呼ばれることを確認する。"""
    fake = _FakeConnection(fail_on_execute_at=3)
    monkeypatch.setattr(repo, "_connect", lambda _path: fake)

    payload = AggregateWriteIn(
        process=ProcessInfoIn(
            process_id="p1",
            tool_id="TOOL_A",
            chamber_id="CH1",
            recipe_id="UNKNOWN",
            start_ts="2026-03-11T00:00:00",
            end_ts="2026-03-11T00:00:01",
            raw_csv_path="data/detail/p1.csv",
        ),
        step_windows=[],
        parameters=[],
    )

    with pytest.raises(RuntimeError, match="forced db failure"):
        repo.write_aggregate_atomic(payload)

    assert fake.rollback_called is True
    assert fake.commit_called is False
    assert fake.close_called is True
