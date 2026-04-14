from __future__ import annotations

import re
import sqlite3
from collections.abc import Iterator
from dataclasses import dataclass
from datetime import UTC, datetime
from uuid import uuid4

import pytest
from fastapi.testclient import TestClient

from portfolio_fdc.db_api import app as db_app
from portfolio_fdc.db_api.chart_repository import ActiveChartSetView
from portfolio_fdc.db_api.db import MAIN_DB, _init_schema
from tests.test_utils import assert_validation_error_envelope

_INSERT_CHART_SQL = """
    INSERT INTO ChartsV2(
        chart_set_id, tool_id, chamber_id, recipe_id, parameter,
        step_no, feature_type, warn_low, warn_high, crit_low, crit_high,
        updated_at, updated_by, update_reason, update_source
    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
"""


@dataclass(frozen=True)
class SeededActiveChartsContext:
    active_chart_set_id: int
    inactive_chart_set_id: int
    restore_active_set_id: int
    active_tool_id: str
    active_other_tool_id: str
    inactive_tool_id: str


def _create_chart_set_with_charts(
    con: sqlite3.Connection,
    *,
    set_name: str,
    now: str,
    chart_rows: list[tuple[object, ...]],
) -> int:
    con.execute(
        "INSERT INTO ChartSet(name, note, created_at, created_by) VALUES (?, ?, ?, ?)",
        (set_name, "test", now, "test"),
    )
    chart_set_id = int(con.execute("SELECT last_insert_rowid()").fetchone()[0])
    for row in chart_rows:
        con.execute(_INSERT_CHART_SQL, (chart_set_id, *row))
    return chart_set_id


@pytest.fixture
def seeded_active_chart_rows() -> Iterator[SeededActiveChartsContext]:
    _init_schema(MAIN_DB)
    con = sqlite3.connect(MAIN_DB.as_posix())
    context: SeededActiveChartsContext | None = None
    try:
        now = datetime.now(UTC).isoformat()
        suffix = uuid4().hex
        prev_active_row = con.execute(
            "SELECT chart_set_id FROM ActiveChartSet WHERE id = 1"
        ).fetchone()
        if prev_active_row is None:
            raise RuntimeError("ActiveChartSet row with id=1 is required")
        restore_active_set_id = int(prev_active_row[0])

        active_tool_id = f"TOOL_ACTIVE_PRIMARY_{suffix}"
        active_other_tool_id = f"TOOL_ACTIVE_SECONDARY_{suffix}"
        inactive_tool_id = f"TOOL_INACTIVE_{suffix}"

        active_chart_set_id = _create_chart_set_with_charts(
            con,
            set_name=f"active_set_{suffix}",
            now=now,
            chart_rows=[
                (
                    active_tool_id,
                    "CH_ACTIVE",
                    "RECIPE_ACTIVE",
                    "dc_bias",
                    1,
                    "mean",
                    1.4,
                    2.6,
                    1.2,
                    2.8,
                    "2026-04-14T09:00:00+09:00",
                    "tester",
                    "test-seed",
                    "test",
                ),
                (
                    active_other_tool_id,
                    "CH_OTHER",
                    "RECIPE_OTHER",
                    "cl2_flow",
                    2,
                    "max",
                    10.0,
                    20.0,
                    9.0,
                    21.0,
                    "2026-04-14T00:00:00Z",
                    "tester",
                    "test-seed",
                    "test",
                ),
            ],
        )
        inactive_chart_set_id = _create_chart_set_with_charts(
            con,
            set_name=f"inactive_set_{suffix}",
            now=now,
            chart_rows=[
                (
                    inactive_tool_id,
                    "CH_ACTIVE",
                    "RECIPE_ACTIVE",
                    "dc_bias",
                    1,
                    "mean",
                    0.4,
                    1.6,
                    0.2,
                    1.8,
                    "2026-04-14T00:00:00Z",
                    "tester",
                    "test-seed",
                    "test",
                )
            ],
        )

        con.execute(
            "INSERT OR REPLACE INTO ActiveChartSet("
            "id, chart_set_id, updated_at, updated_by"
            ") VALUES (1, ?, ?, ?)",
            (active_chart_set_id, "2026-04-14T09:00:00+09:00", "test"),
        )
        con.commit()

        context = SeededActiveChartsContext(
            active_chart_set_id=active_chart_set_id,
            inactive_chart_set_id=inactive_chart_set_id,
            restore_active_set_id=restore_active_set_id,
            active_tool_id=active_tool_id,
            active_other_tool_id=active_other_tool_id,
            inactive_tool_id=inactive_tool_id,
        )
        yield context
    finally:
        con.close()
        if context is not None:
            cleanup = sqlite3.connect(MAIN_DB.as_posix())
            try:
                cleanup.execute(
                    "INSERT OR REPLACE INTO ActiveChartSet("
                    "id, chart_set_id, updated_at, updated_by"
                    ") VALUES (1, ?, ?, ?)",
                    (
                        context.restore_active_set_id,
                        datetime.now(UTC).isoformat(),
                        "test-cleanup",
                    ),
                )
                for chart_set_id in (context.active_chart_set_id, context.inactive_chart_set_id):
                    cleanup.execute(
                        "DELETE FROM ChartsHistory WHERE chart_set_id = ?",
                        (chart_set_id,),
                    )
                    cleanup.execute("DELETE FROM ChartsV2 WHERE chart_set_id = ?", (chart_set_id,))
                    cleanup.execute(
                        "DELETE FROM ChartSet WHERE chart_set_id = ? AND chart_set_id != ?",
                        (chart_set_id, context.restore_active_set_id),
                    )
                cleanup.commit()
            finally:
                cleanup.close()


def test_get_active_charts_returns_active_chart_set_payload(
    client: TestClient,
    seeded_active_chart_rows: SeededActiveChartsContext,
) -> None:
    seeded = seeded_active_chart_rows

    res = client.get("/charts/active")

    assert res.status_code == 200
    body = res.json()
    assert body["ok"] is True
    data = body["data"]
    assert data["active_chart_set_id"] == seeded.active_chart_set_id
    assert data["activated_at"] == "2026-04-14T00:00:00.000Z"

    charts = data["charts"]
    assert len(charts) == 2
    tool_scoped = [chart for chart in charts if chart["parameter"] in {"dc_bias", "cl2_flow"}]
    assert len(tool_scoped) == 2
    first = next(chart for chart in charts if chart["parameter"] == "dc_bias")
    assert re.fullmatch(r"CHART_\d+", first["chart_id"])
    assert first["step_no"] == 1
    assert first["feature_type"] == "mean"
    assert first["warning_lcl"] == 1.4
    assert first["warning_ucl"] == 2.6
    assert first["critical_lcl"] == 1.2
    assert first["critical_ucl"] == 2.8


def test_get_active_charts_filters_by_tool_chamber_and_recipe(
    client: TestClient,
    seeded_active_chart_rows: SeededActiveChartsContext,
) -> None:
    seeded = seeded_active_chart_rows

    by_tool = client.get("/charts/active", params={"tool_id": seeded.active_tool_id})
    assert by_tool.status_code == 200
    by_tool_charts = by_tool.json()["data"]["charts"]
    assert len(by_tool_charts) == 1
    assert by_tool_charts[0]["parameter"] == "dc_bias"

    by_combo = client.get(
        "/charts/active",
        params={"chamber_id": "CH_ACTIVE", "recipe_id": "RECIPE_ACTIVE"},
    )
    assert by_combo.status_code == 200
    by_combo_charts = by_combo.json()["data"]["charts"]
    assert len(by_combo_charts) == 1
    assert by_combo_charts[0]["parameter"] == "dc_bias"


def test_get_active_charts_excludes_inactive_chart_set_rows(
    client: TestClient,
    seeded_active_chart_rows: SeededActiveChartsContext,
) -> None:
    res = client.get("/charts/active", params={"chamber_id": "CH_ACTIVE"})

    assert res.status_code == 200
    charts = res.json()["data"]["charts"]
    assert len(charts) == 1
    assert charts[0]["critical_lcl"] == 1.2
    assert charts[0]["critical_ucl"] == 2.8


def test_get_active_charts_returns_empty_object_when_active_chart_set_missing(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def return_missing_active_chart_set(*args, **kwargs):
        _ = args, kwargs
        return ActiveChartSetView(
            active_chart_set_id=None,
            activated_at=None,
            charts=[],
        )

    monkeypatch.setattr(
        db_app._chart_repository,
        "find_active_chart_set",
        return_missing_active_chart_set,
    )

    res = client.get("/charts/active")

    assert res.status_code == 200
    assert res.json() == {
        "ok": True,
        "data": {
            "active_chart_set_id": None,
            "activated_at": None,
            "charts": [],
        },
    }


@pytest.mark.parametrize("query_key", ["tool_id", "chamber_id", "recipe_id"])
def test_get_active_charts_rejects_invalid_string_filters(
    client: TestClient,
    query_key: str,
) -> None:
    res = client.get("/charts/active", params={query_key: "INVALID VALUE"})

    assert res.status_code == 422
    assert_validation_error_envelope(res.json(), expected_loc_fragment=query_key)


def test_get_active_charts_returns_503_on_operational_error(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """DB接続系の OperationalError は 503 へ分類されることを確認する。"""

    def fail_active_query(*args, **kwargs):
        _ = args, kwargs
        raise sqlite3.OperationalError("unable to open database file")

    monkeypatch.setattr(db_app._chart_repository, "find_active_chart_set", fail_active_query)

    res = client.get("/charts/active")

    assert res.status_code == 503
    assert res.json()["detail"] == "Database temporarily unavailable"


def test_get_active_charts_returns_500_on_database_error(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """DB処理系の DatabaseError は 500 へ分類されることを確認する。"""

    def fail_active_query(*args, **kwargs):
        _ = args, kwargs
        raise sqlite3.IntegrityError("constraint failed")

    monkeypatch.setattr(db_app._chart_repository, "find_active_chart_set", fail_active_query)

    res = client.get("/charts/active")

    assert res.status_code == 500
    assert res.json()["detail"] == "Database operation failed"
