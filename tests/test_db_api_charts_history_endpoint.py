from __future__ import annotations

import re
import sqlite3
from collections.abc import Iterator
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from uuid import uuid4

import pytest
from fastapi.testclient import TestClient

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
class SeededChartsHistoryContext:
    chart_set_id: int
    chart_id: str


@pytest.fixture
def seeded_charts_history_context() -> Iterator[SeededChartsHistoryContext]:
    _init_schema(MAIN_DB)
    con = sqlite3.connect(MAIN_DB.as_posix())
    context: SeededChartsHistoryContext | None = None
    chart_set_id: int | None = None
    try:
        now = datetime.now(UTC).isoformat()
        suffix = uuid4().hex

        con.execute(
            "INSERT INTO ChartSet(name, note, created_at, created_by) VALUES (?, ?, ?, ?)",
            (f"history_set_{suffix}", "test", now, "test"),
        )
        chart_set_id = int(con.execute("SELECT last_insert_rowid()").fetchone()[0])

        con.execute(
            _INSERT_CHART_SQL,
            (
                chart_set_id,
                f"TOOL_HISTORY_{suffix}",
                "CH_HISTORY",
                "RECIPE_HISTORY",
                "dc_bias",
                1,
                "mean",
                1.4,
                2.6,
                1.2,
                2.8,
                "2026-04-14T00:00:00Z",
                "tester",
                "seed",
                "test",
            ),
        )
        chart_pk = int(con.execute("SELECT last_insert_rowid()").fetchone()[0])

        base = datetime(2026, 4, 14, 0, 0, 0, tzinfo=UTC)
        for index in range(120):
            changed_at = (base + timedelta(seconds=index)).isoformat().replace("+00:00", "Z")
            if index == 0:
                changed_at = "2026-04-14T09:00:00.123999+09:00"
            change_source = "normal_pr" if index % 2 == 0 else "emergency_manual"
            con.execute(
                """
                INSERT INTO ChartsHistory(
                    chart_set_id, tool_id, chamber_id, recipe_id, parameter,
                    step_no, feature_type, old_warn_low, old_warn_high,
                    old_crit_low, old_crit_high, new_warn_low, new_warn_high,
                    new_crit_low, new_crit_high, changed_at, changed_by,
                    change_reason, change_source
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    chart_set_id,
                    f"TOOL_HISTORY_{suffix}",
                    "CH_HISTORY",
                    "RECIPE_HISTORY",
                    "dc_bias",
                    1,
                    "mean",
                    1.0,
                    2.0,
                    0.8,
                    2.2,
                    1.1,
                    2.1,
                    0.9,
                    2.3,
                    changed_at,
                    "tester",
                    f"reason-{index}",
                    change_source,
                ),
            )

        con.commit()
        context = SeededChartsHistoryContext(
            chart_set_id=chart_set_id,
            chart_id=f"CHART_{chart_pk}",
        )
        yield context
    finally:
        con.close()
        if chart_set_id is not None:
            cleanup = sqlite3.connect(MAIN_DB.as_posix())
            try:
                cleanup.execute(
                    "DELETE FROM ChartsHistory WHERE chart_set_id = ?",
                    (chart_set_id,),
                )
                cleanup.execute(
                    "DELETE FROM ChartsV2 WHERE chart_set_id = ?",
                    (chart_set_id,),
                )
                cleanup.execute(
                    "DELETE FROM ChartSet WHERE chart_set_id = ?",
                    (chart_set_id,),
                )
                cleanup.commit()
            finally:
                cleanup.close()


def test_get_charts_history_returns_contract_fields(
    client: TestClient,
    seeded_charts_history_context: SeededChartsHistoryContext,
) -> None:
    seeded = seeded_charts_history_context

    res = client.get("/charts/history", params={"chart_set_id": seeded.chart_set_id, "limit": 500})

    assert res.status_code == 200
    body = res.json()
    assert body["ok"] is True
    data = body["data"]
    assert len(data) == 120

    first = data[0]
    expected_keys = {
        "history_id",
        "chart_id",
        "chart_set_id",
        "change_source",
        "change_reason",
        "before",
        "after",
        "changed_by",
        "changed_at",
    }
    assert expected_keys.issubset(first.keys())
    assert re.fullmatch(r"HIS_\d+", first["history_id"])
    assert first["chart_id"] == seeded.chart_id
    assert first["chart_set_id"] == seeded.chart_set_id
    assert re.fullmatch(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d{3}Z", first["changed_at"])

    before = first["before"]
    after = first["after"]
    assert before == {
        "warning_lcl": 1.0,
        "warning_ucl": 2.0,
        "critical_lcl": 0.8,
        "critical_ucl": 2.2,
    }
    assert after == {
        "warning_lcl": 1.1,
        "warning_ucl": 2.1,
        "critical_lcl": 0.9,
        "critical_ucl": 2.3,
    }

    reason_0_row = next((item for item in data if item["change_reason"] == "reason-0"), None)
    assert reason_0_row is not None, "reason-0 history record not found"
    assert reason_0_row["changed_at"] == "2026-04-14T00:00:00.123Z", (
        f"Timestamp normalization failed: +09:00 with microsecond 123999 "
        f"should convert to UTC 00:00:00.123Z, got {reason_0_row['changed_at']}"
    )


def test_get_charts_history_applies_default_limit_without_limit_param(
    client: TestClient,
    seeded_charts_history_context: SeededChartsHistoryContext,
) -> None:
    seeded = seeded_charts_history_context

    res = client.get("/charts/history", params={"chart_set_id": seeded.chart_set_id})

    assert res.status_code == 200
    body = res.json()
    assert body["ok"] is True
    assert len(body["data"]) == 100


def test_get_charts_history_supports_chart_id_and_change_source_filters(
    client: TestClient,
    seeded_charts_history_context: SeededChartsHistoryContext,
) -> None:
    seeded = seeded_charts_history_context

    res = client.get(
        "/charts/history",
        params={
            "chart_id": seeded.chart_id,
            "change_source": "normal_pr",
            "limit": 500,
        },
    )

    assert res.status_code == 200
    rows = res.json()["data"]
    assert len(rows) == 60
    assert all(item["chart_id"] == seeded.chart_id for item in rows)
    assert all(item["change_source"] == "normal_pr" for item in rows)


def test_get_charts_history_applies_default_limit_with_chart_id_filter(
    client: TestClient,
    seeded_charts_history_context: SeededChartsHistoryContext,
) -> None:
    seeded = seeded_charts_history_context

    res = client.get(
        "/charts/history",
        params={"chart_id": seeded.chart_id},
    )

    assert res.status_code == 200
    body = res.json()
    assert body["ok"] is True
    assert len(body["data"]) == 100


def test_get_charts_history_supports_limit_and_offset(
    client: TestClient,
    seeded_charts_history_context: SeededChartsHistoryContext,
) -> None:
    seeded = seeded_charts_history_context

    first_page = client.get(
        "/charts/history",
        params={"chart_set_id": seeded.chart_set_id, "limit": 1, "offset": 0},
    )
    second_page = client.get(
        "/charts/history",
        params={"chart_set_id": seeded.chart_set_id, "limit": 1, "offset": 1},
    )

    assert first_page.status_code == 200
    assert second_page.status_code == 200
    first_row = first_page.json()["data"][0]
    second_row = second_page.json()["data"][0]
    assert first_row["history_id"] != second_row["history_id"]


def test_get_charts_history_returns_empty_when_offset_exceeds_total(
    client: TestClient,
    seeded_charts_history_context: SeededChartsHistoryContext,
) -> None:
    seeded = seeded_charts_history_context

    res = client.get(
        "/charts/history",
        params={"chart_set_id": seeded.chart_set_id, "offset": 1000},
    )

    assert res.status_code == 200
    body = res.json()
    assert body["ok"] is True
    assert body["data"] == []


def test_get_charts_history_supports_from_to_filter(
    client: TestClient,
    seeded_charts_history_context: SeededChartsHistoryContext,
) -> None:
    seeded = seeded_charts_history_context

    res = client.get(
        "/charts/history",
        params={
            "chart_set_id": seeded.chart_set_id,
            "from_ts": "2026-04-14T00:00:10Z",
            "to_ts": "2026-04-14T00:00:12Z",
            "limit": 500,
        },
    )

    assert res.status_code == 200
    rows = res.json()["data"]
    assert len(rows) == 3
    assert all(
        "2026-04-14T00:00:10.000Z" <= item["changed_at"] <= "2026-04-14T00:00:12.999Z"
        for item in rows
    )


def test_get_charts_history_supports_from_ts_only_filter(
    client: TestClient,
    seeded_charts_history_context: SeededChartsHistoryContext,
) -> None:
    seeded = seeded_charts_history_context

    res = client.get(
        "/charts/history",
        params={
            "chart_set_id": seeded.chart_set_id,
            "from_ts": "2026-04-14T00:01:58Z",
            "limit": 500,
        },
    )

    assert res.status_code == 200
    body = res.json()
    assert body["ok"] is True
    rows = body["data"]
    assert len(rows) == 2
    assert all(item["changed_at"] >= "2026-04-14T00:01:58.000Z" for item in rows)


def test_get_charts_history_supports_to_ts_only_filter(
    client: TestClient,
    seeded_charts_history_context: SeededChartsHistoryContext,
) -> None:
    seeded = seeded_charts_history_context

    res = client.get(
        "/charts/history",
        params={
            "chart_set_id": seeded.chart_set_id,
            "to_ts": "2026-04-14T00:00:01Z",
            "limit": 500,
        },
    )

    assert res.status_code == 200
    body = res.json()
    assert body["ok"] is True
    rows = body["data"]
    assert len(rows) == 2
    assert {item["change_reason"] for item in rows} == {"reason-0", "reason-1"}
    assert all(item["changed_at"] <= "2026-04-14T00:00:01.000Z" for item in rows)


def test_get_charts_history_returns_empty_for_non_matching_filters(
    client: TestClient,
    seeded_charts_history_context: SeededChartsHistoryContext,
) -> None:
    seeded = seeded_charts_history_context

    res = client.get(
        "/charts/history",
        params={
            "chart_set_id": seeded.chart_set_id,
            "change_source": "nonexistent_source",
        },
    )

    assert res.status_code == 200
    body = res.json()
    assert body["ok"] is True
    assert body["data"] == []


def test_get_charts_history_rejects_invalid_chart_id_pattern(client: TestClient) -> None:
    res = client.get("/charts/history", params={"chart_id": "CHART_INVALID"})

    assert res.status_code == 422
    assert_validation_error_envelope(res.json(), expected_loc_fragment="chart_id")


def test_get_charts_history_rejects_limit_exceeds_max(client: TestClient) -> None:
    res = client.get("/charts/history", params={"limit": 501})

    assert res.status_code == 422
    assert_validation_error_envelope(res.json(), expected_loc_fragment="limit")


def test_get_charts_history_rejects_negative_offset(client: TestClient) -> None:
    res = client.get("/charts/history", params={"offset": -1})

    assert res.status_code == 422
    assert_validation_error_envelope(res.json(), expected_loc_fragment="offset")


def test_get_charts_history_rejects_invalid_time_range(client: TestClient) -> None:
    res = client.get(
        "/charts/history",
        params={
            "from_ts": "2026-04-14T00:01:00Z",
            "to_ts": "2026-04-14T00:00:00Z",
        },
    )

    assert res.status_code == 400
    assert "end_ts must be greater than or equal to start_ts" in res.json()["detail"]


def test_get_charts_history_rejects_mixed_naive_and_aware_timestamps(client: TestClient) -> None:
    res = client.get(
        "/charts/history",
        params={
            "from_ts": "2026-04-14T00:00:00",
            "to_ts": "2026-04-14T00:00:00Z",
        },
    )

    assert res.status_code == 400
    assert "same timezone format" in res.json()["detail"]


def test_get_charts_history_rejects_chart_id_out_of_int64_range(client: TestClient) -> None:
    # Try chart_id with numeric part exceeding int64 max (2**63 - 1)
    out_of_range_pk = 2**63
    res = client.get("/charts/history", params={"chart_id": f"CHART_{out_of_range_pk}"})

    assert res.status_code == 400
    assert res.json()["detail"] == "Invalid chart_id"
