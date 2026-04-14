from __future__ import annotations

import re
import sqlite3
from dataclasses import dataclass
from datetime import UTC, datetime
from uuid import uuid4

from fastapi.testclient import TestClient

from portfolio_fdc.db_api.db import MAIN_DB, _init_schema

_INSERT_CHART_SQL = """
    INSERT INTO ChartsV2(
        chart_set_id, tool_id, chamber_id, recipe_id, parameter,
        step_no, feature_type, warn_low, warn_high, crit_low, crit_high,
        updated_at, updated_by, update_reason, update_source
    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
"""


@dataclass(frozen=True)
class SeededChartsContext:
    """seed された chart テストデータの後処理用コンテキスト。"""

    tool_primary: str
    tool_secondary: str
    restore_active_set_id: int
    chart_set_ids: tuple[int, ...]


def _create_chart_set_with_charts(
    con: sqlite3.Connection,
    *,
    set_name: str,
    now: str,
    chart_rows: list[tuple[object, ...]],
) -> int:
    """ChartSet を作成し、紐づく ChartsV2 レコードを投入する。"""
    con.execute(
        "INSERT INTO ChartSet(name, note, created_at, created_by) VALUES (?, ?, ?, ?)",
        (set_name, "test", now, "test"),
    )
    chart_set_id = int(con.execute("SELECT last_insert_rowid()").fetchone()[0])

    for row in chart_rows:
        con.execute(_INSERT_CHART_SQL, (chart_set_id, *row))

    return chart_set_id


def _seed_chart_rows_for_get_charts() -> SeededChartsContext:
    """GET /charts テスト用に active/inactive の 2 chart を投入する。"""
    _init_schema(MAIN_DB)
    con = sqlite3.connect(MAIN_DB.as_posix())
    try:
        now = datetime.now(UTC).isoformat()
        suffix = uuid4().hex
        tool_active = f"TOOL_GET_CHARTS_ACTIVE_{suffix}"
        tool_inactive = f"TOOL_GET_CHARTS_INACTIVE_{suffix}"

        prev_active_row = con.execute(
            "SELECT chart_set_id FROM ActiveChartSet WHERE id = 1"
        ).fetchone()
        if prev_active_row is None:
            raise RuntimeError("ActiveChartSet row with id=1 is required")
        prev_active_set_id = int(prev_active_row[0])

        active_set_id = _create_chart_set_with_charts(
            con,
            set_name=f"active_set_{suffix}",
            now=now,
            chart_rows=[
                (
                    tool_active,
                    "CH1",
                    "RECIPE_A",
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
                )
            ],
        )
        inactive_set_id = _create_chart_set_with_charts(
            con,
            set_name=f"inactive_set_{suffix}",
            now=now,
            chart_rows=[
                (
                    tool_inactive,
                    "CH1",
                    "RECIPE_A",
                    "dc_bias",
                    1,
                    "mean",
                    1.0,
                    2.0,
                    0.8,
                    2.2,
                    "2026-04-14T00:00:00Z",
                    "tester",
                    "test-seed",
                    "test",
                )
            ],
        )

        con.execute(
            "INSERT OR REPLACE INTO ActiveChartSet(id, chart_set_id, updated_at, updated_by)"
            " VALUES (1, ?, ?, ?)",
            (active_set_id, now, "test"),
        )

        con.commit()
        return SeededChartsContext(
            tool_primary=tool_active,
            tool_secondary=tool_inactive,
            restore_active_set_id=prev_active_set_id,
            chart_set_ids=(active_set_id, inactive_set_id),
        )
    finally:
        con.close()


def _cleanup_seeded_chart_rows(context: SeededChartsContext) -> None:
    """テスト投入した chart データを後片付けする。"""
    con = sqlite3.connect(MAIN_DB.as_posix())
    try:
        con.execute(
            "INSERT OR REPLACE INTO ActiveChartSet(id, chart_set_id, updated_at, updated_by)"
            " VALUES (1, ?, ?, ?)",
            (context.restore_active_set_id, datetime.now(UTC).isoformat(), "test-cleanup"),
        )

        for chart_set_id in context.chart_set_ids:
            con.execute(
                "DELETE FROM ChartsV2 WHERE chart_set_id = ?",
                (chart_set_id,),
            )
            con.execute(
                "DELETE FROM ChartSet WHERE chart_set_id = ? AND chart_set_id != ?",
                (chart_set_id, context.restore_active_set_id),
            )

        con.commit()
    finally:
        con.close()


def _seed_chart_rows_for_filter_tests() -> SeededChartsContext:
    """各クエリフィルタ検証用に属性が異なる 2 chart を投入する。"""
    _init_schema(MAIN_DB)
    con = sqlite3.connect(MAIN_DB.as_posix())
    try:
        now = datetime.now(UTC).isoformat()
        suffix = uuid4().hex
        tool_match = f"TOOL_FILTER_MATCH_{suffix}"
        tool_other = f"TOOL_FILTER_OTHER_{suffix}"

        prev_active_row = con.execute(
            "SELECT chart_set_id FROM ActiveChartSet WHERE id = 1"
        ).fetchone()
        if prev_active_row is None:
            raise RuntimeError("ActiveChartSet row with id=1 is required")
        prev_active_set_id = int(prev_active_row[0])

        chart_set_id = _create_chart_set_with_charts(
            con,
            set_name=f"filter_set_{suffix}",
            now=now,
            chart_rows=[
                (
                    tool_match,
                    "CH_FILTER_MATCH",
                    "RECIPE_FILTER_MATCH",
                    "dc_bias",
                    2,
                    "mean",
                    1.4,
                    2.6,
                    1.2,
                    2.8,
                    "2026-04-14T00:00:00Z",
                    "tester",
                    "test-seed",
                    "test",
                ),
                (
                    tool_other,
                    "CH_FILTER_OTHER",
                    "RECIPE_FILTER_OTHER",
                    "cl2_flow",
                    3,
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

        con.execute(
            "INSERT OR REPLACE INTO ActiveChartSet(id, chart_set_id, updated_at, updated_by)"
            " VALUES (1, ?, ?, ?)",
            (chart_set_id, now, "test"),
        )

        con.commit()
        return SeededChartsContext(
            tool_primary=tool_match,
            tool_secondary=tool_other,
            restore_active_set_id=prev_active_set_id,
            chart_set_ids=(chart_set_id,),
        )
    finally:
        con.close()


def _assert_all_rows_match(data: list[dict[str, object]], key: str, expected: object) -> None:
    """返却された全要素が指定キーの期待値を満たすことを確認する。"""
    assert data
    assert all(item.get(key) == expected for item in data)


def test_get_charts_returns_chart_rows_with_contract_fields(client: TestClient) -> None:
    """GET /charts が契約フィールドを返すことを確認する。"""
    seeded = _seed_chart_rows_for_get_charts()
    tool_active = seeded.tool_primary
    tool_inactive = seeded.tool_secondary
    try:
        res = client.get("/charts")

        assert res.status_code == 200
        body = res.json()
        assert body["ok"] is True

        rows = [item for item in body["data"] if item["tool_id"] in {tool_active, tool_inactive}]
        assert len(rows) == 2

        for row in rows:
            assert re.fullmatch(r"CHART_\d+", row["chart_id"])
            assert isinstance(row["chart_set_id"], int)
            assert row["chamber_id"] == "CH1"
            assert row["recipe_id"] == "RECIPE_A"
            assert row["parameter"] == "dc_bias"
            assert row["step_no"] == 1
            assert row["feature_type"] == "mean"
            assert row["lcl"] == row["critical_lcl"]
            assert row["ucl"] == row["critical_ucl"]
            assert isinstance(row["version"], int)
            assert row["version"] >= 1
            assert re.fullmatch(
                r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d{3}Z",
                row["updated_at"],
            )

        active_row = next(item for item in rows if item["tool_id"] == tool_active)
        inactive_row = next(item for item in rows if item["tool_id"] == tool_inactive)
        assert active_row["is_active"] is True
        assert inactive_row["is_active"] is False
    finally:
        _cleanup_seeded_chart_rows(seeded)


def test_get_charts_supports_tool_filter_and_active_only(client: TestClient) -> None:
    """tool_id フィルタと active_only フィルタが機能することを確認する。"""
    seeded = _seed_chart_rows_for_get_charts()
    tool_active = seeded.tool_primary
    tool_inactive = seeded.tool_secondary
    try:
        filtered = client.get("/charts", params={"tool_id": tool_active})
        assert filtered.status_code == 200
        filtered_data = filtered.json()["data"]
        assert filtered_data
        assert all(item["tool_id"] == tool_active for item in filtered_data)
        assert all(item["tool_id"] != tool_inactive for item in filtered_data)

        active_only = client.get("/charts", params={"active_only": True})
        assert active_only.status_code == 200
        active_only_data = active_only.json()["data"]
        assert all(item["is_active"] is True for item in active_only_data)
        assert any(item["tool_id"] == tool_active for item in active_only_data)
        assert all(item["tool_id"] != tool_inactive for item in active_only_data)
    finally:
        _cleanup_seeded_chart_rows(seeded)


def test_get_charts_supports_chamber_filter(client: TestClient) -> None:
    """chamber_id フィルタが機能することを確認する。"""
    seeded = _seed_chart_rows_for_filter_tests()
    tool_match = seeded.tool_primary
    tool_other = seeded.tool_secondary
    try:
        res = client.get("/charts", params={"chamber_id": "CH_FILTER_MATCH"})
        assert res.status_code == 200
        data = [item for item in res.json()["data"] if item["tool_id"] in {tool_match, tool_other}]
        _assert_all_rows_match(data, "chamber_id", "CH_FILTER_MATCH")
    finally:
        _cleanup_seeded_chart_rows(seeded)


def test_get_charts_supports_recipe_filter(client: TestClient) -> None:
    """recipe_id フィルタが機能することを確認する。"""
    seeded = _seed_chart_rows_for_filter_tests()
    tool_match = seeded.tool_primary
    tool_other = seeded.tool_secondary
    try:
        res = client.get("/charts", params={"recipe_id": "RECIPE_FILTER_MATCH"})
        assert res.status_code == 200
        data = [item for item in res.json()["data"] if item["tool_id"] in {tool_match, tool_other}]
        _assert_all_rows_match(data, "recipe_id", "RECIPE_FILTER_MATCH")
    finally:
        _cleanup_seeded_chart_rows(seeded)


def test_get_charts_supports_parameter_filter(client: TestClient) -> None:
    """parameter フィルタが機能することを確認する。"""
    seeded = _seed_chart_rows_for_filter_tests()
    tool_match = seeded.tool_primary
    tool_other = seeded.tool_secondary
    try:
        res = client.get("/charts", params={"parameter": "dc_bias"})
        assert res.status_code == 200
        data = [item for item in res.json()["data"] if item["tool_id"] in {tool_match, tool_other}]
        _assert_all_rows_match(data, "parameter", "dc_bias")
    finally:
        _cleanup_seeded_chart_rows(seeded)


def test_get_charts_supports_positive_step_no_filter(client: TestClient) -> None:
    """step_no 正数フィルタが機能することを確認する。"""
    seeded = _seed_chart_rows_for_filter_tests()
    tool_match = seeded.tool_primary
    tool_other = seeded.tool_secondary
    try:
        res = client.get("/charts", params={"step_no": 2})
        assert res.status_code == 200
        data = [item for item in res.json()["data"] if item["tool_id"] in {tool_match, tool_other}]
        _assert_all_rows_match(data, "step_no", 2)
    finally:
        _cleanup_seeded_chart_rows(seeded)


def test_get_charts_supports_feature_type_filter(client: TestClient) -> None:
    """feature_type フィルタが機能することを確認する。"""
    seeded = _seed_chart_rows_for_filter_tests()
    tool_match = seeded.tool_primary
    tool_other = seeded.tool_secondary
    try:
        res = client.get("/charts", params={"feature_type": "mean"})
        assert res.status_code == 200
        data = [item for item in res.json()["data"] if item["tool_id"] in {tool_match, tool_other}]
        _assert_all_rows_match(data, "feature_type", "mean")
    finally:
        _cleanup_seeded_chart_rows(seeded)


def test_get_charts_rejects_negative_step_no(client: TestClient) -> None:
    """step_no が負数のとき 422 を返すことを確認する。"""
    res = client.get("/charts", params={"step_no": -1})

    assert res.status_code == 422
