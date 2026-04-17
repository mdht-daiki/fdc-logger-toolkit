from __future__ import annotations

import re
import sqlite3
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from datetime import UTC, datetime
from uuid import uuid4

import pytest
from fastapi.testclient import TestClient

from portfolio_fdc.db_api.datetime_util import to_utc_millis
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
class SeededChartsContext:
    """seed された chart テストデータの後処理用コンテキスト。"""

    tool_primary: str
    tool_secondary: str
    restore_active_set_id: int
    chart_set_ids: tuple[int, ...]


SeedBuilder = Callable[
    [sqlite3.Connection, str, str],
    tuple[str, str, tuple[int, ...], int],
]


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


@pytest.fixture
def seeded_chart_rows_for_get_charts() -> Iterator[SeededChartsContext]:
    """GET /charts テスト用に active/inactive の 2 chart を投入する。"""

    def _build_seed(
        con: sqlite3.Connection,
        now: str,
        suffix: str,
    ) -> tuple[str, str, tuple[int, ...], int]:
        tool_active = f"TOOL_GET_CHARTS_ACTIVE_{suffix}"
        tool_inactive = f"TOOL_GET_CHARTS_INACTIVE_{suffix}"

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
        return tool_active, tool_inactive, (active_set_id, inactive_set_id), active_set_id

    yield from _seed_chart_rows_fixture(_build_seed)


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
                "DELETE FROM ChartsHistory WHERE chart_set_id = ?",
                (chart_set_id,),
            )
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


@pytest.fixture
def seeded_chart_rows_for_filter_tests() -> Iterator[SeededChartsContext]:
    """各クエリフィルタ検証用に属性が異なる 2 chart を投入する。"""

    def _build_seed(
        con: sqlite3.Connection,
        now: str,
        suffix: str,
    ) -> tuple[str, str, tuple[int, ...], int]:
        tool_match = f"TOOL_FILTER_MATCH_{suffix}"
        tool_other = f"TOOL_FILTER_OTHER_{suffix}"

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
        return tool_match, tool_other, (chart_set_id,), chart_set_id

    yield from _seed_chart_rows_fixture(_build_seed)


def _seed_chart_rows_fixture(seed_builder: SeedBuilder) -> Iterator[SeededChartsContext]:
    """Chart seed fixture の共通セットアップ/teardown を提供する。"""
    _init_schema(MAIN_DB)
    con = sqlite3.connect(MAIN_DB.as_posix())
    context: SeededChartsContext | None = None
    try:
        now = datetime.now(UTC).isoformat()
        suffix = uuid4().hex

        prev_active_row = con.execute(
            "SELECT chart_set_id FROM ActiveChartSet WHERE id = 1"
        ).fetchone()
        if prev_active_row is None:
            raise RuntimeError("ActiveChartSet row with id=1 is required")
        prev_active_set_id = int(prev_active_row[0])

        (
            tool_primary,
            tool_secondary,
            chart_set_ids,
            active_chart_set_id,
        ) = seed_builder(con, now, suffix)

        con.execute(
            "INSERT OR REPLACE INTO ActiveChartSet(id, chart_set_id, updated_at, updated_by)"
            " VALUES (1, ?, ?, ?)",
            (active_chart_set_id, now, "test"),
        )

        context = SeededChartsContext(
            tool_primary=tool_primary,
            tool_secondary=tool_secondary,
            restore_active_set_id=prev_active_set_id,
            chart_set_ids=chart_set_ids,
        )

        con.commit()
        yield context
    finally:
        con.close()
        if context is not None:
            _cleanup_seeded_chart_rows(context)


def _assert_all_rows_match(data: list[dict[str, object]], key: str, expected: object) -> None:
    """返却された全要素が指定キーの期待値を満たすことを確認する。"""
    assert data
    assert all(item.get(key) == expected for item in data)


def _insert_chart_history(
    chart_set_id: int,
    *,
    tool_id: str,
    chamber_id: str,
    recipe_id: str,
    parameter: str,
    step_no: int,
    feature_type: str,
    count: int,
) -> None:
    """指定 chart key に対応する履歴レコードを件数分だけ投入する。"""
    con = sqlite3.connect(MAIN_DB.as_posix())
    try:
        for index in range(count):
            changed_at = (
                datetime(2026, 4, 14, 0, 0, index, tzinfo=UTC).isoformat().replace("+00:00", "Z")
            )
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
                    tool_id,
                    chamber_id,
                    recipe_id,
                    parameter,
                    step_no,
                    feature_type,
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
                    "history-seed",
                    "test",
                ),
            )
        con.commit()
    finally:
        con.close()


def test_get_charts_returns_chart_rows_with_contract_fields(
    client: TestClient,
    seeded_chart_rows_for_get_charts: SeededChartsContext,
) -> None:
    """GET /charts が契約フィールドを返すことを確認する。"""
    seeded = seeded_chart_rows_for_get_charts
    tool_active = seeded.tool_primary
    tool_inactive = seeded.tool_secondary
    expected_warning_ranges = {
        tool_active: (1.4, 2.6),
        tool_inactive: (1.0, 2.0),
    }
    expected_critical_ranges = {
        tool_active: (1.2, 2.8),
        tool_inactive: (0.8, 2.2),
    }
    expected_updated_at = {
        tool_active: "2026-04-14T00:00:00.000Z",
        tool_inactive: "2026-04-14T00:00:00.000Z",
    }
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
        expected_warning_lcl, expected_warning_ucl = expected_warning_ranges[row["tool_id"]]
        expected_critical_lcl, expected_critical_ucl = expected_critical_ranges[row["tool_id"]]
        assert isinstance(row["warning_lcl"], float)
        assert isinstance(row["warning_ucl"], float)
        assert row["warning_lcl"] == expected_warning_lcl
        assert row["warning_ucl"] == expected_warning_ucl
        assert row["critical_lcl"] == expected_critical_lcl
        assert row["critical_ucl"] == expected_critical_ucl
        assert row["lcl"] == expected_critical_lcl
        assert row["ucl"] == expected_critical_ucl
        assert isinstance(row["version"], int)
        assert row["version"] >= 1
        assert row["updated_at"] == expected_updated_at[row["tool_id"]]

    active_row = next(item for item in rows if item["tool_id"] == tool_active)
    inactive_row = next(item for item in rows if item["tool_id"] == tool_inactive)
    assert active_row["is_active"] is True
    assert inactive_row["is_active"] is False


def test_get_charts_computes_version_from_history_count(
    client: TestClient,
    seeded_chart_rows_for_get_charts: SeededChartsContext,
) -> None:
    """version が履歴件数 + 1 で返ることを確認する。"""
    seeded = seeded_chart_rows_for_get_charts
    tool_active = seeded.tool_primary
    active_set_id = seeded.chart_set_ids[0]
    _insert_chart_history(
        active_set_id,
        tool_id=tool_active,
        chamber_id="CH1",
        recipe_id="RECIPE_A",
        parameter="dc_bias",
        step_no=1,
        feature_type="mean",
        count=2,
    )

    res = client.get("/charts", params={"tool_id": tool_active})

    assert res.status_code == 200
    data = res.json()["data"]
    assert len(data) == 1
    assert all(item["tool_id"] == tool_active for item in data)
    assert data[0]["version"] == 3


def test_get_charts_supports_tool_filter_and_active_only(
    client: TestClient,
    seeded_chart_rows_for_get_charts: SeededChartsContext,
) -> None:
    """tool_id フィルタと active_only フィルタが機能することを確認する。"""
    seeded = seeded_chart_rows_for_get_charts
    tool_active = seeded.tool_primary
    tool_inactive = seeded.tool_secondary

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


def test_get_charts_handles_missing_active_chart_set_row(
    client: TestClient,
    seeded_chart_rows_for_get_charts: SeededChartsContext,
) -> None:
    """ActiveChartSet 未設定時は is_active=False, active_onlyは空になることを確認する。"""
    seeded = seeded_chart_rows_for_get_charts
    tool_active = seeded.tool_primary
    tool_inactive = seeded.tool_secondary

    con = sqlite3.connect(MAIN_DB.as_posix())
    try:
        con.execute("DELETE FROM ActiveChartSet WHERE id = 1")
        con.commit()
    finally:
        con.close()

    res = client.get("/charts")
    assert res.status_code == 200
    data = [item for item in res.json()["data"] if item["tool_id"] in {tool_active, tool_inactive}]
    assert len(data) == 2
    assert all(item["is_active"] is False for item in data)

    active_only = client.get("/charts", params={"active_only": True})
    assert active_only.status_code == 200
    active_only_data = [
        item
        for item in active_only.json()["data"]
        if item["tool_id"] in {tool_active, tool_inactive}
    ]
    assert active_only_data == []


def test_get_charts_supports_chamber_filter(
    client: TestClient,
    seeded_chart_rows_for_filter_tests: SeededChartsContext,
) -> None:
    """chamber_id フィルタが機能することを確認する。"""
    seeded = seeded_chart_rows_for_filter_tests
    tool_match = seeded.tool_primary
    tool_other = seeded.tool_secondary

    res = client.get("/charts", params={"chamber_id": "CH_FILTER_MATCH"})
    assert res.status_code == 200
    data = [item for item in res.json()["data"] if item["tool_id"] in {tool_match, tool_other}]
    _assert_all_rows_match(data, "chamber_id", "CH_FILTER_MATCH")


def test_get_charts_supports_recipe_filter(
    client: TestClient,
    seeded_chart_rows_for_filter_tests: SeededChartsContext,
) -> None:
    """recipe_id フィルタが機能することを確認する。"""
    seeded = seeded_chart_rows_for_filter_tests
    tool_match = seeded.tool_primary
    tool_other = seeded.tool_secondary

    res = client.get("/charts", params={"recipe_id": "RECIPE_FILTER_MATCH"})
    assert res.status_code == 200
    data = [item for item in res.json()["data"] if item["tool_id"] in {tool_match, tool_other}]
    _assert_all_rows_match(data, "recipe_id", "RECIPE_FILTER_MATCH")


def test_get_charts_supports_parameter_filter(
    client: TestClient,
    seeded_chart_rows_for_filter_tests: SeededChartsContext,
) -> None:
    """parameter フィルタが機能することを確認する。"""
    seeded = seeded_chart_rows_for_filter_tests
    tool_match = seeded.tool_primary
    tool_other = seeded.tool_secondary

    res = client.get("/charts", params={"parameter": "dc_bias"})
    assert res.status_code == 200
    data = [item for item in res.json()["data"] if item["tool_id"] in {tool_match, tool_other}]
    _assert_all_rows_match(data, "parameter", "dc_bias")


def test_get_charts_supports_positive_step_no_filter(
    client: TestClient,
    seeded_chart_rows_for_filter_tests: SeededChartsContext,
) -> None:
    """step_no 正数フィルタが機能することを確認する。"""
    seeded = seeded_chart_rows_for_filter_tests
    tool_match = seeded.tool_primary
    tool_other = seeded.tool_secondary

    res = client.get("/charts", params={"step_no": 2})
    assert res.status_code == 200
    data = [item for item in res.json()["data"] if item["tool_id"] in {tool_match, tool_other}]
    _assert_all_rows_match(data, "step_no", 2)


def test_get_charts_supports_feature_type_filter(
    client: TestClient,
    seeded_chart_rows_for_filter_tests: SeededChartsContext,
) -> None:
    """feature_type フィルタが機能することを確認する。"""
    seeded = seeded_chart_rows_for_filter_tests
    tool_match = seeded.tool_primary
    tool_other = seeded.tool_secondary

    res = client.get("/charts", params={"feature_type": "mean"})
    assert res.status_code == 200
    data = [item for item in res.json()["data"] if item["tool_id"] in {tool_match, tool_other}]
    _assert_all_rows_match(data, "feature_type", "mean")


def test_get_charts_supports_combined_chamber_and_recipe_filters(
    client: TestClient,
    seeded_chart_rows_for_filter_tests: SeededChartsContext,
) -> None:
    """chamber_id と recipe_id の同時指定で AND 条件が機能することを確認する。"""
    seeded = seeded_chart_rows_for_filter_tests
    tool_match = seeded.tool_primary
    tool_other = seeded.tool_secondary

    res = client.get(
        "/charts",
        params={
            "chamber_id": "CH_FILTER_MATCH",
            "recipe_id": "RECIPE_FILTER_MATCH",
        },
    )

    assert res.status_code == 200
    data = [item for item in res.json()["data"] if item["tool_id"] in {tool_match, tool_other}]
    assert len(data) == 1
    assert data[0]["tool_id"] == tool_match
    assert data[0]["chamber_id"] == "CH_FILTER_MATCH"
    assert data[0]["recipe_id"] == "RECIPE_FILTER_MATCH"


def test_get_charts_returns_empty_for_non_matching_combined_filters(
    client: TestClient,
    seeded_chart_rows_for_filter_tests: SeededChartsContext,
) -> None:
    """同時指定条件が交差しない場合、該当データが返らないことを確認する。"""
    seeded = seeded_chart_rows_for_filter_tests
    tool_match = seeded.tool_primary
    tool_other = seeded.tool_secondary

    res = client.get(
        "/charts",
        params={
            "chamber_id": "CH_FILTER_MATCH",
            "recipe_id": "RECIPE_FILTER_OTHER",
        },
    )

    assert res.status_code == 200
    data = [item for item in res.json()["data"] if item["tool_id"] in {tool_match, tool_other}]
    assert data == []


def test_get_charts_returns_empty_envelope_when_no_match(client: TestClient) -> None:
    """一致するチャートがない場合に ok:true / data:[] を返すことを確認する。"""
    res = client.get("/charts", params={"tool_id": "TOOL_NONEXISTENT_BOUNDARY"})

    assert res.status_code == 200
    body = res.json()
    assert body["ok"] is True
    assert body["data"] == []


def test_get_charts_accepts_step_no_zero(client: TestClient) -> None:
    """step_no=0 は ge=0 の境界値として 422 にならないことを確認する。"""
    res = client.get("/charts", params={"step_no": 0})

    assert res.status_code == 200
    body = res.json()
    assert body["ok"] is True
    assert isinstance(body["data"], list)


def test_get_charts_rejects_negative_step_no(client: TestClient) -> None:
    """step_no が負数のとき 422 を返すことを確認する。"""
    res = client.get("/charts", params={"step_no": -1})

    assert res.status_code == 422
    assert_validation_error_envelope(res.json(), expected_loc_fragment="step_no")


def test_get_charts_rejects_invalid_tool_id_pattern(client: TestClient) -> None:
    """tool_id が許可されない形式のとき 422 を返すことを確認する。"""
    res = client.get("/charts", params={"tool_id": "TOOL INVALID"})

    assert res.status_code == 422
    assert_validation_error_envelope(res.json(), expected_loc_fragment="tool_id")


def test_to_utc_millis_truncates_microseconds_not_rounds() -> None:
    """updated_at 正規化は四捨五入ではなくミリ秒切り捨てであることを確認する。"""
    assert to_utc_millis("2026-04-14T00:00:00.123999+00:00") == "2026-04-14T00:00:00.123Z"
    assert to_utc_millis("2026-04-14T00:00:00.123500+00:00") == "2026-04-14T00:00:00.123Z"


def test_to_utc_millis_normalizes_year_boundary() -> None:
    """年越し直前のタイムスタンプを UTC ミリ秒固定に正規化することを確認する。"""
    assert to_utc_millis("2025-12-31T23:59:59.999999+00:00") == "2025-12-31T23:59:59.999Z"


def test_to_utc_millis_converts_positive_offset_to_utc() -> None:
    """正のタイムゾーンオフセット（例: +09:00 JST）を UTC に変換することを確認する。"""
    # 2026-04-14T10:00:00.500+09:00 == 2026-04-14T01:00:00.500Z
    assert to_utc_millis("2026-04-14T10:00:00.500000+09:00") == "2026-04-14T01:00:00.500Z"


def test_to_utc_millis_converts_negative_offset_to_utc() -> None:
    """負のタイムゾーンオフセット（例: -05:00 EST）を UTC に変換することを確認する。"""
    # 2026-04-14T00:00:00.000-05:00 == 2026-04-14T05:00:00.000Z
    assert to_utc_millis("2026-04-14T00:00:00.000000-05:00") == "2026-04-14T05:00:00.000Z"


def test_to_utc_millis_treats_naive_datetime_as_utc() -> None:
    """タイムゾーン情報のない naive datetime は UTC として扱うことを確認する。"""
    assert to_utc_millis("2026-04-14T12:00:00.000000") == "2026-04-14T12:00:00.000Z"
