from __future__ import annotations

import json
import re
import sqlite3
from collections.abc import Iterator
from dataclasses import dataclass
from datetime import UTC, datetime
from uuid import uuid4

import pytest
from fastapi.testclient import TestClient

from portfolio_fdc.db_api.db import MAIN_DB, _init_schema
from tests.test_utils import assert_validation_error_envelope


@dataclass(frozen=True)
class SeededJudgeResultsContext:
    process_id_with_lot: str
    process_id_without_lot: str
    recipe_id: str
    chart_id: str


@pytest.fixture
def seeded_judge_results_context() -> Iterator[SeededJudgeResultsContext]:
    """GET /judge/results テスト用に ProcessInfo/JudgementResults を投入する。"""
    _init_schema(MAIN_DB)
    suffix = uuid4().hex[:10]
    process_id_with_lot = f"P_JUDGE_LOT_{suffix}"
    process_id_without_lot = f"P_JUDGE_NA_{suffix}"
    recipe_id = f"RECIPE_JUDGE_{suffix}"

    con = sqlite3.connect(MAIN_DB.as_posix())
    try:
        con.execute(
            """
            INSERT INTO ProcessInfo(
                process_id, tool_id, chamber_id, recipe_id,
                start_ts, end_ts, raw_csv_path, lot_id, wafer_id
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                process_id_with_lot,
                f"TOOL_{suffix}",
                "CH1",
                recipe_id,
                "2026-04-17T00:00:00+00:00",
                "2026-04-17T00:05:00+00:00",
                f"data/detail/{process_id_with_lot}.csv",
                f"LOT_{suffix}",
                "W01",
            ),
        )
        con.execute(
            """
            INSERT INTO ProcessInfo(
                process_id, tool_id, chamber_id, recipe_id,
                start_ts, end_ts, raw_csv_path, lot_id, wafer_id
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                process_id_without_lot,
                f"TOOL_{suffix}",
                "CH1",
                recipe_id,
                "2026-04-17T01:00:00+00:00",
                "2026-04-17T01:05:00+00:00",
                f"data/detail/{process_id_without_lot}.csv",
                None,
                None,
            ),
        )

        con.execute(
            """
            INSERT INTO JudgementResults(
                process_id, tool_id, chamber_id, recipe_id, status, judged_at, message_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                process_id_with_lot,
                f"TOOL_{suffix}",
                "CH1",
                recipe_id,
                "WARN",
                "2026-04-17T09:00:00.123999+09:00",
                json.dumps(
                    {
                        "chart_id": "CHART_100",
                        "step_no": 1,
                        "feature_type": "mean",
                        "feature_value": 2.72,
                    }
                ),
            ),
        )
        con.execute(
            """
            INSERT INTO JudgementResults(
                process_id, tool_id, chamber_id, recipe_id, status, judged_at, message_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                process_id_without_lot,
                f"TOOL_{suffix}",
                "CH1",
                recipe_id,
                "OK",
                "2026-04-17T01:10:00+00:00",
                json.dumps(
                    {
                        "chart_id": "CHART_200",
                        "step_no": 2,
                        "feature_type": "std",
                        "feature_value": 1.23,
                    }
                ),
            ),
        )
        con.commit()

        yield SeededJudgeResultsContext(
            process_id_with_lot=process_id_with_lot,
            process_id_without_lot=process_id_without_lot,
            recipe_id=recipe_id,
            chart_id="CHART_100",
        )
    finally:
        cleanup = sqlite3.connect(MAIN_DB.as_posix())
        try:
            cleanup.execute(
                "DELETE FROM JudgementResults WHERE process_id IN (?, ?)",
                (process_id_with_lot, process_id_without_lot),
            )
            cleanup.execute(
                "DELETE FROM ProcessInfo WHERE process_id IN (?, ?)",
                (process_id_with_lot, process_id_without_lot),
            )
            cleanup.commit()
        finally:
            cleanup.close()
        con.close()


def test_get_judge_results_returns_contract_fields(
    client: TestClient,
    seeded_judge_results_context: SeededJudgeResultsContext,
) -> None:
    """GET /judge/results が契約フィールドと UTC ミリ秒時刻を返すことを検証する。"""
    seeded = seeded_judge_results_context

    res = client.get("/judge/results", params={"recipe_id": seeded.recipe_id, "limit": 1000})

    assert res.status_code == 200
    body = res.json()
    assert body["ok"] is True
    rows = body["data"]
    assert len(rows) == 2

    expected_keys = {
        "result_id",
        "chart_id",
        "process_id",
        "lot_id",
        "wafer_id",
        "recipe_id",
        "step_no",
        "feature_type",
        "feature_value",
        "level",
        "judged_at",
        "process_start_ts",
    }

    for row in rows:
        assert expected_keys.issubset(row.keys())
        assert re.fullmatch(r"JR_\d+", row["result_id"])
        assert row["level"] in {"OK", "WARN", "NG"}
        assert re.fullmatch(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d{3}Z", row["judged_at"])
        assert re.fullmatch(
            r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d{3}Z",
            row["process_start_ts"],
        )

    warn_row = next(item for item in rows if item["level"] == "WARN")
    assert warn_row["chart_id"] == "CHART_100"
    assert warn_row["feature_type"] == "mean"
    assert warn_row["feature_value"] == 2.72
    assert warn_row["judged_at"] == "2026-04-17T00:00:00.123Z"

    missing_lot_row = next(
        item for item in rows if item["process_id"] == seeded.process_id_without_lot
    )
    assert missing_lot_row["lot_id"] is None
    assert missing_lot_row["wafer_id"] is None


def test_get_judge_results_supports_filters(
    client: TestClient,
    seeded_judge_results_context: SeededJudgeResultsContext,
) -> None:
    """chart_id + level フィルタが機能することを検証する。"""
    seeded = seeded_judge_results_context

    res = client.get(
        "/judge/results",
        params={
            "recipe_id": seeded.recipe_id,
            "chart_id": seeded.chart_id,
            "level": "WARN",
        },
    )

    assert res.status_code == 200
    rows = res.json()["data"]
    assert len(rows) == 1
    assert rows[0]["chart_id"] == seeded.chart_id
    assert rows[0]["level"] == "WARN"


def test_get_judge_results_supports_pagination(
    client: TestClient,
    seeded_judge_results_context: SeededJudgeResultsContext,
) -> None:
    """limit/offset のページングが機能することを検証する。"""
    seeded = seeded_judge_results_context

    first_page = client.get(
        "/judge/results",
        params={"recipe_id": seeded.recipe_id, "limit": 1, "offset": 0},
    )
    second_page = client.get(
        "/judge/results",
        params={"recipe_id": seeded.recipe_id, "limit": 1, "offset": 1},
    )

    assert first_page.status_code == 200
    assert second_page.status_code == 200

    first_rows = first_page.json()["data"]
    second_rows = second_page.json()["data"]
    assert len(first_rows) == 1
    assert len(second_rows) == 1
    assert first_rows[0]["result_id"] != second_rows[0]["result_id"]


def test_get_judge_results_supports_combined_filters(
    client: TestClient,
    seeded_judge_results_context: SeededJudgeResultsContext,
) -> None:
    """process_id + lot_id + from_ts/to_ts の複合フィルタが機能することを検証する。"""
    seeded = seeded_judge_results_context

    from_ts = datetime(2026, 4, 16, 0, 0, tzinfo=UTC).isoformat()
    to_ts = datetime(2026, 4, 17, 10, 0, tzinfo=UTC).isoformat()

    res = client.get(
        "/judge/results",
        params={
            "process_id": seeded.process_id_with_lot,
            "lot_id": seeded.process_id_with_lot.replace("P_JUDGE_LOT_", "LOT_"),
            "from_ts": from_ts,
            "to_ts": to_ts,
        },
    )

    assert res.status_code == 200
    rows = res.json()["data"]
    assert len(rows) == 1
    assert rows[0]["process_id"] == seeded.process_id_with_lot
    assert rows[0]["lot_id"] == seeded.process_id_with_lot.replace("P_JUDGE_LOT_", "LOT_")
    assert rows[0]["judged_at"] >= from_ts.replace("+00:00", "Z")
    assert rows[0]["judged_at"] <= to_ts.replace("+00:00", "Z")


def test_get_judge_results_returns_400_for_invalid_timestamp_range(client: TestClient) -> None:
    """from_ts > to_ts は 400 を返すことを検証する。"""
    from_ts = datetime(2026, 4, 17, 2, 0, tzinfo=UTC).isoformat()
    to_ts = datetime(2026, 4, 17, 1, 0, tzinfo=UTC).isoformat()

    res = client.get("/judge/results", params={"from_ts": from_ts, "to_ts": to_ts})

    assert res.status_code == 400
    assert res.json()["detail"] == "end_ts must be greater than or equal to start_ts"


def test_get_judge_results_returns_422_for_invalid_level(client: TestClient) -> None:
    """契約外 level は 422 の validation error を返すことを検証する。"""
    res = client.get("/judge/results", params={"level": "BAD"})

    assert res.status_code == 422
    assert_validation_error_envelope(
        res.json(),
        expected_loc_fragment="query",
        expected_message_fragment="pattern",
    )
