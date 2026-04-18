from __future__ import annotations

import json
import logging
import re
import sqlite3
from collections.abc import Iterator
from dataclasses import dataclass
from datetime import UTC, datetime
from uuid import uuid4

import pytest
from fastapi.testclient import TestClient

from portfolio_fdc.db_api.db import MAIN_DB, _connect, _init_schema
from tests.test_utils import assert_validation_error_envelope


@dataclass(frozen=True)
class SeededJudgeResultsContext:
    process_id_with_lot: str
    process_id_without_lot: str
    recipe_id: str
    chart_id: str
    result_id_with_lot: str


@pytest.fixture
def seeded_judge_results_context() -> Iterator[SeededJudgeResultsContext]:
    """GET /judge/results テスト用に ProcessInfo/JudgementResults を投入する。"""
    _init_schema(MAIN_DB)
    suffix = uuid4().hex[:10]
    process_id_with_lot = f"P_JUDGE_LOT_{suffix}"
    process_id_without_lot = f"P_JUDGE_NA_{suffix}"
    recipe_id = f"RECIPE_JUDGE_{suffix}"
    chart_set_name = f"judge_result_chart_set_{suffix}"

    con = _connect(MAIN_DB)
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

        now = datetime.now(UTC).isoformat()
        con.execute(
            "INSERT INTO ChartSet(name, note, created_at, created_by) VALUES (?, ?, ?, ?)",
            (chart_set_name, "test", now, "test"),
        )
        chart_set_id = int(con.execute("SELECT last_insert_rowid()").fetchone()[0])
        con.execute(
            """
            INSERT INTO ChartsV2(
                chart_set_id, tool_id, chamber_id, recipe_id, parameter,
                step_no, feature_type, warn_low, warn_high, crit_low, crit_high,
                updated_at, updated_by, update_reason, update_source
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                chart_set_id,
                f"TOOL_{suffix}",
                "CH1",
                recipe_id,
                "dc_bias",
                1,
                "mean",
                1.4,
                2.6,
                1.2,
                2.8,
                now,
                "test",
                "seed",
                "test",
            ),
        )
        chart_pk = int(con.execute("SELECT last_insert_rowid()").fetchone()[0])
        chart_id = f"CHART_{chart_pk}"

        warn_row_cursor = con.execute(
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
                        "chart_id": chart_id,
                        "parameter": "dc_bias",
                        "step_no": 1,
                        "feature_type": "mean",
                        "feature_value": 2.72,
                        "stop_api_called": False,
                        "stop_api_status": "NOT_CALLED",
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
        warn_row_id = warn_row_cursor.lastrowid
        if warn_row_id is None:
            raise RuntimeError("Failed to seed JudgementResults row")
        result_id_with_lot = f"JR_{warn_row_id}"

        yield SeededJudgeResultsContext(
            process_id_with_lot=process_id_with_lot,
            process_id_without_lot=process_id_without_lot,
            recipe_id=recipe_id,
            chart_id=chart_id,
            result_id_with_lot=result_id_with_lot,
        )
    finally:
        cleanup = _connect(MAIN_DB)
        try:
            cleanup.execute(
                "DELETE FROM JudgementResults WHERE process_id IN (?, ?)",
                (process_id_with_lot, process_id_without_lot),
            )
            cleanup.execute(
                "DELETE FROM ProcessInfo WHERE process_id IN (?, ?)",
                (process_id_with_lot, process_id_without_lot),
            )
            cleanup.execute(
                (
                    "DELETE FROM ChartsV2 WHERE chart_set_id IN "
                    "(SELECT chart_set_id FROM ChartSet WHERE name = ?)"
                ),
                (chart_set_name,),
            )
            cleanup.execute(
                "DELETE FROM ChartSet WHERE name = ?",
                (chart_set_name,),
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
    assert warn_row["chart_id"] == seeded.chart_id
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


def test_get_judge_result_by_id_returns_contract_fields(
    client: TestClient,
    seeded_judge_results_context: SeededJudgeResultsContext,
) -> None:
    """GET /judge/results/{result_id} が契約フィールドと詳細情報を返す。"""
    seeded = seeded_judge_results_context

    res = client.get(f"/judge/results/{seeded.result_id_with_lot}")

    assert res.status_code == 200
    body = res.json()
    assert body["ok"] is True
    data = body["data"]

    expected_keys = {
        "result_id",
        "chart_id",
        "process_id",
        "lot_id",
        "wafer_id",
        "tool_id",
        "chamber_id",
        "recipe_id",
        "parameter",
        "step_no",
        "feature_type",
        "feature_value",
        "warning_lcl",
        "warning_ucl",
        "critical_lcl",
        "critical_ucl",
        "level",
        "judged_at",
        "process_start_ts",
        "stop_api_called",
        "stop_api_status",
    }

    assert expected_keys.issubset(data.keys())
    assert data["result_id"] == seeded.result_id_with_lot
    assert data["chart_id"] == seeded.chart_id
    assert data["process_id"] == seeded.process_id_with_lot
    assert data["level"] == "WARN"
    assert data["parameter"] == "dc_bias"
    assert data["step_no"] == 1
    assert data["feature_type"] == "mean"
    assert data["feature_value"] == 2.72
    assert data["warning_lcl"] == 1.4
    assert data["warning_ucl"] == 2.6
    assert data["critical_lcl"] == 1.2
    assert data["critical_ucl"] == 2.8
    assert data["stop_api_called"] is False
    assert data["stop_api_status"] == "NOT_CALLED"
    assert re.fullmatch(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d{3}Z", data["judged_at"])
    assert re.fullmatch(
        r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d{3}Z",
        data["process_start_ts"],
    )
    assert data["judged_at"] == "2026-04-17T00:00:00.123Z"


def test_get_judge_result_by_id_returns_not_found_envelope(client: TestClient) -> None:
    """未知 result_id は契約どおり 404 NOT_FOUND envelope を返す。"""
    res = client.get("/judge/results/JR_999999999")

    assert res.status_code == 404
    body = res.json()
    assert body == {
        "ok": False,
        "error": {
            "code": "NOT_FOUND",
            "message": "judge result not found",
            "details": {"result_id": "JR_999999999"},
        },
    }


def test_get_judge_result_by_id_returns_422_for_invalid_result_id(client: TestClient) -> None:
    """契約外 result_id 形式は 422 の validation error を返す。"""
    res = client.get("/judge/results/INVALID")

    assert res.status_code == 422
    assert_validation_error_envelope(
        res.json(),
        expected_loc_fragment="result_id",
        expected_message_fragment="pattern",
    )


def test_get_judge_result_by_id_returns_400_for_non_positive_result_id(client: TestClient) -> None:
    """形式は正しいが 1 未満の result_id は 400 を返す。"""
    res = client.get("/judge/results/JR_0")

    assert res.status_code == 400
    assert res.json() == {"detail": "Invalid result_id"}


def test_get_judge_result_by_id_does_not_enrich_thresholds_for_fractional_chart_id(
    client: TestClient,
    seeded_judge_results_context: SeededJudgeResultsContext,
) -> None:
    """float chart_id は正規化して返すが detail 閾値補完には使わない。"""
    seeded = seeded_judge_results_context
    chart_pk = int(seeded.chart_id.split("_", maxsplit=1)[1])

    con = _connect(MAIN_DB)
    try:
        cursor = con.execute(
            """
            INSERT INTO JudgementResults(
                process_id, tool_id, chamber_id, recipe_id, status, judged_at, message_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                seeded.process_id_without_lot,
                "TOOL_FRACTIONAL_CHART",
                "CH1",
                seeded.recipe_id,
                "WARN",
                "2026-04-17T02:00:00+00:00",
                json.dumps(
                    {
                        "chart_id": chart_pk + 0.7,
                        "feature_value": 9.99,
                    }
                ),
            ),
        )
        con.commit()
        result_row_id = cursor.lastrowid
        if result_row_id is None:
            raise RuntimeError("Failed to seed fractional chart_id judge result")

        res = client.get(f"/judge/results/JR_{result_row_id}")

        assert res.status_code == 200
        data = res.json()["data"]
        assert data["chart_id"] == seeded.chart_id
        assert data["parameter"] is None
        assert data["step_no"] is None
        assert data["feature_type"] is None
        assert data["warning_lcl"] is None
        assert data["warning_ucl"] is None
        assert data["critical_lcl"] is None
        assert data["critical_ucl"] is None
        assert data["feature_value"] == 9.99
    finally:
        con.execute(
            "DELETE FROM JudgementResults WHERE process_id = ? AND tool_id = ?",
            (seeded.process_id_without_lot, "TOOL_FRACTIONAL_CHART"),
        )
        con.commit()
        con.close()


def test_get_judge_result_by_id_returns_500_when_detail_conversion_fails(
    client: TestClient,
    seeded_judge_results_context: SeededJudgeResultsContext,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """_to_judge_result_detail_view が None を返す場合（invalid 時刻）に HTTP 500 を返す。"""
    seeded = seeded_judge_results_context

    con = _connect(MAIN_DB)
    try:
        cursor = con.execute(
            """
            INSERT INTO JudgementResults(
                process_id, tool_id, chamber_id, recipe_id, status, judged_at, message_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                seeded.process_id_without_lot,
                "TOOL_DATA_CORRUPTION_TEST",
                "CH1",
                seeded.recipe_id,
                "WARN",
                "INVALID_TIMESTAMP",
                json.dumps(
                    {
                        "chart_id": "CHART_999",
                        "feature_value": 1.0,
                    }
                ),
            ),
        )
        con.commit()
        result_row_id = cursor.lastrowid
        if result_row_id is None:
            raise RuntimeError("Failed to seed invalid judge result")

        with caplog.at_level(logging.ERROR, logger="portfolio_fdc.db_api.app"):
            res = client.get(f"/judge/results/JR_{result_row_id}")

        assert res.status_code == 500
        body = res.json()
        expected = {"detail": "Internal server error"}
        assert body == expected
        assert "JUDGE_DATA_CORRUPTION" in caplog.text
    finally:
        con.execute(
            "DELETE FROM JudgementResults WHERE process_id = ? AND tool_id = ?",
            (seeded.process_id_without_lot, "TOOL_DATA_CORRUPTION_TEST"),
        )
        con.commit()
        con.close()


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


def test_get_judge_results_pagination_ignores_invalid_status_rows(
    client: TestClient,
    seeded_judge_results_context: SeededJudgeResultsContext,
) -> None:
    """invalid status 行が存在しても SQL 側で除外されページングが崩れないことを検証する。"""
    seeded = seeded_judge_results_context
    con = _connect(MAIN_DB)
    try:
        # judged_at を最新にして、SQL 除外が無いと先頭ページを壊すデータを作る。
        con.execute(
            """
            INSERT INTO JudgementResults(
                process_id, tool_id, chamber_id, recipe_id, status, judged_at, message_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                seeded.process_id_with_lot,
                "TOOL_INVALID_STATUS",
                "CH1",
                seeded.recipe_id,
                "BAD",
                "2026-04-17T23:59:59+00:00",
                json.dumps(
                    {
                        "chart_id": "CHART_999",
                        "step_no": 9,
                        "feature_type": "mean",
                        "feature_value": 9.99,
                    }
                ),
            ),
        )
        con.commit()

        res = client.get(
            "/judge/results",
            params={"recipe_id": seeded.recipe_id, "limit": 1, "offset": 0},
        )

        assert res.status_code == 200
        rows = res.json()["data"]
        assert len(rows) == 1
        assert rows[0]["level"] in {"OK", "WARN", "NG"}
        assert rows[0]["level"] != "BAD"
    finally:
        con.execute(
            "DELETE FROM JudgementResults WHERE recipe_id = ? AND status = ?",
            (seeded.recipe_id, "BAD"),
        )
        con.commit()
        con.close()


def test_get_judge_results_skips_row_with_invalid_judged_at(
    client: TestClient,
    seeded_judge_results_context: SeededJudgeResultsContext,
) -> None:
    """不正 judged_at を含む行があっても 500 にならず当該行をスキップする。"""
    seeded = seeded_judge_results_context
    con = _connect(MAIN_DB)
    try:
        con.execute(
            """
            INSERT INTO JudgementResults(
                process_id, tool_id, chamber_id, recipe_id, status, judged_at, message_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                seeded.process_id_without_lot,
                "TOOL_BAD_TS",
                "CH1",
                seeded.recipe_id,
                "OK",
                "NOT_A_TIMESTAMP",
                json.dumps(
                    {
                        "chart_id": "CHART_777",
                        "step_no": 7,
                        "feature_type": "mean",
                        "feature_value": 7.77,
                    }
                ),
            ),
        )
        con.commit()

        res = client.get(
            "/judge/results",
            params={"recipe_id": seeded.recipe_id, "limit": 1000},
        )

        assert res.status_code == 200
        rows = res.json()["data"]
        # Seeded valid rows (2) should remain; invalid judged_at row is skipped.
        assert len(rows) == 2
        assert all(row["chart_id"] != "CHART_777" for row in rows)
    finally:
        con.execute(
            "DELETE FROM JudgementResults WHERE recipe_id = ? AND tool_id = ?",
            (seeded.recipe_id, "TOOL_BAD_TS"),
        )
        con.commit()
        con.close()


def test_get_judge_results_skips_row_with_invalid_process_start_ts(
    client: TestClient,
    seeded_judge_results_context: SeededJudgeResultsContext,
) -> None:
    """不正 ProcessInfo.start_ts を持つ process の行を 500 なくスキップする。"""
    seeded = seeded_judge_results_context
    con = _connect(MAIN_DB)
    original_start_ts = con.execute(
        "SELECT start_ts FROM ProcessInfo WHERE process_id = ?",
        (seeded.process_id_with_lot,),
    ).fetchone()
    try:
        con.execute(
            "UPDATE ProcessInfo SET start_ts = ? WHERE process_id = ?",
            ("BAD_START_TS", seeded.process_id_with_lot),
        )
        con.commit()

        res = client.get(
            "/judge/results",
            params={"recipe_id": seeded.recipe_id, "limit": 1000},
        )

        assert res.status_code == 200
        rows = res.json()["data"]
        assert len(rows) == 1
        assert all(row["process_id"] != seeded.process_id_with_lot for row in rows)
    finally:
        if original_start_ts is not None:
            con.execute(
                "UPDATE ProcessInfo SET start_ts = ? WHERE process_id = ?",
                (original_start_ts[0], seeded.process_id_with_lot),
            )
        con.commit()
        con.close()


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
    judged_dt = datetime.fromisoformat(rows[0]["judged_at"].replace("Z", "+00:00"))
    from_dt = datetime.fromisoformat(from_ts.replace("Z", "+00:00"))
    to_dt = datetime.fromisoformat(to_ts.replace("Z", "+00:00"))
    assert from_dt <= judged_dt <= to_dt


def test_get_judge_results_returns_400_for_invalid_timestamp_range(client: TestClient) -> None:
    """from_ts > to_ts は 400 を返すことを検証する。"""
    from_ts = datetime(2026, 4, 17, 2, 0, tzinfo=UTC).isoformat()
    to_ts = datetime(2026, 4, 17, 1, 0, tzinfo=UTC).isoformat()

    res = client.get("/judge/results", params={"from_ts": from_ts, "to_ts": to_ts})

    assert res.status_code == 400
    assert res.json()["detail"] == "end_ts must be greater than or equal to start_ts"


def test_get_judge_results_returns_400_for_naive_from_ts(client: TestClient) -> None:
    """naive datetime は _normalize_query_datetime で 400 を返すことを検証する。"""
    res = client.get(
        "/judge/results",
        params={
            "from_ts": "2026-04-17T01:00:00",
            "to_ts": "2026-04-17T02:00:00",
        },
    )

    assert res.status_code == 400
    assert res.json()["detail"] == "from_ts and to_ts must be timezone-aware datetimes"


def test_get_judge_results_returns_400_for_from_ts_without_to_ts(client: TestClient) -> None:
    """from_ts のみ指定は 400 を返すことを検証する。"""
    res = client.get(
        "/judge/results",
        params={"from_ts": "2026-04-17T01:00:00+00:00"},
    )

    assert res.status_code == 400
    assert res.json()["detail"] == "from_ts and to_ts must be specified together"


def test_get_judge_results_returns_400_for_to_ts_without_from_ts(client: TestClient) -> None:
    """to_ts のみ指定は 400 を返すことを検証する。"""
    res = client.get(
        "/judge/results",
        params={"to_ts": "2026-04-17T01:00:00+00:00"},
    )

    assert res.status_code == 400
    assert res.json()["detail"] == "from_ts and to_ts must be specified together"


def test_get_judge_results_returns_422_for_invalid_level(client: TestClient) -> None:
    """契約外 level は 422 の validation error を返すことを検証する。"""
    res = client.get("/judge/results", params={"level": "BAD"})

    assert res.status_code == 422
    assert_validation_error_envelope(
        res.json(),
        expected_loc_fragment="query",
        expected_message_fragment="pattern",
    )


def test_foreign_key_constraint_prevents_orphaned_judge_results() -> None:
    """外部キー制約により、孤立 JudgementResults が作成できないことを検証する。

    設計: ProcessInfo と JudgementResults は1年の保存期間内に一緒に削除される。
    外部キー制約により、ProcessInfo が存在する場合のみ JudgementResults を作成でき、
    ProcessInfo 削除時には先に JudgementResults を削除する必要がある。
    これにより、INNER JOIN による無言除外を防ぐ。
    """
    _init_schema(MAIN_DB)
    suffix = uuid4().hex[:10]
    process_id = f"P_FK_TEST_{suffix}"

    con = _connect(MAIN_DB)
    try:
        # ProcessInfo に行を挿入
        con.execute(
            """
            INSERT INTO ProcessInfo(
                process_id, tool_id, chamber_id, recipe_id,
                start_ts, end_ts, raw_csv_path
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                process_id,
                "TOOL_FK",
                "CH1",
                "RECIPE_FK",
                "2026-04-17T00:00:00+00:00",
                "2026-04-17T00:05:00+00:00",
                f"data/{process_id}.csv",
            ),
        )
        con.commit()

        # ProcessInfo が存在する場合、JudgementResults 挿入は成功
        con.execute(
            """
            INSERT INTO JudgementResults(
                process_id, tool_id, chamber_id, recipe_id, status, judged_at, message_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                process_id,
                "TOOL_FK",
                "CH1",
                "RECIPE_FK",
                "OK",
                "2026-04-17T00:10:00+00:00",
                json.dumps({"chart_id": "CHART_1"}),
            ),
        )
        con.commit()

        # ProcessInfo を削除しようとしても、JudgementResults が参照しているため
        # 外部キー制約により IntegrityError が発生
        with pytest.raises(sqlite3.IntegrityError, match="FOREIGN KEY constraint failed"):
            con.execute("DELETE FROM ProcessInfo WHERE process_id = ?", (process_id,))
            con.commit()

        # ProcessInfo を削除するには、先に JudgementResults を削除する必要がある
        con.execute("DELETE FROM JudgementResults WHERE process_id = ?", (process_id,))
        con.commit()

        # 今度は ProcessInfo の削除に成功
        con.execute("DELETE FROM ProcessInfo WHERE process_id = ?", (process_id,))
        con.commit()

    finally:
        # クリーンアップ
        con.execute("DELETE FROM JudgementResults WHERE process_id = ?", (process_id,))
        con.execute("DELETE FROM ProcessInfo WHERE process_id = ?", (process_id,))
        con.commit()
        con.close()
