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
from tests.utils.test_utils import assert_validation_error_envelope


@dataclass(frozen=True)
class SeededJudgeResultsContext:
    process_id_with_lot: str
    process_id_without_lot: str
    recipe_id: str
    chart_id: str
    result_id_with_lot: str
    result_id_without_lot: str


def insert_process_info(
    process_id: str,
    tool_id: str,
    recipe_id: str,
    lot_id: str | None,
    wafer_id: str | None,
    start_ts: str,
    end_ts: str,
) -> None:
    """ProcessInfo 1 行を投入する。"""
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
                process_id,
                tool_id,
                "CH1",
                recipe_id,
                start_ts,
                end_ts,
                f"data/detail/{process_id}.csv",
                lot_id,
                wafer_id,
            ),
        )
        con.commit()
    finally:
        con.close()


def insert_chart_set_and_chart(
    chart_set_name: str,
    chart_attrs: dict[str, str | int | float],
) -> tuple[str, int]:
    """ChartSet/ChartsV2 を投入し、chart_id と chart_pk を返す。"""
    con = _connect(MAIN_DB)
    try:
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
                chart_attrs["tool_id"],
                chart_attrs.get("chamber_id", "CH1"),
                chart_attrs["recipe_id"],
                chart_attrs.get("parameter", "dc_bias"),
                chart_attrs.get("step_no", 1),
                chart_attrs.get("feature_type", "mean"),
                chart_attrs.get("warn_low", 1.4),
                chart_attrs.get("warn_high", 2.6),
                chart_attrs.get("crit_low", 1.2),
                chart_attrs.get("crit_high", 2.8),
                now,
                "test",
                "seed",
                "test",
            ),
        )
        chart_pk = int(con.execute("SELECT last_insert_rowid()").fetchone()[0])
        con.commit()
        return f"CHART_{chart_pk}", chart_pk
    finally:
        con.close()


def insert_judgement_result(
    process_id: str,
    tool_id: str,
    chamber_id: str,
    recipe_id: str,
    status: str,
    judged_at: str,
    message_json: dict[str, object],
) -> int:
    """JudgementResults 1 行を投入し、rowid を返す。"""
    con = _connect(MAIN_DB)
    try:
        cursor = con.execute(
            """
            INSERT INTO JudgementResults(
                process_id, tool_id, chamber_id, recipe_id, status, judged_at, message_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                process_id,
                tool_id,
                chamber_id,
                recipe_id,
                status,
                judged_at,
                json.dumps(message_json),
            ),
        )
        con.commit()
        row_id = cursor.lastrowid
        if row_id is None:
            raise RuntimeError("Failed to seed JudgementResults row")
        return int(row_id)
    finally:
        con.close()


def cleanup_seeded_judge_results(process_ids: tuple[str, str], chart_set_name: str) -> None:
    """seeded_judge_results_context で投入したデータを削除する。"""
    con = _connect(MAIN_DB)
    try:
        con.execute(
            "DELETE FROM JudgementResults WHERE process_id IN (?, ?)",
            process_ids,
        )
        con.execute(
            "DELETE FROM ProcessInfo WHERE process_id IN (?, ?)",
            process_ids,
        )
        con.execute(
            (
                "DELETE FROM ChartsV2 WHERE chart_set_id IN "
                "(SELECT chart_set_id FROM ChartSet WHERE name = ?)"
            ),
            (chart_set_name,),
        )
        con.execute(
            "DELETE FROM ChartSet WHERE name = ?",
            (chart_set_name,),
        )
        con.commit()
    finally:
        con.close()


@pytest.fixture
def seeded_judge_results_context() -> Iterator[SeededJudgeResultsContext]:
    """GET /judge/results テスト用に ProcessInfo/JudgementResults を投入する。"""
    _init_schema(MAIN_DB)
    suffix = uuid4().hex[:10]
    process_id_with_lot = f"P_JUDGE_LOT_{suffix}"
    process_id_without_lot = f"P_JUDGE_NA_{suffix}"
    recipe_id = f"RECIPE_JUDGE_{suffix}"
    chart_set_name = f"judge_result_chart_set_{suffix}"
    tool_id = f"TOOL_{suffix}"

    try:
        insert_process_info(
            process_id=process_id_with_lot,
            tool_id=tool_id,
            recipe_id=recipe_id,
            lot_id=f"LOT_{suffix}",
            wafer_id="W01",
            start_ts="2026-04-17T00:00:00+00:00",
            end_ts="2026-04-17T00:05:00+00:00",
        )
        insert_process_info(
            process_id=process_id_without_lot,
            tool_id=tool_id,
            recipe_id=recipe_id,
            lot_id=None,
            wafer_id=None,
            start_ts="2026-04-17T01:00:00+00:00",
            end_ts="2026-04-17T01:05:00+00:00",
        )

        chart_id, _ = insert_chart_set_and_chart(
            chart_set_name=chart_set_name,
            chart_attrs={
                "tool_id": tool_id,
                "recipe_id": recipe_id,
                "chamber_id": "CH1",
                "parameter": "dc_bias",
                "step_no": 1,
                "feature_type": "mean",
                "warn_low": 1.4,
                "warn_high": 2.6,
                "crit_low": 1.2,
                "crit_high": 2.8,
            },
        )

        warn_row_id = insert_judgement_result(
            process_id=process_id_with_lot,
            tool_id=tool_id,
            chamber_id="CH1",
            recipe_id=recipe_id,
            status="WARN",
            judged_at="2026-04-17T09:00:00.123999+09:00",
            message_json={
                "chart_id": chart_id,
                "parameter": "dc_bias",
                "step_no": 1,
                "feature_type": "mean",
                "feature_value": 2.72,
                "stop_api_called": False,
                "stop_api_status": "NOT_CALLED",
            },
        )
        ok_row_id = insert_judgement_result(
            process_id=process_id_without_lot,
            tool_id=tool_id,
            chamber_id="CH1",
            recipe_id=recipe_id,
            status="OK",
            judged_at="2026-04-17T01:10:00+00:00",
            message_json={
                "chart_id": "CHART_200",
                "step_no": 2,
                "feature_type": "std",
                "feature_value": 1.23,
            },
        )
        result_id_with_lot = f"JR_{warn_row_id}"
        result_id_without_lot = f"JR_{ok_row_id}"

        yield SeededJudgeResultsContext(
            process_id_with_lot=process_id_with_lot,
            process_id_without_lot=process_id_without_lot,
            recipe_id=recipe_id,
            chart_id=chart_id,
            result_id_with_lot=result_id_with_lot,
            result_id_without_lot=result_id_without_lot,
        )
    finally:
        cleanup_seeded_judge_results(
            process_ids=(process_id_with_lot, process_id_without_lot),
            chart_set_name=chart_set_name,
        )


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


def test_get_judge_result_by_id_returns_stop_api_true_status(
    client: TestClient,
    seeded_judge_results_context: SeededJudgeResultsContext,
) -> None:
    """stop_api_called=True の payload を detail がそのまま返す。"""
    seeded = seeded_judge_results_context

    result_row_id = insert_judgement_result(
        process_id=seeded.process_id_with_lot,
        tool_id="TOOL_STOP_API_TRUE",
        chamber_id="CH1",
        recipe_id=seeded.recipe_id,
        status="WARN",
        judged_at="2026-04-17T02:10:00+00:00",
        message_json={
            "chart_id": seeded.chart_id,
            "parameter": "dc_bias",
            "step_no": 1,
            "feature_type": "mean",
            "feature_value": 2.72,
            "stop_api_called": True,
            "stop_api_status": "CALLED_SUCCESS",
        },
    )

    con = _connect(MAIN_DB)
    try:
        res = client.get(f"/judge/results/JR_{result_row_id}")

        assert res.status_code == 200
        data = res.json()["data"]
        assert data["stop_api_called"] is True
        assert data["stop_api_status"] == "CALLED_SUCCESS"
    finally:
        con.execute(
            "DELETE FROM JudgementResults WHERE process_id = ? AND tool_id = ?",
            (seeded.process_id_with_lot, "TOOL_STOP_API_TRUE"),
        )
        con.commit()
        con.close()


def test_get_judge_result_by_id_returns_stop_api_defaults_when_fields_missing(
    client: TestClient,
    seeded_judge_results_context: SeededJudgeResultsContext,
) -> None:
    """stop_api_* 欠落時は既定値（False/NOT_CALLED）を返す。"""
    seeded = seeded_judge_results_context

    res = client.get(f"/judge/results/{seeded.result_id_without_lot}")

    assert res.status_code == 200
    data = res.json()["data"]
    assert data["stop_api_called"] is False
    assert data["stop_api_status"] == "NOT_CALLED"


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


def test_get_judge_result_by_id_normalizes_leading_zeros(
    client: TestClient,
    seeded_judge_results_context: SeededJudgeResultsContext,
) -> None:
    """先頭ゼロ付き result_id は同一 PK に正規化され、正規形で返る。"""
    seeded = seeded_judge_results_context
    result_pk = seeded.result_id_with_lot.split("_", maxsplit=1)[1]
    leading_zero_result_id = f"JR_0{result_pk}"

    res = client.get(f"/judge/results/{leading_zero_result_id}")

    assert res.status_code == 200
    body = res.json()
    assert body["ok"] is True
    assert body["data"]["result_id"] == seeded.result_id_with_lot
    assert body["data"]["process_id"] == seeded.process_id_with_lot
    assert body["data"]["chart_id"] == seeded.chart_id


def test_get_judge_result_by_id_does_not_enrich_thresholds_for_fractional_chart_id(
    client: TestClient,
    seeded_judge_results_context: SeededJudgeResultsContext,
) -> None:
    """float chart_id は正規化して返すが detail 閾値補完には使わない。"""
    seeded = seeded_judge_results_context
    chart_pk = int(seeded.chart_id.split("_", maxsplit=1)[1])

    result_row_id = insert_judgement_result(
        process_id=seeded.process_id_without_lot,
        tool_id="TOOL_FRACTIONAL_CHART",
        chamber_id="CH1",
        recipe_id=seeded.recipe_id,
        status="WARN",
        judged_at="2026-04-17T02:00:00+00:00",
        message_json={
            "chart_id": chart_pk + 0.7,
            "feature_value": 9.99,
        },
    )

    con = _connect(MAIN_DB)
    try:
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


def test_get_judge_result_by_id_returns_null_thresholds_when_chart_thresholds_are_null(
    client: TestClient,
    seeded_judge_results_context: SeededJudgeResultsContext,
) -> None:
    """一致する ChartsV2 行の閾値が NULL の場合は detail でも None を返す。"""
    seeded = seeded_judge_results_context
    chart_pk = int(seeded.chart_id.split("_", maxsplit=1)[1])

    con = _connect(MAIN_DB)
    try:
        con.execute(
            """
            UPDATE ChartsV2
            SET warn_low = NULL, warn_high = NULL, crit_low = NULL, crit_high = NULL
            WHERE id = ?
            """,
            (chart_pk,),
        )
        con.commit()

        res = client.get(f"/judge/results/{seeded.result_id_with_lot}")

        assert res.status_code == 200
        data = res.json()["data"]
        assert data["warning_lcl"] is None
        assert data["warning_ucl"] is None
        assert data["critical_lcl"] is None
        assert data["critical_ucl"] is None
    finally:
        con.close()


def test_get_judge_result_by_id_returns_500_when_detail_conversion_fails(
    client: TestClient,
    seeded_judge_results_context: SeededJudgeResultsContext,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """_to_judge_result_detail_view が None を返す場合（invalid 時刻）に HTTP 500 を返す。"""
    seeded = seeded_judge_results_context

    result_row_id = insert_judgement_result(
        process_id=seeded.process_id_without_lot,
        tool_id="TOOL_DATA_CORRUPTION_TEST",
        chamber_id="CH1",
        recipe_id=seeded.recipe_id,
        status="WARN",
        judged_at="INVALID_TIMESTAMP",
        message_json={
            "chart_id": "CHART_999",
            "feature_value": 1.0,
        },
    )

    con = _connect(MAIN_DB)
    try:
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


def test_get_judge_result_by_id_returns_500_when_process_start_ts_is_corrupted(
    client: TestClient,
    seeded_judge_results_context: SeededJudgeResultsContext,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """invalid ProcessInfo.start_ts で detail 変換が失敗した場合に HTTP 500 を返す。"""
    seeded = seeded_judge_results_context

    con = _connect(MAIN_DB)
    original_start_ts = con.execute(
        "SELECT start_ts FROM ProcessInfo WHERE process_id = ?",
        (seeded.process_id_without_lot,),
    ).fetchone()
    try:
        result_row_id = insert_judgement_result(
            process_id=seeded.process_id_without_lot,
            tool_id="TOOL_DATA_CORRUPTION_TEST",
            chamber_id="CH1",
            recipe_id=seeded.recipe_id,
            status="WARN",
            judged_at="2026-04-17T02:30:00+00:00",
            message_json={
                "chart_id": "CHART_999",
                "feature_value": 1.0,
            },
        )
        con.execute(
            "UPDATE ProcessInfo SET start_ts = ? WHERE process_id = ?",
            ("INVALID_TIMESTAMP", seeded.process_id_without_lot),
        )
        con.commit()

        with caplog.at_level(logging.ERROR, logger="portfolio_fdc.db_api.app"):
            res = client.get(f"/judge/results/JR_{result_row_id}")

        assert res.status_code == 500
        body = res.json()
        expected = {"detail": "Internal server error"}
        assert body == expected
        assert "JUDGE_DATA_CORRUPTION" in caplog.text
    finally:
        if original_start_ts is not None:
            con.execute(
                "UPDATE ProcessInfo SET start_ts = ? WHERE process_id = ?",
                (original_start_ts[0], seeded.process_id_without_lot),
            )
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
