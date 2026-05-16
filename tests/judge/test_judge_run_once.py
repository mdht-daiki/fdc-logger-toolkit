from __future__ import annotations

import json
import sqlite3
from datetime import UTC, datetime
from pathlib import Path

import pytest

from portfolio_fdc.db_api.db import _init_schema
from portfolio_fdc.judge.run_once import JudgeEngine, run_once


def _connect(db_path: Path) -> sqlite3.Connection:
    con = sqlite3.connect(db_path.as_posix())
    con.execute("PRAGMA foreign_keys=ON;")
    return con


def _seed_chart_set_and_active_chart(db_path: Path) -> int:
    con = _connect(db_path)
    try:
        now = datetime.now(UTC).isoformat()
        con.execute(
            "INSERT INTO ChartSet(name, note, created_at, created_by) VALUES (?, ?, ?, ?)",
            ("judge-mvp", "test", now, "tester"),
        )
        chart_set_id = int(con.execute("SELECT last_insert_rowid()").fetchone()[0])
        con.execute(
            (
                "UPDATE ActiveChartSet SET chart_set_id = ?, updated_at = ?, "
                "updated_by = ? WHERE id = 1"
            ),
            (chart_set_id, now, "tester"),
        )
        con.commit()
        return chart_set_id
    finally:
        con.close()


def _insert_chart(
    db_path: Path,
    chart_set_id: int,
    *,
    tool_id: str,
    chamber_id: str,
    recipe_id: str,
    parameter: str,
    step_no: int,
    feature_type: str,
    warn_low: float,
    warn_high: float,
    crit_low: float,
    crit_high: float,
) -> str:
    con = _connect(db_path)
    try:
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
                tool_id,
                chamber_id,
                recipe_id,
                parameter,
                step_no,
                feature_type,
                warn_low,
                warn_high,
                crit_low,
                crit_high,
                "2026-05-13T00:00:00.000Z",
                "tester",
                "seed",
                "test",
            ),
        )
        chart_id = int(con.execute("SELECT last_insert_rowid()").fetchone()[0])
        con.commit()
        return f"CHART_{chart_id}"
    finally:
        con.close()


def _insert_process(
    db_path: Path,
    *,
    process_id: str,
    tool_id: str,
    chamber_id: str,
    recipe_id: str,
) -> None:
    con = _connect(db_path)
    try:
        con.execute(
            """
            INSERT INTO ProcessInfo(
                process_id, tool_id, chamber_id, recipe_id, start_ts, end_ts, raw_csv_path
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                process_id,
                tool_id,
                chamber_id,
                recipe_id,
                "2026-05-13T08:00:00+09:00",
                "2026-05-13T08:05:00+09:00",
                "data/raw/test.csv",
            ),
        )
        con.commit()
    finally:
        con.close()


def _insert_parameter(
    db_path: Path,
    *,
    process_id: str,
    parameter: str,
    step_no: int,
    feature_type: str,
    feature_value: float,
) -> None:
    con = _connect(db_path)
    try:
        con.execute(
            """
            INSERT INTO Parameters(process_id, parameter, step_no, feature_type, feature_value)
            VALUES (?, ?, ?, ?, ?)
            """,
            (process_id, parameter, step_no, feature_type, feature_value),
        )
        con.commit()
    finally:
        con.close()


def _read_judge_results(db_path: Path) -> list[tuple[str, dict[str, object]]]:
    con = _connect(db_path)
    try:
        rows = con.execute(
            """
            SELECT status, message_json
            FROM JudgementResults
            ORDER BY id
            """,
        ).fetchall()
        return [(str(row[0]), json.loads(str(row[1]))) for row in rows]
    finally:
        con.close()


def test_judge_run_once_records_warn_and_ng_results(tmp_path: Path) -> None:
    db_path = tmp_path / "main.db"
    _init_schema(db_path)
    chart_set_id = _seed_chart_set_and_active_chart(db_path)

    tool_id = "TOOL_JUDGE"
    chamber_id = "CH1"
    recipe_id = "RECIPE_JUDGE"
    process_id = "P_JUDGE_1"

    _insert_chart(
        db_path,
        chart_set_id,
        tool_id=tool_id,
        chamber_id=chamber_id,
        recipe_id=recipe_id,
        parameter="dc_bias",
        step_no=1,
        feature_type="mean",
        warn_low=1.4,
        warn_high=2.6,
        crit_low=1.2,
        crit_high=2.8,
    )
    _insert_chart(
        db_path,
        chart_set_id,
        tool_id=tool_id,
        chamber_id=chamber_id,
        recipe_id=recipe_id,
        parameter="cl2_flow",
        step_no=2,
        feature_type="mean",
        warn_low=10.0,
        warn_high=20.0,
        crit_low=8.0,
        crit_high=22.0,
    )

    _insert_process(
        db_path,
        process_id=process_id,
        tool_id=tool_id,
        chamber_id=chamber_id,
        recipe_id=recipe_id,
    )
    _insert_parameter(
        db_path,
        process_id=process_id,
        parameter="dc_bias",
        step_no=1,
        feature_type="mean",
        feature_value=2.7,
    )
    _insert_parameter(
        db_path,
        process_id=process_id,
        parameter="cl2_flow",
        step_no=2,
        feature_type="mean",
        feature_value=23.5,
    )

    notifications: list[dict[str, object]] = []
    stop_calls: list[dict[str, object]] = []

    summary = JudgeEngine(
        db_path=db_path,
        notification_sink=notifications.append,
        stop_api_hook=stop_calls.append,
    ).run_once()

    assert summary.evaluated == 2
    assert summary.written == 3
    assert summary.notifications_attempted == 2
    assert summary.stop_api_attempted == 1
    assert summary.notifications_failed == 0
    assert summary.stop_api_failed == 0
    assert summary.skipped_without_chart == 0
    assert len(notifications) == 2
    assert len(stop_calls) == 1

    results = _read_judge_results(db_path)
    payload_by_chart_id = {payload["chart_id"]: (status, payload) for status, payload in results}

    assert payload_by_chart_id["CHART_1"][0] == "WARN"
    assert payload_by_chart_id["CHART_1"][1]["notification_status"] == "CALLED"
    assert payload_by_chart_id["CHART_1"][1]["stop_api_status"] == "NOT_CALLED"

    assert payload_by_chart_id["CHART_2"][0] == "NG"
    assert payload_by_chart_id["CHART_2"][1]["notification_status"] == "CALLED"
    assert payload_by_chart_id["CHART_2"][1]["stop_api_called"] is True
    assert payload_by_chart_id["CHART_2"][1]["stop_api_status"] == "CALLED"


def test_judge_run_once_marks_failed_stop_api_without_crashing(tmp_path: Path) -> None:
    db_path = tmp_path / "main.db"
    _init_schema(db_path)
    chart_set_id = _seed_chart_set_and_active_chart(db_path)

    tool_id = "TOOL_JUDGE_FAIL"
    chamber_id = "CH1"
    recipe_id = "RECIPE_JUDGE_FAIL"
    process_id = "P_JUDGE_FAIL"

    _insert_chart(
        db_path,
        chart_set_id,
        tool_id=tool_id,
        chamber_id=chamber_id,
        recipe_id=recipe_id,
        parameter="dc_bias",
        step_no=1,
        feature_type="mean",
        warn_low=1.4,
        warn_high=2.6,
        crit_low=1.2,
        crit_high=2.8,
    )
    _insert_process(
        db_path,
        process_id=process_id,
        tool_id=tool_id,
        chamber_id=chamber_id,
        recipe_id=recipe_id,
    )
    _insert_parameter(
        db_path,
        process_id=process_id,
        parameter="dc_bias",
        step_no=1,
        feature_type="mean",
        feature_value=3.2,
    )

    def _raise_stop_api(_payload: dict[str, object]) -> None:
        raise RuntimeError("stop api failed")

    summary = run_once(db_path=db_path, stop_api_hook=_raise_stop_api)

    assert summary.evaluated == 1
    assert summary.written == 2
    assert summary.notifications_attempted == 1
    assert summary.stop_api_attempted == 1
    assert summary.stop_api_failed == 1

    results = _read_judge_results(db_path)
    assert results[0][0] == "NG"
    assert results[0][1]["stop_api_status"] == "FAILED"
    assert results[0][1]["stop_api_called"] is True


@pytest.mark.parametrize(
    (
        "feature_value",
        "expected_status",
        "expected_notification_attempted",
        "expected_stop_attempted",
        "expected_written",
    ),
    [
        (1.4, "OK", 0, 0, 1),
        (2.6, "OK", 0, 0, 1),
        (1.2, "WARN", 1, 0, 1),
        (2.8, "WARN", 1, 0, 1),
        (1.39, "WARN", 1, 0, 1),
        (2.81, "NG", 1, 1, 2),
    ],
)
def test_judge_evaluate_boundaries(
    tmp_path: Path,
    feature_value: float,
    expected_status: str,
    expected_notification_attempted: int,
    expected_stop_attempted: int,
    expected_written: int,
) -> None:
    db_path = tmp_path / "main.db"
    _init_schema(db_path)
    chart_set_id = _seed_chart_set_and_active_chart(db_path)

    tool_id = "TOOL_BOUNDARY"
    chamber_id = "CH1"
    recipe_id = "RECIPE_BOUNDARY"
    process_id = f"P_BOUNDARY_{str(feature_value).replace('.', '_')}"

    _insert_chart(
        db_path,
        chart_set_id,
        tool_id=tool_id,
        chamber_id=chamber_id,
        recipe_id=recipe_id,
        parameter="dc_bias",
        step_no=1,
        feature_type="mean",
        warn_low=1.4,
        warn_high=2.6,
        crit_low=1.2,
        crit_high=2.8,
    )
    _insert_process(
        db_path,
        process_id=process_id,
        tool_id=tool_id,
        chamber_id=chamber_id,
        recipe_id=recipe_id,
    )
    _insert_parameter(
        db_path,
        process_id=process_id,
        parameter="dc_bias",
        step_no=1,
        feature_type="mean",
        feature_value=feature_value,
    )

    summary = JudgeEngine(db_path=db_path).run_once(process_id=process_id)

    assert summary.evaluated == 1
    assert summary.written == expected_written
    assert summary.notifications_attempted == expected_notification_attempted
    assert summary.stop_api_attempted == expected_stop_attempted

    results = _read_judge_results(db_path)
    assert len(results) == 1
    assert results[0][0] == expected_status


def test_judge_handles_timeout(tmp_path: Path) -> None:
    db_path = tmp_path / "main.db"
    _init_schema(db_path)
    chart_set_id = _seed_chart_set_and_active_chart(db_path)

    tool_id = "TOOL_TIMEOUT"
    chamber_id = "CH1"
    recipe_id = "RECIPE_TIMEOUT"

    _insert_chart(
        db_path,
        chart_set_id,
        tool_id=tool_id,
        chamber_id=chamber_id,
        recipe_id=recipe_id,
        parameter="warn_param",
        step_no=1,
        feature_type="mean",
        warn_low=1.4,
        warn_high=2.6,
        crit_low=1.2,
        crit_high=2.8,
    )
    _insert_chart(
        db_path,
        chart_set_id,
        tool_id=tool_id,
        chamber_id=chamber_id,
        recipe_id=recipe_id,
        parameter="ng_param",
        step_no=1,
        feature_type="mean",
        warn_low=10.0,
        warn_high=20.0,
        crit_low=8.0,
        crit_high=22.0,
    )

    _insert_process(
        db_path,
        process_id="P_TIMEOUT_WARN",
        tool_id=tool_id,
        chamber_id=chamber_id,
        recipe_id=recipe_id,
    )
    _insert_parameter(
        db_path,
        process_id="P_TIMEOUT_WARN",
        parameter="warn_param",
        step_no=1,
        feature_type="mean",
        feature_value=2.7,
    )

    _insert_process(
        db_path,
        process_id="P_TIMEOUT_NG",
        tool_id=tool_id,
        chamber_id=chamber_id,
        recipe_id=recipe_id,
    )
    _insert_parameter(
        db_path,
        process_id="P_TIMEOUT_NG",
        parameter="ng_param",
        step_no=1,
        feature_type="mean",
        feature_value=23.0,
    )

    def _raise_timeout(_payload: dict[str, object]) -> None:
        raise TimeoutError("timeout")

    summary = run_once(
        db_path=db_path,
        notification_sink=_raise_timeout,
        stop_api_hook=_raise_timeout,
    )

    assert summary.evaluated == 2
    assert summary.written == 3
    assert summary.notifications_attempted == 2
    assert summary.notifications_failed == 2
    assert summary.stop_api_attempted == 1
    assert summary.stop_api_failed == 1

    results = _read_judge_results(db_path)
    by_process_id = {payload["process_id"]: (status, payload) for status, payload in results}

    assert by_process_id["P_TIMEOUT_WARN"][0] == "WARN"
    assert by_process_id["P_TIMEOUT_WARN"][1]["notification_status"] == "FAILED"

    assert by_process_id["P_TIMEOUT_NG"][0] == "NG"
    assert by_process_id["P_TIMEOUT_NG"][1]["notification_status"] == "FAILED"
    assert by_process_id["P_TIMEOUT_NG"][1]["stop_api_status"] == "FAILED"


def test_judge_run_once_skips_already_judged_processes(tmp_path: Path) -> None:
    db_path = tmp_path / "main.db"
    _init_schema(db_path)
    chart_set_id = _seed_chart_set_and_active_chart(db_path)

    tool_id = "TOOL_SKIP"
    chamber_id = "CH1"
    recipe_id = "RECIPE_SKIP"
    process_id = "P_SKIP_1"

    _insert_chart(
        db_path,
        chart_set_id,
        tool_id=tool_id,
        chamber_id=chamber_id,
        recipe_id=recipe_id,
        parameter="dc_bias",
        step_no=1,
        feature_type="mean",
        warn_low=1.4,
        warn_high=2.6,
        crit_low=1.2,
        crit_high=2.8,
    )
    _insert_process(
        db_path,
        process_id=process_id,
        tool_id=tool_id,
        chamber_id=chamber_id,
        recipe_id=recipe_id,
    )
    _insert_parameter(
        db_path,
        process_id=process_id,
        parameter="dc_bias",
        step_no=1,
        feature_type="mean",
        feature_value=2.9,
    )

    first = run_once(db_path=db_path)
    second = run_once(db_path=db_path)

    assert first.evaluated == 1
    assert second.evaluated == 0

    results = _read_judge_results(db_path)
    assert len(results) == 1


def test_judge_run_once_skips_already_judged_process_with_explicit_id(tmp_path: Path) -> None:
    db_path = tmp_path / "main.db"
    _init_schema(db_path)
    chart_set_id = _seed_chart_set_and_active_chart(db_path)

    tool_id = "TOOL_SKIP_BY_ID"
    chamber_id = "CH1"
    recipe_id = "RECIPE_SKIP_BY_ID"
    process_id = "P_SKIP_BY_ID_1"

    _insert_chart(
        db_path,
        chart_set_id,
        tool_id=tool_id,
        chamber_id=chamber_id,
        recipe_id=recipe_id,
        parameter="dc_bias",
        step_no=1,
        feature_type="mean",
        warn_low=1.4,
        warn_high=2.6,
        crit_low=1.2,
        crit_high=2.8,
    )
    _insert_process(
        db_path,
        process_id=process_id,
        tool_id=tool_id,
        chamber_id=chamber_id,
        recipe_id=recipe_id,
    )
    _insert_parameter(
        db_path,
        process_id=process_id,
        parameter="dc_bias",
        step_no=1,
        feature_type="mean",
        feature_value=2.9,
    )

    first = run_once(db_path=db_path, process_id=process_id)
    second = run_once(db_path=db_path, process_id=process_id)

    assert first.evaluated == 1
    assert second.evaluated == 0

    results = _read_judge_results(db_path)
    assert len(results) == 1
