from __future__ import annotations

import sqlite3
from collections.abc import Iterator
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from uuid import uuid4

import pytest
from fastapi.testclient import TestClient

from portfolio_fdc.db_api import app as db_app
from portfolio_fdc.db_api.chart_repository import ChartRepository
from portfolio_fdc.db_api.db import MAIN_DB, _connect, _init_schema


@dataclass(frozen=True)
class SeededChartPointsContext:
    chart_set_id: int
    chart_id: str
    chart_pk: int
    process_ids: tuple[str, ...]


@dataclass(frozen=True)
class SeededChartOnlyContext:
    chart_set_id: int
    chart_id: str
    chart_pk: int


def _insert_chart_set(con: sqlite3.Connection, suffix: str) -> int:
    now = datetime.now(UTC).isoformat()
    con.execute(
        "INSERT INTO ChartSet(name, note, created_at, created_by) VALUES (?, ?, ?, ?)",
        (f"points_set_{suffix}", "test", now, "test"),
    )
    return int(con.execute("SELECT last_insert_rowid()").fetchone()[0])


def _insert_chart(
    con: sqlite3.Connection,
    chart_set_id: int,
    suffix: str,
) -> int:
    now = datetime.now(UTC).isoformat()
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
            f"TOOL_POINTS_{suffix}",
            "CH_POINTS",
            f"RECIPE_POINTS_{suffix}",
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
    return int(con.execute("SELECT last_insert_rowid()").fetchone()[0])


def _insert_process_and_parameter(
    con: sqlite3.Connection,
    process_id: str,
    suffix: str,
    *,
    start_ts: str,
    feature_value: float,
) -> None:
    con.execute(
        """
        INSERT INTO ProcessInfo(
            process_id, tool_id, chamber_id, recipe_id, start_ts, end_ts, raw_csv_path
        ) VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        (
            process_id,
            f"TOOL_POINTS_{suffix}",
            "CH_POINTS",
            f"RECIPE_POINTS_{suffix}",
            start_ts,
            start_ts,
            f"data/raw/{process_id}.csv",
        ),
    )
    con.execute(
        """
        INSERT INTO Parameters(process_id, parameter, step_no, feature_type, feature_value)
        VALUES (?, ?, ?, ?, ?)
        """,
        (process_id, "dc_bias", 1, "mean", feature_value),
    )


def _cleanup_chart_set(chart_set_id: int, process_ids: tuple[str, ...]) -> None:
    con = _connect(MAIN_DB)
    try:
        if process_ids:
            placeholders = ",".join("?" for _ in process_ids)
            con.execute(
                f"DELETE FROM Parameters WHERE process_id IN ({placeholders})",
                process_ids,
            )
            con.execute(
                f"DELETE FROM ProcessInfo WHERE process_id IN ({placeholders})",
                process_ids,
            )

        con.execute("DELETE FROM ChartsV2 WHERE chart_set_id = ?", (chart_set_id,))
        con.execute("DELETE FROM ChartSet WHERE chart_set_id = ?", (chart_set_id,))
        con.commit()
    finally:
        con.close()


@pytest.fixture
def seeded_chart_points_context() -> Iterator[SeededChartPointsContext]:
    _init_schema(MAIN_DB)
    suffix = uuid4().hex[:10]
    process_ids: tuple[str, ...] = ()
    chart_set_id: int | None = None

    con = _connect(MAIN_DB)
    try:
        # Use savepoint to ensure atomic setup and proper rollback on error
        con.execute("SAVEPOINT setup_context")
        try:
            chart_set_id = _insert_chart_set(con, suffix)
            chart_pk = _insert_chart(con, chart_set_id, suffix)

            base = datetime(2026, 4, 14, 0, 0, 0, tzinfo=UTC)
            p1 = f"p_points_{suffix}_1"
            p2 = f"p_points_{suffix}_2"
            p3 = f"p_points_{suffix}_3"
            process_ids = (p1, p2, p3)

            _insert_process_and_parameter(
                con,
                p1,
                suffix,
                start_ts=(base + timedelta(minutes=0)).isoformat().replace("+00:00", "Z"),
                feature_value=1.1,
            )
            _insert_process_and_parameter(
                con,
                p2,
                suffix,
                start_ts=(base + timedelta(minutes=1)).isoformat().replace("+00:00", "Z"),
                feature_value=1.2,
            )
            _insert_process_and_parameter(
                con,
                p3,
                suffix,
                start_ts=(base + timedelta(minutes=2)).isoformat().replace("+00:00", "Z"),
                feature_value=1.3,
            )

            con.commit()

            yield SeededChartPointsContext(
                chart_set_id=chart_set_id,
                chart_id=f"CHART_{chart_pk}",
                chart_pk=chart_pk,
                process_ids=process_ids,
            )
        except Exception:
            con.execute("ROLLBACK TO setup_context")
            raise
    finally:
        con.close()
        if chart_set_id is not None:
            _cleanup_chart_set(chart_set_id, process_ids)


@pytest.fixture
def seeded_chart_only_context() -> Iterator[SeededChartOnlyContext]:
    _init_schema(MAIN_DB)
    suffix = uuid4().hex[:10]
    chart_set_id: int | None = None

    con = _connect(MAIN_DB)
    try:
        # Use savepoint for atomic setup
        con.execute("SAVEPOINT setup_context")
        try:
            chart_set_id = _insert_chart_set(con, suffix)
            chart_pk = _insert_chart(con, chart_set_id, suffix)
            con.commit()

            yield SeededChartOnlyContext(
                chart_set_id=chart_set_id,
                chart_id=f"CHART_{chart_pk}",
                chart_pk=chart_pk,
            )
        except Exception:
            con.execute("ROLLBACK TO setup_context")
            raise
    finally:
        con.close()
        if chart_set_id is not None:
            _cleanup_chart_set(chart_set_id, ())


def _insert_waveform_process(process_id: str, raw_csv_path: str) -> None:
    con = _connect(MAIN_DB)
    try:
        con.execute(
            """
            INSERT INTO ProcessInfo(
                process_id, tool_id, chamber_id, recipe_id, start_ts, end_ts, raw_csv_path
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                process_id,
                "TOOL_WAVE",
                "CH_WAVE",
                "RCP_WAVE",
                "2026-04-14T00:00:00Z",
                "2026-04-14T00:10:00Z",
                raw_csv_path,
            ),
        )
        con.commit()
    finally:
        con.close()


def _delete_waveform_process(process_id: str) -> None:
    con = _connect(MAIN_DB)
    try:
        con.execute("DELETE FROM ProcessInfo WHERE process_id = ?", (process_id,))
        con.commit()
    finally:
        con.close()


# --- GET /charts/{chart_id}/points ---


def test_get_chart_points_returns_points(
    client: TestClient, seeded_chart_points_context: SeededChartPointsContext
) -> None:
    seeded = seeded_chart_points_context

    res = client.get(f"/charts/{seeded.chart_id}/points")

    assert res.status_code == 200
    body = res.json()
    assert body["ok"] is True
    rows = body["data"]
    assert len(rows) == 3

    # start_ts DESC のため、最新 process が先頭になる。
    assert [row["process_id"] for row in rows] == [
        seeded.process_ids[2],
        seeded.process_ids[1],
        seeded.process_ids[0],
    ]
    # Verify all response fields (process_start_ts, raw_csv_path, feature_value)
    assert rows[0]["feature_value"] == 1.3
    # process_start_ts is converted to UTC ISO 8601 format with milliseconds by to_utc_millis()
    assert isinstance(rows[0]["process_start_ts"], str)
    assert rows[0]["process_start_ts"].endswith("Z")
    # raw_csv_path matches seeded process
    assert rows[0]["raw_csv_path"] == f"data/raw/{seeded.process_ids[2]}.csv"
    # Verify ordering by start_ts
    assert rows[1]["raw_csv_path"] == f"data/raw/{seeded.process_ids[1]}.csv"
    assert rows[2]["raw_csv_path"] == f"data/raw/{seeded.process_ids[0]}.csv"


def test_get_chart_points_returns_empty_for_no_matching_records(
    client: TestClient,
    seeded_chart_only_context: SeededChartOnlyContext,
) -> None:
    seeded = seeded_chart_only_context

    res = client.get(f"/charts/{seeded.chart_id}/points")

    assert res.status_code == 200
    assert res.json() == {"ok": True, "data": []}


def test_get_chart_points_rejects_invalid_chart_id(client: TestClient) -> None:
    # CHART_0 は path pattern を満たすが parse 時に無効として 400 になる。
    res = client.get("/charts/CHART_0/points")

    assert res.status_code == 400
    assert res.json()["detail"] == "Invalid chart_id"


def test_get_chart_points_respects_limit_parameter(
    client: TestClient, seeded_chart_points_context: SeededChartPointsContext
) -> None:
    """Verify that limit query parameter limits returned rows."""
    seeded = seeded_chart_points_context

    # Request with limit=2
    res = client.get(f"/charts/{seeded.chart_id}/points?limit=2")

    assert res.status_code == 200
    rows = res.json()["data"]
    # Should return only 2 rows (most recent first)
    assert len(rows) == 2
    assert rows[0]["process_id"] == seeded.process_ids[2]
    assert rows[1]["process_id"] == seeded.process_ids[1]


def test_get_chart_points_rejects_limit_too_small(client: TestClient) -> None:
    """Verify that limit=0 is rejected with 422 (Pydantic validation)."""
    res = client.get("/charts/CHART_999/points?limit=0")

    assert res.status_code == 422


def test_get_chart_points_rejects_limit_too_large(client: TestClient) -> None:
    """Verify that limit=501 (exceeds max=500) is rejected with 422."""
    res = client.get("/charts/CHART_999/points?limit=501")

    assert res.status_code == 422


def test_get_chart_points_rejects_oversized_chart_pk(client: TestClient) -> None:
    """Verify that oversized chart_pk (exceeds int64 range) is rejected with 400."""
    # CHART_99999999999999999999 exceeds int64 max (9223372036854775807)
    res = client.get("/charts/CHART_99999999999999999999/points")

    assert res.status_code == 400
    assert "Invalid chart_id" in res.json().get("detail", "")


def test_get_chart_points_response_field_types(
    client: TestClient, seeded_chart_points_context: SeededChartPointsContext
) -> None:
    """Verify response field types and structures."""
    seeded = seeded_chart_points_context

    res = client.get(f"/charts/{seeded.chart_id}/points")

    assert res.status_code == 200
    rows = res.json()["data"]
    assert len(rows) > 0

    # Check all required fields and types for first row
    point = rows[0]
    assert isinstance(point["process_id"], str)
    assert len(point["process_id"]) > 0
    assert isinstance(point["feature_value"], (int, float))
    # process_start_ts should be UTC ISO 8601 format string with 'Z' suffix
    assert isinstance(point["process_start_ts"], str)
    assert point["process_start_ts"].endswith("Z")
    # raw_csv_path can be string or null
    assert point["raw_csv_path"] is None or isinstance(point["raw_csv_path"], str)


# --- ChartRepository.find_chart_points ---


def test_find_chart_points_returns_rows(
    seeded_chart_points_context: SeededChartPointsContext,
) -> None:
    seeded = seeded_chart_points_context
    repository = ChartRepository()

    rows = repository.find_chart_points(chart_pk=seeded.chart_pk, limit=10)

    assert len(rows) == 3
    assert rows[0].process_id == seeded.process_ids[2]
    assert rows[0].feature_value == 1.3


def test_find_chart_points_returns_empty_for_no_matching_records(
    seeded_chart_only_context: SeededChartOnlyContext,
) -> None:
    seeded = seeded_chart_only_context
    repository = ChartRepository()

    rows = repository.find_chart_points(chart_pk=seeded.chart_pk, limit=10)

    assert rows == []


def test_find_chart_points_applies_limit(
    seeded_chart_points_context: SeededChartPointsContext,
) -> None:
    seeded = seeded_chart_points_context
    repository = ChartRepository()

    rows = repository.find_chart_points(chart_pk=seeded.chart_pk, limit=1)

    assert len(rows) == 1
    assert rows[0].process_id == seeded.process_ids[2]


# --- GET /processes/{process_id}/waveform-preview ---


def test_get_process_waveform_preview_returns_404_when_process_not_found(
    client: TestClient,
) -> None:
    process_id = f"wave_not_found_{uuid4().hex[:8]}"

    res = client.get(f"/processes/{process_id}/waveform-preview")

    assert res.status_code == 404
    assert res.json()["detail"] == "process not found"


def test_get_process_waveform_preview_returns_empty_points_when_raw_csv_path_null(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    Test handling of NULL raw_csv_path via detailed monkeypatch.

    While current schema enforces NOT NULL on raw_csv_path, this test preserves
    coverage for the NULL-handling code path (_build_waveform_preview), which is
    valuable for schema evolution and explicit behavior verification.

    Uses detailed cursor mock (not just constant (None,)) to simulate real DB
    interaction more faithfully: the cursor must return exactly one column from
    a single-row result when raw_csv_path is NULL.
    """

    class _DetailedCursor:
        """Mock cursor that returns a single NULL value on fetchone()."""

        def __init__(self) -> None:
            self._called = False

        def fetchone(self) -> tuple[None] | None:
            if not self._called:
                self._called = True
                return (None,)  # Simulate NULL raw_csv_path column
            return None

    class _DetailedConnection:
        """Mock connection that returns _DetailedCursor for execute()."""

        def execute(self, *_args: object, **_kwargs: object) -> _DetailedCursor:
            return _DetailedCursor()

        def close(self) -> None:
            pass

    monkeypatch.setattr(db_app, "_connect", lambda _db_path: _DetailedConnection())

    res = client.get("/processes/wave_null_path/waveform-preview")

    assert res.status_code == 200
    body = res.json()
    assert body["ok"] is True
    assert body["data"]["source_path"] is None
    assert body["data"]["points"] == []


def test_get_process_waveform_preview_returns_empty_points_when_file_missing(
    client: TestClient,
    tmp_path: Path,
) -> None:
    process_id = f"wave_missing_{uuid4().hex[:8]}"
    missing_path = tmp_path / "missing_wave.csv"

    try:
        _insert_waveform_process(process_id, missing_path.as_posix())

        res = client.get(f"/processes/{process_id}/waveform-preview")

        assert res.status_code == 200
        body = res.json()
        assert body["ok"] is True
        assert body["data"]["points"] == []
        assert body["data"]["source_path"].endswith("missing_wave.csv")
    finally:
        _delete_waveform_process(process_id)


@pytest.mark.parametrize(
    ("process_id_prefix", "csv_filename", "csv_content", "expected_points"),
    [
        (
            "wave_data_header",
            "wave_with_data_header.csv",
            "# preface\nDATA\ntimestamp,signal\nt1,1.5\nt2,2.5\n",
            [{"x": "t1", "y": 1.5}, {"x": "t2", "y": 2.5}],
        ),
        (
            "wave_no_data_header",
            "wave_without_data_header.csv",
            "timestamp,signal\nt1,3.0\nt2,4.0\n",
            [{"x": "t1", "y": 3.0}, {"x": "t2", "y": 4.0}],
        ),
        (
            "wave_no_numeric",
            "wave_no_numeric.csv",
            "timestamp,label\nt1,high\nt2,low\n",
            [],
        ),
    ],
)
def test_get_process_waveform_preview_csv_parsing_variants(
    client: TestClient,
    tmp_path: Path,
    process_id_prefix: str,
    csv_filename: str,
    csv_content: str,
    expected_points: list[dict[str, float | str]],
) -> None:
    process_id = f"{process_id_prefix}_{uuid4().hex[:8]}"
    csv_path = tmp_path / csv_filename
    csv_path.write_text(csv_content, encoding="utf-8")

    try:
        _insert_waveform_process(process_id, csv_path.as_posix())

        res = client.get(f"/processes/{process_id}/waveform-preview")

        assert res.status_code == 200
        body = res.json()
        assert body["ok"] is True
        assert body["data"]["points"] == expected_points
    finally:
        _delete_waveform_process(process_id)
