from __future__ import annotations

import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any
from uuid import uuid4

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from portfolio_fdc.db_api import app as db_app
from portfolio_fdc.db_api.chart_repository import ChartRepository
from portfolio_fdc.db_api.datetime_util import to_utc_millis
from portfolio_fdc.db_api.db import MAIN_DB, _connect, _init_schema
from tests.utils.test_utils import assert_validation_error_envelope


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


@contextmanager
def _waveform_process_ctx(process_id: str, raw_csv_path: str) -> Iterator[None]:
    _insert_waveform_process(process_id, raw_csv_path)
    try:
        yield
    finally:
        _delete_waveform_process(process_id)


def assert_waveform_preview_response(
    response: Any,
    expected_process_id: str,
    expected_source_path: str | None,
    expected_points: list[dict[str, object]],
) -> None:
    assert response.status_code == 200
    body = response.json()
    assert body["ok"] is True
    assert body["data"]["process_id"] == expected_process_id
    assert body["data"]["source_path"] == expected_source_path
    assert body["data"]["points"] == expected_points


def assert_chart_points_rows(
    rows: list[dict[str, object]],
    expected_process_ids: tuple[str, ...],
    *,
    expected_first_feature_value: float,
) -> None:
    assert len(rows) == len(expected_process_ids)
    assert [row["process_id"] for row in rows] == list(expected_process_ids)

    for row, process_id in zip(rows, expected_process_ids, strict=True):
        assert isinstance(row["process_start_ts"], str)
        assert row["process_start_ts"].endswith("Z")
        assert row["raw_csv_path"] == f"data/raw/{process_id}.csv"

    assert rows[0]["feature_value"] == expected_first_feature_value


# --- GET /charts/{chart_id}/points ---


def test_get_chart_points_returns_points(
    client: TestClient, seeded_chart_points_context: SeededChartPointsContext
) -> None:
    seeded = seeded_chart_points_context

    res = client.get(f"/charts/{seeded.chart_id}/points")

    assert res.status_code == 200
    body = res.json()
    assert body["ok"] is True
    assert_chart_points_rows(
        body["data"],
        expected_process_ids=(
            seeded.process_ids[2],
            seeded.process_ids[1],
            seeded.process_ids[0],
        ),
        expected_first_feature_value=1.3,
    )


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
    # Should return only 2 rows (most recent first)
    assert_chart_points_rows(
        res.json()["data"],
        expected_process_ids=(
            seeded.process_ids[2],
            seeded.process_ids[1],
        ),
        expected_first_feature_value=1.3,
    )


def test_get_chart_points_rejects_limit_too_small(client: TestClient) -> None:
    """Verify that limit=0 is rejected with 422 (Pydantic validation)."""
    res = client.get("/charts/CHART_999/points?limit=0")

    assert res.status_code == 422
    assert_validation_error_envelope(res.json(), expected_loc_fragment="limit")


def test_get_chart_points_rejects_limit_too_large(client: TestClient) -> None:
    """Verify that limit=501 (exceeds max=500) is rejected with 422."""
    res = client.get("/charts/CHART_999/points?limit=501")

    assert res.status_code == 422
    assert_validation_error_envelope(res.json(), expected_loc_fragment="limit")


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


def test_get_chart_points_transient_db_error_returns_503(
    client: TestClient,
    seeded_chart_points_context: SeededChartPointsContext,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seeded = seeded_chart_points_context

    def fail_find_chart_points(*args: object, **kwargs: object) -> list[object]:
        _ = args, kwargs
        raise sqlite3.OperationalError("database is locked")

    monkeypatch.setattr(db_app._chart_repository, "find_chart_points", fail_find_chart_points)

    res = client.get(f"/charts/{seeded.chart_id}/points")

    assert res.status_code == 503
    assert res.json()["detail"] == "Database temporarily unavailable"


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
    assert rows[0].process_start_ts == to_utc_millis("2026-04-14T00:02:00Z")
    assert rows[0].raw_csv_path == f"data/raw/{seeded.process_ids[2]}.csv"


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
    assert rows[0].process_start_ts == to_utc_millis("2026-04-14T00:02:00Z")
    assert rows[0].raw_csv_path == f"data/raw/{seeded.process_ids[2]}.csv"


def test_find_chart_points_tiebreaker_with_same_start_ts() -> None:
    suffix = uuid4().hex[:10]
    same_start_ts = "2026-04-14T00:05:00Z"
    process_ids: tuple[str, str] = (
        f"p_points_tie_{suffix}_a",
        f"p_points_tie_{suffix}_b",
    )
    chart_set_id: int | None = None

    _init_schema(MAIN_DB)
    con = _connect(MAIN_DB)
    try:
        chart_set_id = _insert_chart_set(con, suffix)
        chart_pk = _insert_chart(con, chart_set_id, suffix)

        _insert_process_and_parameter(
            con,
            process_ids[0],
            suffix,
            start_ts=same_start_ts,
            feature_value=1.1,
        )
        _insert_process_and_parameter(
            con,
            process_ids[1],
            suffix,
            start_ts=same_start_ts,
            feature_value=1.2,
        )
        con.commit()

        repository = ChartRepository()
        rows = repository.find_chart_points(chart_pk=chart_pk, limit=2)

        assert len(rows) == 2
        # Same start_ts should be tie-broken by Parameters.id DESC (second insert first)
        assert rows[0].process_id == process_ids[1]
        assert rows[1].process_id == process_ids[0]
        assert rows[0].process_start_ts == to_utc_millis(same_start_ts)
        assert rows[1].process_start_ts == to_utc_millis(same_start_ts)
        assert rows[0].raw_csv_path == f"data/raw/{process_ids[1]}.csv"
        assert rows[1].raw_csv_path == f"data/raw/{process_ids[0]}.csv"
    finally:
        con.close()
        if chart_set_id is not None:
            _cleanup_chart_set(chart_set_id, process_ids)


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

        def __init__(self) -> None:
            self._cursor = _DetailedCursor()
            self.last_sql: str | None = None
            self.last_params: tuple[object, ...] | None = None

        def execute(self, sql: str, params: tuple[object, ...]) -> _DetailedCursor:
            self.last_sql = sql
            self.last_params = params
            return self._cursor

        def close(self) -> None:
            pass

    connection = _DetailedConnection()
    monkeypatch.setattr(db_app, "_connect", lambda _db_path: connection)

    res = client.get("/processes/wave_null_path/waveform-preview")

    assert res.status_code == 200
    body = res.json()
    assert body["ok"] is True
    assert body["data"]["process_id"] == "wave_null_path"
    assert body["data"]["source_path"] is None
    assert body["data"]["points"] == []
    assert connection.last_sql == "SELECT raw_csv_path FROM ProcessInfo WHERE process_id = ?"
    assert connection.last_params == ("wave_null_path",)


def test_get_process_waveform_preview_returns_empty_points_when_file_missing(
    client: TestClient,
    tmp_path: Path,
) -> None:
    process_id = f"wave_missing_{uuid4().hex[:8]}"
    missing_path = tmp_path / "missing_wave.csv"

    with _waveform_process_ctx(process_id, missing_path.as_posix()):
        res = client.get(f"/processes/{process_id}/waveform-preview")

        assert res.status_code == 200
        body = res.json()
        assert body["ok"] is True
        assert body["data"]["process_id"] == process_id
        assert body["data"]["points"] == []
        assert body["data"]["source_path"] == missing_path.as_posix()


def test_get_process_waveform_preview_applies_limit_to_tail(
    client: TestClient,
    tmp_path: Path,
) -> None:
    process_id = f"wave_limit_{uuid4().hex[:8]}"
    csv_path = tmp_path / "wave_limit.csv"
    csv_path.write_text(
        "timestamp,signal\n"
        "t1,1.0\n"
        "t2,2.0\n"
        "t3,3.0\n"
        "t4,4.0\n"
        "t5,5.0\n"
        "t6,6.0\n"
        "t7,7.0\n"
        "t8,8.0\n"
        "t9,9.0\n"
        "t10,10.0\n"
        "t11,11.0\n"
        "t12,12.0\n",
        encoding="utf-8",
    )

    with _waveform_process_ctx(process_id, csv_path.as_posix()):
        res = client.get(f"/processes/{process_id}/waveform-preview?limit=10")

        assert_waveform_preview_response(
            res,
            expected_process_id=process_id,
            expected_source_path=csv_path.as_posix(),
            expected_points=[
                {"x": "t3", "y": 3.0},
                {"x": "t4", "y": 4.0},
                {"x": "t5", "y": 5.0},
                {"x": "t6", "y": 6.0},
                {"x": "t7", "y": 7.0},
                {"x": "t8", "y": 8.0},
                {"x": "t9", "y": 9.0},
                {"x": "t10", "y": 10.0},
                {"x": "t11", "y": 11.0},
                {"x": "t12", "y": 12.0},
            ],
        )


@pytest.mark.parametrize("limit", [9, 2001])
def test_get_process_waveform_preview_rejects_invalid_limit(
    client: TestClient,
    limit: int,
) -> None:
    res = client.get(f"/processes/wave_limit_invalid/waveform-preview?limit={limit}")

    assert res.status_code == 422
    assert_validation_error_envelope(
        res.json(),
        expected_loc_fragment="limit",
    )


def test_get_process_waveform_preview_rejects_invalid_process_id_pattern(
    client: TestClient,
) -> None:
    res = client.get("/processes/wave%20invalid/waveform-preview")

    assert res.status_code == 422
    assert_validation_error_envelope(
        res.json(),
        expected_loc_fragment="process_id",
    )


def test_get_process_waveform_preview_returns_null_for_nan_y(
    client: TestClient,
    tmp_path: Path,
) -> None:
    process_id = f"wave_nan_{uuid4().hex[:8]}"
    csv_path = tmp_path / "wave_nan.csv"
    csv_path.write_text("timestamp,signal\nt1,1.5\nt2,\n", encoding="utf-8")

    with _waveform_process_ctx(process_id, csv_path.as_posix()):
        res = client.get(f"/processes/{process_id}/waveform-preview")

        assert_waveform_preview_response(
            res,
            expected_process_id=process_id,
            expected_source_path=csv_path.as_posix(),
            expected_points=[
                {"x": "t1", "y": 1.5},
                {"x": "t2", "y": None},
            ],
        )


def test_get_process_waveform_preview_returns_empty_points_on_parser_error(
    client: TestClient,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    process_id = f"wave_parser_{uuid4().hex[:8]}"
    csv_path = tmp_path / "wave_parser.csv"
    csv_path.write_text("timestamp,signal\nt1,1.0\n", encoding="utf-8")

    def raise_parser_error(*args: object, **kwargs: object) -> pd.DataFrame:
        _ = args, kwargs
        raise pd.errors.ParserError("synthetic parse failure")

    monkeypatch.setattr(pd, "read_csv", raise_parser_error)

    with _waveform_process_ctx(process_id, csv_path.as_posix()):
        res = client.get(f"/processes/{process_id}/waveform-preview")

        assert_waveform_preview_response(
            res,
            expected_process_id=process_id,
            expected_source_path=csv_path.as_posix(),
            expected_points=[],
        )


def test_get_process_waveform_preview_returns_empty_points_for_header_only_csv(
    client: TestClient,
    tmp_path: Path,
) -> None:
    process_id = f"wave_empty_{uuid4().hex[:8]}"
    csv_path = tmp_path / "wave_header_only.csv"
    csv_path.write_text("timestamp,signal\n", encoding="utf-8")

    with _waveform_process_ctx(process_id, csv_path.as_posix()):
        res = client.get(f"/processes/{process_id}/waveform-preview")

        assert_waveform_preview_response(
            res,
            expected_process_id=process_id,
            expected_source_path=csv_path.as_posix(),
            expected_points=[],
        )


def test_get_process_waveform_preview_resolves_relative_source_path(
    client: TestClient,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    process_id = f"wave_relative_{uuid4().hex[:8]}"
    relative_path = Path("data/raw") / f"{process_id}.csv"
    absolute_path = tmp_path / relative_path
    absolute_path.parent.mkdir(parents=True, exist_ok=True)
    absolute_path.write_text("timestamp,signal\nt1,7.0\n", encoding="utf-8")

    monkeypatch.setattr(
        db_app.pathlib.Path,
        "cwd",
        classmethod(lambda _cls: tmp_path),
    )

    with _waveform_process_ctx(process_id, relative_path.as_posix()):
        res = client.get(f"/processes/{process_id}/waveform-preview")

        assert_waveform_preview_response(
            res,
            expected_process_id=process_id,
            expected_source_path=absolute_path.as_posix(),
            expected_points=[{"x": "t1", "y": 7.0}],
        )


def test_get_process_waveform_preview_uses_first_numeric_column(
    client: TestClient,
    tmp_path: Path,
) -> None:
    process_id = f"wave_multi_numeric_{uuid4().hex[:8]}"
    csv_path = tmp_path / "wave_multi_numeric.csv"
    csv_path.write_text(
        "timestamp,primary_signal,secondary_signal\nt1,1.0,10.0\nt2,2.0,20.0\n",
        encoding="utf-8",
    )

    with _waveform_process_ctx(process_id, csv_path.as_posix()):
        res = client.get(f"/processes/{process_id}/waveform-preview")

        assert_waveform_preview_response(
            res,
            expected_process_id=process_id,
            expected_source_path=csv_path.as_posix(),
            expected_points=[
                {"x": "t1", "y": 1.0},
                {"x": "t2", "y": 2.0},
            ],
        )


def test_get_process_waveform_preview_returns_503_on_transient_db_error(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail_build(*args: object, **kwargs: object) -> dict[str, object]:
        _ = args, kwargs
        raise sqlite3.OperationalError("database is locked")

    monkeypatch.setattr(db_app, "_build_waveform_preview", fail_build)

    res = client.get("/processes/wave_db_error/waveform-preview")

    assert res.status_code == 503
    assert res.json()["detail"] == "Database temporarily unavailable"


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
    expected_points: list[dict[str, object]],
) -> None:
    process_id = f"{process_id_prefix}_{uuid4().hex[:8]}"
    csv_path = tmp_path / csv_filename
    csv_path.write_text(csv_content, encoding="utf-8")

    with _waveform_process_ctx(process_id, csv_path.as_posix()):
        res = client.get(f"/processes/{process_id}/waveform-preview")

        assert_waveform_preview_response(
            res,
            expected_process_id=process_id,
            expected_source_path=csv_path.as_posix(),
            expected_points=expected_points,
        )
