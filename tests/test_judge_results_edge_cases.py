"""Edge case tests for judge results repository and API contract."""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Iterator
from dataclasses import dataclass
from uuid import uuid4

import pytest
from fastapi.testclient import TestClient

from portfolio_fdc.db_api.db import MAIN_DB, _init_schema
from portfolio_fdc.db_api.judge_repository import (
    _extract_chart_id,
    _to_float_or_none,
    _to_int_or_none,
)


@dataclass(frozen=True)
class SeededEdgeCaseContext:
    recipe_id: str
    process_id_float_chart: str
    process_id_leading_zero_chart: str
    process_id_nan_feature: str
    process_id_inf_feature: str


@pytest.fixture
def seeded_edge_case_context() -> Iterator[SeededEdgeCaseContext]:
    """Edge case test data: float chart_id, NaN/Infinity feature values."""
    _init_schema(MAIN_DB)
    suffix = uuid4().hex[:10]
    recipe_id = f"RECIPE_EDGE_{suffix}"
    process_id_float_chart = f"P_FLOAT_CHART_{suffix}"
    process_id_leading_zero_chart = f"P_LEADING_ZERO_{suffix}"
    process_id_nan_feature = f"P_NAN_FEAT_{suffix}"
    process_id_inf_feature = f"P_INF_FEAT_{suffix}"

    con = sqlite3.connect(MAIN_DB.as_posix())
    try:
        # Insert processes
        for process_id in [
            process_id_float_chart,
            process_id_leading_zero_chart,
            process_id_nan_feature,
            process_id_inf_feature,
        ]:
            con.execute(
                """
                INSERT INTO ProcessInfo(
                    process_id, tool_id, chamber_id, recipe_id,
                    start_ts, end_ts, raw_csv_path
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    process_id,
                    f"TOOL_{suffix}",
                    "CH1",
                    recipe_id,
                    "2026-04-17T00:00:00+00:00",
                    "2026-04-17T00:05:00+00:00",
                    f"path_{process_id}",
                ),
            )

        # Case 1: float chart_id (100.9) -> should normalize to CHART_100
        con.execute(
            """
            INSERT INTO JudgementResults(
                process_id, tool_id, chamber_id, recipe_id, status, judged_at, message_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                process_id_float_chart,
                f"TOOL_{suffix}",
                "CH1",
                recipe_id,
                "OK",
                "2026-04-17T00:01:00+00:00",
                json.dumps(
                    {
                        "chart_id": 100.9,  # Float chart_id
                        "step_no": 1,
                        "feature_type": "mean",
                        "feature_value": 1.5,
                    }
                ),
            ),
        )

        # Case 1b: leading-zero numeric-text chart_id ("001") -> CHART_1
        con.execute(
            """
            INSERT INTO JudgementResults(
                process_id, tool_id, chamber_id, recipe_id, status, judged_at, message_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                process_id_leading_zero_chart,
                f"TOOL_{suffix}",
                "CH1",
                recipe_id,
                "OK",
                "2026-04-17T00:01:30+00:00",
                json.dumps(
                    {
                        "chart_id": "001",
                        "step_no": 1,
                        "feature_type": "mean",
                        "feature_value": 1.0,
                    }
                ),
            ),
        )

        # Case 2: NaN feature_value
        con.execute(
            """
            INSERT INTO JudgementResults(
                process_id, tool_id, chamber_id, recipe_id, status, judged_at, message_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                process_id_nan_feature,
                f"TOOL_{suffix}",
                "CH1",
                recipe_id,
                "OK",
                "2026-04-17T00:02:00+00:00",
                json.dumps(
                    {
                        "chart_id": 200,
                        "step_no": 2,
                        "feature_type": "std",
                        "feature_value": float("nan"),  # NaN
                    }
                ),
            ),
        )

        # Case 3: Infinity feature_value
        con.execute(
            """
            INSERT INTO JudgementResults(
                process_id, tool_id, chamber_id, recipe_id, status, judged_at, message_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                process_id_inf_feature,
                f"TOOL_{suffix}",
                "CH1",
                recipe_id,
                "OK",
                "2026-04-17T00:03:00+00:00",
                json.dumps(
                    {
                        "chart_id": 300,
                        "step_no": 3,
                        "feature_type": "max",
                        "feature_value": float("inf"),  # Infinity
                    }
                ),
            ),
        )

        con.commit()

        yield SeededEdgeCaseContext(
            recipe_id=recipe_id,
            process_id_float_chart=process_id_float_chart,
            process_id_leading_zero_chart=process_id_leading_zero_chart,
            process_id_nan_feature=process_id_nan_feature,
            process_id_inf_feature=process_id_inf_feature,
        )
    finally:
        cleanup = sqlite3.connect(MAIN_DB.as_posix())
        try:
            cleanup.execute(
                "DELETE FROM JudgementResults WHERE recipe_id = ?",
                (recipe_id,),
            )
            cleanup.execute(
                "DELETE FROM ProcessInfo WHERE recipe_id = ?",
                (recipe_id,),
            )
            cleanup.commit()
        finally:
            cleanup.close()
        con.close()


def test_extract_chart_id_float_truncates_to_int() -> None:
    """_extract_chart_id should truncate float to int (100.9 -> CHART_100)."""
    payload = {"chart_id": 100.9}
    result = _extract_chart_id(payload, None)
    assert result == "CHART_100", f"Expected CHART_100 but got {result}"


def test_extract_chart_id_rejects_boolean() -> None:
    """_extract_chart_id should return None for boolean values."""
    payload = {"chart_id": True}
    result = _extract_chart_id(payload, None)
    assert result is None, f"Expected None for boolean but got {result}"


def test_extract_chart_id_rejects_nan() -> None:
    """_extract_chart_id should return None for NaN values."""
    payload = {"chart_id": float("nan")}
    result = _extract_chart_id(payload, None)
    assert result is None, f"Expected None for NaN but got {result}"


def test_extract_chart_id_rejects_positive_infinity() -> None:
    """_extract_chart_id should return None for positive infinity."""
    payload = {"chart_id": float("inf")}
    result = _extract_chart_id(payload, None)
    assert result is None, f"Expected None for +inf but got {result}"


def test_extract_chart_id_rejects_negative_infinity() -> None:
    """_extract_chart_id should return None for negative infinity."""
    payload = {"chart_id": float("-inf")}
    result = _extract_chart_id(payload, None)
    assert result is None, f"Expected None for -inf but got {result}"


def test_extract_chart_id_leading_zero_string_normalizes_to_int_form() -> None:
    """_extract_chart_id should normalize digit-only strings to integer form."""
    payload = {"chart_id": "001"}
    result = _extract_chart_id(payload, None)
    assert result == "CHART_1", f"Expected CHART_1 but got {result}"


def test_to_float_or_none_rejects_nan() -> None:
    """_to_float_or_none should return None for NaN."""
    result = _to_float_or_none(float("nan"))
    assert result is None, f"Expected None for NaN but got {result}"


def test_to_float_or_none_rejects_positive_infinity() -> None:
    """_to_float_or_none should return None for positive infinity."""
    result = _to_float_or_none(float("inf"))
    assert result is None, f"Expected None for +inf but got {result}"


def test_to_float_or_none_rejects_negative_infinity() -> None:
    """_to_float_or_none should return None for negative infinity."""
    result = _to_float_or_none(float("-inf"))
    assert result is None, f"Expected None for -inf but got {result}"


def test_to_float_or_none_accepts_finite_values() -> None:
    """_to_float_or_none should accept finite float values."""
    assert _to_float_or_none(1.5) == 1.5
    assert _to_float_or_none(0.0) == 0.0
    assert _to_float_or_none(-3.14) == -3.14


def test_to_int_or_none_rejects_boolean_values() -> None:
    """_to_int_or_none should return None for bool values."""
    assert _to_int_or_none(True) is None
    assert _to_int_or_none(False) is None


def test_to_int_or_none_rejects_non_finite_floats() -> None:
    """_to_int_or_none should reject NaN and infinities."""
    assert _to_int_or_none(float("inf")) is None
    assert _to_int_or_none(float("-inf")) is None
    assert _to_int_or_none(float("nan")) is None


def test_to_int_or_none_accepts_integral_floats_only() -> None:
    """_to_int_or_none should accept only finite integral floats."""
    assert _to_int_or_none(3.0) == 3
    assert _to_int_or_none(3.14) is None


def test_api_float_chart_id_normalized(
    client: TestClient,
    seeded_edge_case_context: SeededEdgeCaseContext,
) -> None:
    """GET /judge/results should normalize float chart_id 100.9 to CHART_100."""
    seeded = seeded_edge_case_context

    res = client.get(
        "/judge/results",
        params={
            "process_id": seeded.process_id_float_chart,
            "limit": 10,
        },
    )

    assert res.status_code == 200
    rows = res.json()["data"]
    assert len(rows) == 1
    assert rows[0]["chart_id"] == "CHART_100", f"Expected CHART_100 but got {rows[0]['chart_id']}"


def test_api_leading_zero_chart_id_normalized_and_filterable(
    client: TestClient,
    seeded_edge_case_context: SeededEdgeCaseContext,
) -> None:
    """GET /judge/results should normalize "001" to CHART_1 and support SQL filtering."""
    seeded = seeded_edge_case_context

    by_process = client.get(
        "/judge/results",
        params={
            "process_id": seeded.process_id_leading_zero_chart,
            "limit": 10,
        },
    )
    assert by_process.status_code == 200
    process_rows = by_process.json()["data"]
    assert len(process_rows) == 1
    assert process_rows[0]["chart_id"] == "CHART_1"

    by_chart_filter = client.get(
        "/judge/results",
        params={
            "recipe_id": seeded.recipe_id,
            "chart_id": "CHART_1",
            "limit": 100,
        },
    )
    assert by_chart_filter.status_code == 200
    filtered_rows = by_chart_filter.json()["data"]
    assert any(row["process_id"] == seeded.process_id_leading_zero_chart for row in filtered_rows)


def test_api_nan_feature_value_excluded(
    client: TestClient,
    seeded_edge_case_context: SeededEdgeCaseContext,
) -> None:
    """GET /judge/results should return None for NaN feature_value."""
    seeded = seeded_edge_case_context

    res = client.get(
        "/judge/results",
        params={
            "process_id": seeded.process_id_nan_feature,
            "limit": 10,
        },
    )

    assert res.status_code == 200
    rows = res.json()["data"]
    assert len(rows) == 1
    assert rows[0]["feature_value"] is None, (
        f"Expected None for NaN but got {rows[0]['feature_value']}"
    )


def test_api_infinite_feature_value_excluded(
    client: TestClient,
    seeded_edge_case_context: SeededEdgeCaseContext,
) -> None:
    """GET /judge/results should return None for Infinity feature_value."""
    seeded = seeded_edge_case_context

    res = client.get(
        "/judge/results",
        params={
            "process_id": seeded.process_id_inf_feature,
            "limit": 10,
        },
    )

    assert res.status_code == 200
    rows = res.json()["data"]
    assert len(rows) == 1
    assert rows[0]["feature_value"] is None, (
        f"Expected None for Infinity but got {rows[0]['feature_value']}"
    )
