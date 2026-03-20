from __future__ import annotations

from datetime import datetime

import pytest
from pydantic import ValidationError

from portfolio_fdc.db_api.schemas import (
    ProcessInfoIn,
    StepWindowIn,
    validate_timestamp_range,
)


def _process_payload(**overrides: str) -> dict[str, str]:
    payload = {
        "process_id": "p1",
        "tool_id": "TOOL_A",
        "chamber_id": "CH1",
        "recipe_id": "UNKNOWN",
        "start_ts": "2026-03-20T00:00:00+00:00",
        "end_ts": "2026-03-20T00:00:01+00:00",
        "raw_csv_path": "data/detail/p1.csv",
    }
    payload.update(overrides)
    return payload


def _step_payload(**overrides: str | int) -> dict[str, str | int]:
    payload: dict[str, str | int] = {
        "process_id": "p1",
        "step_no": 1,
        "start_ts": "2026-03-20T00:00:00+00:00",
        "end_ts": "2026-03-20T00:00:01+00:00",
        "source_channel": "dc_bias",
    }
    payload.update(overrides)
    return payload


def test_validate_timestamp_range_rejects_inverted_range() -> None:
    start = datetime.fromisoformat("2026-03-20T00:00:01+00:00")
    end = datetime.fromisoformat("2026-03-20T00:00:00+00:00")

    with pytest.raises(ValueError, match="end_ts must be greater than or equal to start_ts"):
        validate_timestamp_range(start, end)


def test_validate_timestamp_range_allows_equal_boundary() -> None:
    ts = datetime.fromisoformat("2026-03-20T00:00:00+00:00")

    validate_timestamp_range(ts, ts)


@pytest.mark.parametrize("bad_field", ["start_ts", "end_ts"])
def test_process_info_rejects_invalid_iso8601_string(bad_field: str) -> None:
    payload = _process_payload(**{bad_field: "not-a-date"})

    with pytest.raises(ValidationError):
        ProcessInfoIn(**payload)


@pytest.mark.parametrize("bad_field", ["start_ts", "end_ts"])
def test_step_window_rejects_invalid_iso8601_string(bad_field: str) -> None:
    payload = _step_payload(**{bad_field: "not-a-date"})

    with pytest.raises(ValidationError):
        StepWindowIn(**payload)


def test_process_info_rejects_mixed_naive_and_aware() -> None:
    with pytest.raises(ValidationError, match="same timezone format"):
        ProcessInfoIn(
            **_process_payload(
                start_ts="2026-03-20T00:00:00",
                end_ts="2026-03-20T00:00:01+00:00",
            )
        )


def test_process_info_rejects_aware_then_naive() -> None:
    with pytest.raises(ValidationError, match="same timezone format"):
        ProcessInfoIn(
            **_process_payload(
                start_ts="2026-03-20T00:00:00+00:00",
                end_ts="2026-03-20T00:00:01",
            )
        )


def test_step_window_rejects_mixed_naive_and_aware() -> None:
    with pytest.raises(ValidationError, match="same timezone format"):
        StepWindowIn(
            **_step_payload(
                start_ts="2026-03-20T00:00:00+09:00",
                end_ts="2026-03-20T00:00:01",
            )
        )


def test_step_window_rejects_naive_then_aware() -> None:
    with pytest.raises(ValidationError, match="same timezone format"):
        StepWindowIn(
            **_step_payload(
                start_ts="2026-03-20T00:00:00",
                end_ts="2026-03-20T00:00:01+09:00",
            )
        )


def test_process_info_accepts_aware_z_and_offset_mix() -> None:
    model = ProcessInfoIn(
        **_process_payload(
            start_ts="2026-03-20T00:00:00Z",
            end_ts="2026-03-20T09:00:01+09:00",
        )
    )

    assert model.start_ts.tzinfo is not None
    assert model.end_ts.tzinfo is not None


def test_step_window_accepts_different_aware_offsets() -> None:
    model = StepWindowIn(
        **_step_payload(
            start_ts="2026-03-20T00:00:00+00:00",
            end_ts="2026-03-20T09:00:01+09:00",
        )
    )

    assert model.start_ts.tzinfo is not None
    assert model.end_ts.tzinfo is not None