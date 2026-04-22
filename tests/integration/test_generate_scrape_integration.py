from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pandas as pd
import pytest

from portfolio_fdc.main import scrape
from portfolio_fdc.tools.sim_data_generator import generate_logger_csv as glc

pytestmark = pytest.mark.integration


def _tool_cfg(tool_id: str) -> dict:
    channels_by_tool = {
        "TOOL_A": {
            "value01": "dc_bias",
            "value02": "cl2_flow",
            "value03": "apc_pressure",
        },
        "TOOL_B": {
            "value01": "apc_pressure",
            "value02": "cl2_flow",
            "value03": "dc_bias",
        },
    }
    return {
        "chamber_id": "CH_INT",
        "channels": channels_by_tool[tool_id],
    }


@pytest.mark.parametrize("tool_id", ["TOOL_A", "TOOL_B"])
def test_generate_then_scrape_full_window(tmp_path: Path, monkeypatch, tool_id: str) -> None:
    raw = tmp_path / "logger_raw.csv"
    monkeypatch.setattr(scrape, "STATE_DIR", tmp_path / "state")

    glc.write_logger_csv(
        path=raw,
        start_ts=datetime.fromisoformat("2026-02-19T00:00:00"),
        seconds=20,
        scenario="normal",
        seed=7,
        append=False,
        tool_id=tool_id,
    )

    out = scrape.scrape_logger_csv(
        raw_csv_path=raw,
        tool_id=tool_id,
        tool_cfg=_tool_cfg(tool_id),
        now=datetime.fromisoformat("2026-02-19T00:00:20"),
        lookback_minutes=1,
    )

    assert len(out) == 20
    assert {"dc_bias", "cl2_flow", "apc_pressure", "tool_id", "chamber_id"}.issubset(out.columns)
    assert set(out["tool_id"].unique()) == {tool_id}
    assert set(out["chamber_id"].unique()) == {"CH_INT"}


@pytest.mark.parametrize("tool_id", ["TOOL_A", "TOOL_B"])
def test_generate_append_then_scrape_incremental_by_state(
    tmp_path: Path,
    monkeypatch,
    tool_id: str,
) -> None:
    raw = tmp_path / "logger_raw.csv"
    monkeypatch.setattr(scrape, "STATE_DIR", tmp_path / "state")

    glc.write_logger_csv(
        path=raw,
        start_ts=datetime.fromisoformat("2026-02-19T00:00:00"),
        seconds=10,
        scenario="normal",
        seed=10,
        append=False,
        tool_id=tool_id,
    )

    first = scrape.scrape_logger_csv(
        raw_csv_path=raw,
        tool_id=tool_id,
        tool_cfg=_tool_cfg(tool_id),
        now=datetime.fromisoformat("2026-02-19T00:00:09"),
        lookback_minutes=1,
    )
    assert len(first) == 10

    glc.write_logger_csv(
        path=raw,
        start_ts=datetime.fromisoformat("2026-02-19T00:00:10"),
        seconds=5,
        scenario="normal",
        seed=11,
        append=True,
        tool_id=tool_id,
    )

    second = scrape.scrape_logger_csv(
        raw_csv_path=raw,
        tool_id=tool_id,
        tool_cfg=_tool_cfg(tool_id),
        now=datetime.fromisoformat("2026-02-19T00:00:14"),
        lookback_minutes=1,
    )

    assert len(second) == 5
    ts = second["timestamp"].dt.strftime("%Y-%m-%dT%H:%M:%S").tolist()
    assert ts[0] == "2026-02-19T00:00:10"
    assert ts[-1] == "2026-02-19T00:00:14"


@pytest.mark.parametrize("tool_id", ["TOOL_A", "TOOL_B"])
@pytest.mark.parametrize("scenario", ["warn", "crit"])
def test_generate_anomaly_then_scrape_keeps_time_window_and_state(
    tmp_path: Path,
    monkeypatch,
    tool_id: str,
    scenario: str,
) -> None:
    raw = tmp_path / "logger_raw.csv"
    monkeypatch.setattr(scrape, "STATE_DIR", tmp_path / "state")

    glc.write_logger_csv(
        path=raw,
        start_ts=datetime.fromisoformat("2026-02-19T01:00:00"),
        seconds=12,
        scenario=scenario,
        seed=21,
        append=False,
        tool_id=tool_id,
    )

    out = scrape.scrape_logger_csv(
        raw_csv_path=raw,
        tool_id=tool_id,
        tool_cfg=_tool_cfg(tool_id),
        now=datetime.fromisoformat("2026-02-19T01:00:11"),
        lookback_minutes=1,
    )

    assert len(out) == 12
    ts = out["timestamp"].dt.strftime("%Y-%m-%dT%H:%M:%S").tolist()
    assert ts[0] == "2026-02-19T01:00:00"
    assert ts[-1] == "2026-02-19T01:00:11"

    saved = scrape.load_last_ts(tool_id)
    assert saved == datetime.fromisoformat("2026-02-19T01:00:11")


@pytest.mark.parametrize("tool_id", ["TOOL_A", "TOOL_B"])
@pytest.mark.slow
def test_large_file_generated_by_glc_is_read_via_tail_path(
    tmp_path: Path,
    monkeypatch,
    tool_id: str,
) -> None:
    raw = tmp_path / "logger_raw_large.csv"
    monkeypatch.setattr(scrape, "STATE_DIR", tmp_path / "state")

    start = datetime.fromisoformat("2026-02-19T02:00:00")
    seconds = 50_000
    glc.write_logger_csv(
        path=raw,
        start_ts=start,
        seconds=seconds,
        scenario="normal",
        seed=99,
        append=False,
        tool_id=tool_id,
    )

    size_mb = raw.stat().st_size / (1024 * 1024)
    assert size_mb > 1.0

    called = {"tail": False}
    original_tail_reader = scrape.read_recent_tail_as_df

    def wrapped_tail_reader(path: Path, columns: list[str], n_lines: int):
        called["tail"] = True
        return original_tail_reader(path, columns, n_lines)

    def fail_read_from_top(path: Path, data_line_no: int):
        raise AssertionError("read_from_top should not be called for large-file tail path test")

    monkeypatch.setattr(scrape, "read_recent_tail_as_df", wrapped_tail_reader)
    monkeypatch.setattr(scrape, "read_from_top", fail_read_from_top)

    now = start + pd.Timedelta(seconds=seconds - 1)
    out = scrape.scrape_logger_csv(
        raw_csv_path=raw,
        tool_id=tool_id,
        tool_cfg=_tool_cfg(tool_id),
        now=now,
        lookback_minutes=1,
        huge_threshold_mb=1,
    )

    assert called["tail"] is True
    assert len(out) == 60
    ts = out["timestamp"].dt.strftime("%Y-%m-%dT%H:%M:%S").tolist()
    assert ts[0] == (now - pd.Timedelta(seconds=59)).strftime("%Y-%m-%dT%H:%M:%S")
    assert ts[-1] == now.strftime("%Y-%m-%dT%H:%M:%S")
