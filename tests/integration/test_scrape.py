from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pandas as pd

from portfolio_fdc.main import scrape


def _write_logger_raw(path: Path, rows: list[str]) -> None:
    lines = [
        "LOGGER_CONFIG,Sampling=1s,Channels=3",
        "RANGE,value01,0,5",
        "RANGE,value02,0,50",
        "RANGE,value03,0,100",
        "DATA",
        "timestamp,value01,value02,value03",
        *rows,
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def test_find_data_header_line_no_success(tmp_path: Path) -> None:
    raw = tmp_path / "logger_raw.csv"
    _write_logger_raw(raw, [])

    assert scrape.find_data_header_line_no(raw) == 4


def test_find_data_header_line_no_not_found_raises(tmp_path: Path) -> None:
    raw = tmp_path / "no_data.csv"
    raw.write_text("a\nb\nc\n", encoding="utf-8")

    import pytest

    with pytest.raises(RuntimeError, match="DATA marker not found"):
        scrape.find_data_header_line_no(raw, max_scan_lines=5)


def test_filter_by_time_window_boundaries() -> None:
    start = datetime.fromisoformat("2026-02-19T10:00:00")
    end = datetime.fromisoformat("2026-02-19T10:30:00")
    df = pd.DataFrame(
        {
            "timestamp": [
                "2026-02-19T10:00:00",
                "2026-02-19T10:00:01",
                "2026-02-19T10:30:00",
                "2026-02-19T10:30:01",
                "bad",
            ],
            "value01": [1, 2, 3, 4, 5],
        }
    )

    out = scrape.filter_by_time_window(df, start, end)

    assert out["timestamp"].dt.strftime("%Y-%m-%dT%H:%M:%S").tolist() == [
        "2026-02-19T10:00:01",
        "2026-02-19T10:30:00",
    ]


def test_filter_by_time_window_missing_timestamp_raises() -> None:
    import pytest

    start = datetime.fromisoformat("2026-02-19T10:00:00")
    end = datetime.fromisoformat("2026-02-19T10:30:00")
    df = pd.DataFrame({"value01": [1, 2, 3]})

    with pytest.raises(ValueError, match="timestamp column is required"):
        scrape.filter_by_time_window(df, start, end)


def test_apply_tool_mapping_adds_ids_and_renames() -> None:
    df = pd.DataFrame(
        {
            "timestamp": ["2026-02-19T10:00:01"],
            "value01": [1.0],
            "value02": [2.0],
        }
    )

    out = scrape.apply_tool_mapping(
        df,
        tool_id="TOOL_A",
        chamber_id="CH9",
        channel_map={"value01": "DC Bias", "value02": "RF Power"},
    )

    assert out["tool_id"].iloc[0] == "TOOL_A"
    assert out["chamber_id"].iloc[0] == "CH9"
    assert "DC Bias" in out.columns
    assert "RF Power" in out.columns


def test_resolve_channel_map_falls_back_to_sensor_map(tmp_path: Path, monkeypatch) -> None:
    sensor_map_csv = tmp_path / "sensor_map.csv"
    sensor_map_csv.write_text(
        "\n".join(
            [
                "tool_id,sensor,parameter",
                "TOOL_X,value01,dc_bias",
                "TOOL_X,value02,cl2_flow",
                "TOOL_X,value03,apc_pressure",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(scrape, "SENSOR_MAP_CSV_PATH", sensor_map_csv)

    out = scrape.resolve_channel_map("TOOL_X", tool_cfg={})

    assert out == {
        "value01": "dc_bias",
        "value02": "cl2_flow",
        "value03": "apc_pressure",
    }


def test_resolve_channel_map_prefers_tool_cfg_channels(tmp_path: Path, monkeypatch) -> None:
    sensor_map_csv = tmp_path / "sensor_map.csv"
    sensor_map_csv.write_text(
        "\n".join(
            [
                "tool_id,sensor,parameter",
                "TOOL_X,value01,dc_bias",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(scrape, "SENSOR_MAP_CSV_PATH", sensor_map_csv)

    out = scrape.resolve_channel_map(
        "TOOL_X",
        tool_cfg={"channels": {"value01": "custom_bias"}},
    )

    assert out == {"value01": "custom_bias"}


def test_scrape_logger_csv_initial_run_saves_latest_state(tmp_path: Path, monkeypatch) -> None:
    raw = tmp_path / "logger_raw.csv"
    _write_logger_raw(
        raw,
        [
            "2026-02-19T10:00:00,1,10,100",
            "2026-02-19T10:15:00,2,20,200",
            "2026-02-19T10:30:00,3,30,300",
        ],
    )
    monkeypatch.setattr(scrape, "STATE_DIR", tmp_path / "state")

    out = scrape.scrape_logger_csv(
        raw_csv_path=raw,
        tool_id="TOOL_A",
        tool_cfg={"channels": {"value01": "DC Bias"}},
        now=datetime.fromisoformat("2026-02-19T10:30:00"),
        lookback_minutes=30,
    )

    assert out["timestamp"].dt.strftime("%Y-%m-%dT%H:%M:%S").tolist() == [
        "2026-02-19T10:15:00",
        "2026-02-19T10:30:00",
    ]
    assert out["tool_id"].iloc[0] == "TOOL_A"
    assert out["chamber_id"].iloc[0] == "CH1"
    state = scrape.load_last_ts("TOOL_A")
    assert state == datetime.fromisoformat("2026-02-19T10:30:00")


def test_scrape_logger_csv_second_run_uses_last_ts(tmp_path: Path, monkeypatch) -> None:
    raw = tmp_path / "logger_raw.csv"
    _write_logger_raw(
        raw,
        [
            "2026-02-19T10:10:00,1,10,100",
            "2026-02-19T10:20:00,2,20,200",
            "2026-02-19T10:25:00,3,30,300",
            "2026-02-19T10:30:00,4,40,400",
        ],
    )
    monkeypatch.setattr(scrape, "STATE_DIR", tmp_path / "state")
    scrape.save_last_ts("TOOL_A", datetime.fromisoformat("2026-02-19T10:20:00"))

    out = scrape.scrape_logger_csv(
        raw_csv_path=raw,
        tool_id="TOOL_A",
        tool_cfg={"channels": {"value01": "DC Bias"}, "chamber_id": "CH2"},
        now=datetime.fromisoformat("2026-02-19T10:30:00"),
        lookback_minutes=30,
    )

    assert out["timestamp"].dt.strftime("%Y-%m-%dT%H:%M:%S").tolist() == [
        "2026-02-19T10:25:00",
        "2026-02-19T10:30:00",
    ]
    assert out["chamber_id"].iloc[0] == "CH2"


def test_scrape_logger_csv_uses_tail_path_when_huge_threshold_small(
    tmp_path: Path,
    monkeypatch,
) -> None:
    raw = tmp_path / "logger_raw.csv"
    _write_logger_raw(raw, ["2026-02-19T10:30:00,1,10,100"])
    monkeypatch.setattr(scrape, "STATE_DIR", tmp_path / "state")

    called = {"tail": False}

    def fake_tail(path: Path, columns: list[str], n_lines: int) -> pd.DataFrame:
        called["tail"] = True
        return pd.DataFrame(
            {
                "timestamp": ["2026-02-19T10:30:00"],
                "value01": [1.0],
                "value02": [10.0],
                "value03": [100.0],
            }
        )

    monkeypatch.setattr(scrape, "read_recent_tail_as_df", fake_tail)

    scrape.scrape_logger_csv(
        raw_csv_path=raw,
        tool_id="TOOL_A",
        tool_cfg={"channels": {"value01": "DC Bias"}},
        now=datetime.fromisoformat("2026-02-19T10:30:00"),
        huge_threshold_mb=0,
    )

    assert called["tail"] is True


def test_scrape_logger_csv_missing_timestamp_column_raises(
    tmp_path: Path,
    monkeypatch,
) -> None:
    import pytest

    raw = tmp_path / "logger_raw.csv"
    lines = [
        "LOGGER_CONFIG,Sampling=1s,Channels=3",
        "RANGE,value01,0,5",
        "DATA",
        "value01,value02,value03",
        "1,10,100",
    ]
    raw.write_text("\n".join(lines) + "\n", encoding="utf-8")
    monkeypatch.setattr(scrape, "STATE_DIR", tmp_path / "state")

    with pytest.raises(ValueError, match="timestamp column is required"):
        scrape.scrape_logger_csv(
            raw_csv_path=raw,
            tool_id="TOOL_A",
            tool_cfg={"channels": {"value01": "DC Bias"}},
            now=datetime.fromisoformat("2026-02-19T10:30:00"),
        )
