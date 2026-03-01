from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from portfolio_fdc.main import aggregate


def _base_df() -> pd.DataFrame:
    ts = pd.date_range("2026-02-19T00:00:00", periods=8, freq="s")
    return pd.DataFrame(
        {
            "timestamp": ts,
            "tool_id": ["TOOL_A"] * len(ts),
            "chamber_id": ["CH1"] * len(ts),
            "dc_bias": [0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0],
            "cl2_flow": [0.0, 10.0, 12.0, 11.0, 0.0, 0.0, 0.0, 0.0],
            "apc_pressure": [100.0, 101.0, 102.0, 103.0, 104.0, 105.0, 106.0, 107.0],
        }
    )


def test_build_processes_edge_detects_one_window() -> None:
    df = _base_df()
    cfg = {
        "key_channels": {"dc_bias": "dc_bias"},
        "edge": {
            "on_threshold": 0.8,
            "off_threshold": 0.4,
            "min_on_sec": 2,
            "merge_gap_sec": 1,
        },
    }

    procs = aggregate.build_processes_edge(df, "TOOL_A", "CH1", cfg)

    assert len(procs) == 1
    p = procs[0]
    assert p["cut_method"] == "edge"
    assert p["recipe_id"] == "UNKNOWN"
    assert p["process_start"] == pd.Timestamp("2026-02-19T00:00:01")
    assert p["process_end"] == pd.Timestamp("2026-02-19T00:00:04")
    assert p["step_windows"] == [
        (
            1,
            pd.Timestamp("2026-02-19T00:00:01"),
            pd.Timestamp("2026-02-19T00:00:04"),
        )
    ]


def test_detail_csv_and_features_and_post_payloads(tmp_path: Path, monkeypatch) -> None:
    df = _base_df()
    step_windows = [
        (
            1,
            pd.Timestamp("2026-02-19T00:00:01"),
            pd.Timestamp("2026-02-19T00:00:04"),
        )
    ]

    seg = df[
        (df["timestamp"] >= step_windows[0][1]) & (df["timestamp"] <= step_windows[0][2])
    ].copy()
    seg_wide = seg[["timestamp", "dc_bias", "cl2_flow", "apc_pressure"]]

    process_id = aggregate.make_process_id(
        "TOOL_A",
        "CH1",
        step_windows[0][1].isoformat(),
        step_windows[0][2].isoformat(),
        "edge",
    )

    long_df = aggregate.wide_to_long_detail(seg_wide, "TOOL_A", "CH1", process_id)
    assert len(long_df) == len(seg_wide) * 3
    assert set(long_df["parameter"].unique()) == {"dc_bias", "cl2_flow", "apc_pressure"}

    out_path = aggregate.save_detail_csv(long_df, tmp_path / "detail", "TOOL_A", "CH1", process_id)
    assert Path(out_path).exists()

    feats = aggregate.compute_features(
        df_wide=df,
        step_windows=step_windows,
        parameters=["dc_bias", "cl2_flow", "apc_pressure"],
        feature_types=["mean", "max", "min", "std"],
    )
    assert len(feats) == 12

    calls: list[tuple[str, Any]] = []

    def fake_api_post(db_api: str, path: str, payload: Any) -> dict[str, bool]:
        calls.append((path, payload))
        return {"ok": True}

    monkeypatch.setattr(aggregate, "api_post", fake_api_post)

    aggregate.post_one_process(
        db_api="http://dummy",
        tool_id="TOOL_A",
        chamber_id="CH1",
        recipe_id="UNKNOWN",
        process_id=process_id,
        start_ts=step_windows[0][1],
        end_ts=step_windows[0][2],
        raw_csv_path=out_path,
    )
    aggregate.post_step_windows(
        db_api="http://dummy",
        process_id=process_id,
        step_windows=step_windows,
        source_channel="dc_bias",
    )
    aggregate.post_features(
        db_api="http://dummy",
        process_id=process_id,
        feats=feats,
    )

    assert [c[0] for c in calls] == ["/processes", "/step_windows/bulk", "/parameters/bulk"]
    assert calls[0][1]["process_id"] == process_id
    assert calls[1][1][0]["step_no"] == 1
    assert calls[2][1][0]["process_id"] == process_id


def test_post_features_skips_empty_payload(monkeypatch) -> None:
    calls: list[tuple[str, Any]] = []

    def fake_api_post(db_api: str, path: str, payload: Any) -> dict[str, bool]:
        calls.append((path, payload))
        return {"ok": True}

    monkeypatch.setattr(aggregate, "api_post", fake_api_post)

    aggregate.post_features(db_api="http://dummy", process_id="p1", feats=[])

    assert calls == []


def test_main_dry_run_skips_db_posts(tmp_path: Path, monkeypatch) -> None:
    input_csv = tmp_path / "scrape_out.csv"
    cfg_yaml = tmp_path / "aggregate_tools.yaml"
    detail_dir = tmp_path / "detail"

    ts = pd.date_range("2026-02-19T00:00:00", periods=8, freq="s")
    df = pd.DataFrame(
        {
            "timestamp": ts,
            "tool_id": ["TOOL_X"] * len(ts),
            "chamber_id": ["CH1"] * len(ts),
            "dc_bias": [0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0],
            "cl2_flow": [0.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 0.0],
            "apc_pressure": [100.0, 101.0, 102.0, 103.0, 104.0, 105.0, 106.0, 107.0],
        }
    )
    df.to_csv(input_csv, index=False)
    cfg_yaml.write_text(yaml.safe_dump({"tools": {}}), encoding="utf-8")

    def _raise_if_called(*args, **kwargs):
        raise AssertionError("DB POST should not be called in dry-run mode")

    monkeypatch.setattr(aggregate, "post_one_process", _raise_if_called)
    monkeypatch.setattr(aggregate, "post_step_windows", _raise_if_called)
    monkeypatch.setattr(aggregate, "post_features", _raise_if_called)

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "aggregate.py",
            "--input",
            str(input_csv),
            "--config",
            str(cfg_yaml),
            "--detail-out",
            str(detail_dir),
            "--dry-run",
        ],
    )

    aggregate.main()

    out_files = list(detail_dir.glob("detail_*.csv"))
    assert out_files
