from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from portfolio_fdc.core.segmentation.classifier import RecipeClassifier
from portfolio_fdc.core.segmentation.models import StepBundle, StepPeak
from portfolio_fdc.main import aggregate

DUMMY_RULES_PATH = (
    Path(__file__).resolve().parents[1]
    / "src"
    / "portfolio_fdc"
    / "configs"
    / "recipe_rules_dummy.yaml"
)


def _make_step_peak(channel: str, value: float) -> StepPeak:
    """分類テスト用に固定時刻の `StepPeak` を生成する。"""
    ts = pd.Timestamp("2026-02-19T00:00:00").to_pydatetime()
    return StepPeak(
        channel=channel,
        start_ts=ts,
        end_ts=ts,
        duration_sec=0.0,
        mean=value,
        max=value,
        min=value,
        std=0.0,
    )


def _base_df() -> pd.DataFrame:
    """エッジ検出の基本入力として使う小さな時系列を返す。"""
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


def _recipe_classify_df() -> tuple[pd.DataFrame, list[tuple[pd.Timestamp, pd.Timestamp]]]:
    """4ステップ分類用のダミー時系列とキューを返す。"""
    ts = pd.date_range("2026-02-19T00:00:00", periods=18, freq="s")
    df = pd.DataFrame(
        {
            "timestamp": ts,
            "tool_id": ["TOOL_A"] * len(ts),
            "chamber_id": ["CH1"] * len(ts),
            "dc_bias": [
                2.0,
                2.0,
                0.0,
                0.01,
                2.8,
                2.8,
                0.0,
                0.01,
                2.4,
                2.4,
                0.0,
                0.01,
                1.7,
                1.7,
                0.0,
                0.0,
                0.0,
                0.0,
            ],
            "cl2_flow": [
                12.0,
                12.0,
                0.0,
                0.0,
                20.0,
                20.0,
                0.0,
                0.0,
                17.0,
                17.0,
                0.0,
                0.0,
                13.0,
                13.0,
                0.0,
                0.0,
                0.0,
                0.0,
            ],
        }
    )
    queue = [
        (pd.Timestamp("2026-02-19T00:00:00"), pd.Timestamp("2026-02-19T00:00:01")),
        (pd.Timestamp("2026-02-19T00:00:04"), pd.Timestamp("2026-02-19T00:00:05")),
        (pd.Timestamp("2026-02-19T00:00:08"), pd.Timestamp("2026-02-19T00:00:09")),
        (pd.Timestamp("2026-02-19T00:00:12"), pd.Timestamp("2026-02-19T00:00:13")),
    ]
    return df, queue


def _recipe_d_3step_df() -> tuple[pd.DataFrame, list[tuple[pd.Timestamp, pd.Timestamp]]]:
    """3ステップ（分割対象）分類用の入力データを返す。"""
    ts = pd.date_range("2026-02-19T01:00:00", periods=12, freq="s")
    df = pd.DataFrame(
        {
            "timestamp": ts,
            "tool_id": ["TOOL_A"] * len(ts),
            "chamber_id": ["CH1"] * len(ts),
            "dc_bias": [1.8, 1.8, 0.0, 2.6, 2.6, 0.0, 2.1, 2.1, 0.0, 0.0, 0.0, 0.0],
            "cl2_flow": [12.0, 12.0, 0.0, 19.0, 19.0, 0.0, 15.0, 15.0, 0.0, 0.0, 0.0, 0.0],
        }
    )
    queue = [
        (pd.Timestamp("2026-02-19T01:00:00"), pd.Timestamp("2026-02-19T01:00:01")),
        (pd.Timestamp("2026-02-19T01:00:03"), pd.Timestamp("2026-02-19T01:00:04")),
        (pd.Timestamp("2026-02-19T01:00:06"), pd.Timestamp("2026-02-19T01:00:07")),
    ]
    return df, queue


def _recipe_b_df() -> tuple[pd.DataFrame, list[tuple[pd.Timestamp, pd.Timestamp]]]:
    """RECIPE_B に一致する入力データを返す。"""
    ts = pd.date_range("2026-02-19T02:00:00", periods=18, freq="s")
    df = pd.DataFrame(
        {
            "timestamp": ts,
            "tool_id": ["TOOL_A"] * len(ts),
            "chamber_id": ["CH1"] * len(ts),
            "dc_bias": [
                1.6,
                1.6,
                0.0,
                0.0,
                3.2,
                3.2,
                0.0,
                0.0,
                2.5,
                2.5,
                0.0,
                0.0,
                1.4,
                1.4,
                0.0,
                0.0,
                0.0,
                0.0,
            ],
            "cl2_flow": [
                10.0,
                10.0,
                0.0,
                0.0,
                22.0,
                22.0,
                0.0,
                0.0,
                16.0,
                16.0,
                0.0,
                0.0,
                11.0,
                11.0,
                0.0,
                0.0,
                0.0,
                0.0,
            ],
        }
    )
    queue = [
        (pd.Timestamp("2026-02-19T02:00:00"), pd.Timestamp("2026-02-19T02:00:01")),
        (pd.Timestamp("2026-02-19T02:00:04"), pd.Timestamp("2026-02-19T02:00:05")),
        (pd.Timestamp("2026-02-19T02:00:08"), pd.Timestamp("2026-02-19T02:00:09")),
        (pd.Timestamp("2026-02-19T02:00:12"), pd.Timestamp("2026-02-19T02:00:13")),
    ]
    return df, queue


def _recipe_c_df() -> tuple[pd.DataFrame, list[tuple[pd.Timestamp, pd.Timestamp]]]:
    """RECIPE_C に一致する入力データを返す。"""
    ts = pd.date_range("2026-02-19T03:00:00", periods=18, freq="s")
    df = pd.DataFrame(
        {
            "timestamp": ts,
            "tool_id": ["TOOL_A"] * len(ts),
            "chamber_id": ["CH1"] * len(ts),
            "dc_bias": [
                2.3,
                2.3,
                0.0,
                0.0,
                2.5,
                2.5,
                0.0,
                0.0,
                2.9,
                2.9,
                0.0,
                0.0,
                2.0,
                2.0,
                0.0,
                0.0,
                0.0,
                0.0,
            ],
            "cl2_flow": [
                15.0,
                15.0,
                0.0,
                0.0,
                18.0,
                18.0,
                0.0,
                0.0,
                21.0,
                21.0,
                0.0,
                0.0,
                14.0,
                14.0,
                0.0,
                0.0,
                0.0,
                0.0,
            ],
        }
    )
    queue = [
        (pd.Timestamp("2026-02-19T03:00:00"), pd.Timestamp("2026-02-19T03:00:01")),
        (pd.Timestamp("2026-02-19T03:00:04"), pd.Timestamp("2026-02-19T03:00:05")),
        (pd.Timestamp("2026-02-19T03:00:08"), pd.Timestamp("2026-02-19T03:00:09")),
        (pd.Timestamp("2026-02-19T03:00:12"), pd.Timestamp("2026-02-19T03:00:13")),
    ]
    return df, queue


def test_classify_recipe_from_peaks_matches_dummy_rules(monkeypatch) -> None:
    """標準ダミーデータが RECIPE_A に分類されることを確認する。"""
    df, queue = _recipe_classify_df()
    monkeypatch.setattr(
        aggregate,
        "RECIPE_RULES_PATH",
        DUMMY_RULES_PATH,
    )

    recipe = aggregate.classify_recipe_from_peaks(queue, df)

    assert recipe == "RECIPE_A"


def test_classify_recipe_from_peaks_returns_unknown_on_out_of_range(monkeypatch) -> None:
    """範囲外値を含む場合は UNKNOWN となることを確認する。"""
    df, queue = _recipe_classify_df()
    df.loc[(df["timestamp"] >= queue[1][0]) & (df["timestamp"] <= queue[1][1]), "dc_bias"] = 9.9
    monkeypatch.setattr(
        aggregate,
        "RECIPE_RULES_PATH",
        DUMMY_RULES_PATH,
    )

    recipe = aggregate.classify_recipe_from_peaks(queue, df)

    assert recipe == "UNKNOWN"


def test_classify_recipe_from_peaks_matches_dummy_3step_rule(monkeypatch) -> None:
    """3ステップルールが正しく分類されることを確認する。"""
    df, queue = _recipe_d_3step_df()
    monkeypatch.setattr(
        aggregate,
        "RECIPE_RULES_PATH",
        DUMMY_RULES_PATH,
    )

    recipe = aggregate.classify_recipe_from_peaks(queue, df)

    assert recipe == "RECIPE_D_3STEP"


def test_classify_recipe_from_peaks_matches_dummy_recipe_b(monkeypatch) -> None:
    """RECIPE_B 相当データが RECIPE_B に分類されることを確認する。"""
    df, queue = _recipe_b_df()
    monkeypatch.setattr(aggregate, "RECIPE_RULES_PATH", DUMMY_RULES_PATH)

    recipe = aggregate.classify_recipe_from_peaks(queue, df)

    assert recipe == "RECIPE_B"


def test_classify_recipe_from_peaks_matches_dummy_recipe_c(monkeypatch) -> None:
    """RECIPE_C 相当データが RECIPE_C に分類されることを確認する。"""
    df, queue = _recipe_c_df()
    monkeypatch.setattr(aggregate, "RECIPE_RULES_PATH", DUMMY_RULES_PATH)

    recipe = aggregate.classify_recipe_from_peaks(queue, df)

    assert recipe == "RECIPE_C"


def test_classify_recipe_from_peaks_matches_on_rule_boundaries(monkeypatch) -> None:
    """閾値境界値でも一致判定されることを確認する。"""
    df, queue = _recipe_classify_df()
    df.loc[
        (df["timestamp"] >= queue[0][0]) & (df["timestamp"] <= queue[0][1]), ["dc_bias", "cl2_flow"]
    ] = [1.8, 10.0]
    df.loc[
        (df["timestamp"] >= queue[1][0]) & (df["timestamp"] <= queue[1][1]), ["dc_bias", "cl2_flow"]
    ] = [3.0, 22.0]
    df.loc[
        (df["timestamp"] >= queue[2][0]) & (df["timestamp"] <= queue[2][1]), ["dc_bias", "cl2_flow"]
    ] = [2.6, 19.0]
    df.loc[
        (df["timestamp"] >= queue[3][0]) & (df["timestamp"] <= queue[3][1]), ["dc_bias", "cl2_flow"]
    ] = [1.9, 15.0]
    monkeypatch.setattr(aggregate, "RECIPE_RULES_PATH", DUMMY_RULES_PATH)

    recipe = aggregate.classify_recipe_from_peaks(queue, df)

    assert recipe == "RECIPE_A"


def test_classify_recipe_from_peaks_returns_unknown_on_empty_queue(monkeypatch) -> None:
    """キューが空なら UNKNOWN を返すことを確認する。"""
    df, _ = _recipe_classify_df()
    monkeypatch.setattr(aggregate, "RECIPE_RULES_PATH", DUMMY_RULES_PATH)

    recipe = aggregate.classify_recipe_from_peaks([], df)

    assert recipe == "UNKNOWN"


def test_classify_recipe_from_peaks_returns_unknown_without_timestamp(monkeypatch) -> None:
    """timestamp 列が欠落した入力で UNKNOWN となることを確認する。"""
    df, queue = _recipe_classify_df()
    monkeypatch.setattr(aggregate, "RECIPE_RULES_PATH", DUMMY_RULES_PATH)

    recipe = aggregate.classify_recipe_from_peaks(queue, df.drop(columns=["timestamp"]))

    assert recipe == "UNKNOWN"


def test_recipe_classifier_returns_unknown_when_channels_missing() -> None:
    """必須チャネルが欠損する bundle で UNKNOWN になることを確認する。"""
    classifier = RecipeClassifier(aggregate.load_yaml(DUMMY_RULES_PATH))

    bundles = [
        StepBundle(
            step_no=1,
            dc_bias=_make_step_peak("dc_bias", 2.0),
            cl2_flow=_make_step_peak("cl2_flow", 12.0),
        ),
        StepBundle(step_no=2, dc_bias=None, cl2_flow=_make_step_peak("cl2_flow", 20.0)),
        StepBundle(
            step_no=3,
            dc_bias=_make_step_peak("dc_bias", 2.4),
            cl2_flow=_make_step_peak("cl2_flow", 17.0),
        ),
        StepBundle(
            step_no=4,
            dc_bias=_make_step_peak("dc_bias", 1.7),
            cl2_flow=_make_step_peak("cl2_flow", 13.0),
        ),
    ]

    recipe = classifier.classify(bundles)

    assert recipe == "UNKNOWN"


def test_recipe_classifier_presplit_validates_fourth_bundle() -> None:
    """presplit 4本目の負例/正例が正しく判定されることを確認する。"""
    classifier = RecipeClassifier(aggregate.load_yaml(DUMMY_RULES_PATH))

    bundles = [
        StepBundle(
            step_no=1,
            dc_bias=_make_step_peak("dc_bias", 1.8),
            cl2_flow=_make_step_peak("cl2_flow", 12.0),
        ),
        StepBundle(
            step_no=2,
            dc_bias=_make_step_peak("dc_bias", 2.6),
            cl2_flow=_make_step_peak("cl2_flow", 19.0),
        ),
        StepBundle(
            step_no=3,
            dc_bias=_make_step_peak("dc_bias", 2.1),
            cl2_flow=_make_step_peak("cl2_flow", 15.0),
        ),
        StepBundle(
            step_no=4,
            dc_bias=_make_step_peak("dc_bias", 9.9),
            cl2_flow=_make_step_peak("cl2_flow", 15.0),
        ),
    ]

    recipe = classifier.classify(bundles)

    assert recipe == "UNKNOWN"

    valid_bundles = [
        StepBundle(
            step_no=1,
            dc_bias=_make_step_peak("dc_bias", 1.8),
            cl2_flow=_make_step_peak("cl2_flow", 12.0),
        ),
        StepBundle(
            step_no=2,
            dc_bias=_make_step_peak("dc_bias", 2.6),
            cl2_flow=_make_step_peak("cl2_flow", 19.0),
        ),
        StepBundle(
            step_no=3,
            dc_bias=_make_step_peak("dc_bias", 2.1),
            cl2_flow=_make_step_peak("cl2_flow", 15.0),
        ),
        StepBundle(
            step_no=4,
            dc_bias=_make_step_peak("dc_bias", 2.0),
            cl2_flow=_make_step_peak("cl2_flow", 14.0),
        ),
    ]

    valid_recipe = classifier.classify(valid_bundles)

    assert valid_recipe == "RECIPE_D_3STEP"


def test_build_processes_edge_detects_one_window() -> None:
    """エッジ法で1つのプロセス窓が検出されることを確認する。"""
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
    """detail保存・特徴量生成・POST payload 生成の一連処理を確認する。"""
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
        """API 呼び出しを記録するテストダブル。"""
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
    """空特徴量入力時に API 呼び出しを行わないことを確認する。"""
    calls: list[tuple[str, Any]] = []

    def fake_api_post(db_api: str, path: str, payload: Any) -> dict[str, bool]:
        """API 呼び出しを記録するテストダブル。"""
        calls.append((path, payload))
        return {"ok": True}

    monkeypatch.setattr(aggregate, "api_post", fake_api_post)

    aggregate.post_features(db_api="http://dummy", process_id="p1", feats=[])

    assert calls == []


def test_main_dry_run_skips_db_posts(tmp_path: Path, monkeypatch) -> None:
    """dry-run モードでは DB POST が呼ばれないことを確認する。"""
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
        """呼ばれたら失敗させるガード。"""
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


def test_main_non_dry_run_cleans_up_on_post_error(tmp_path: Path, monkeypatch) -> None:
    """POST 途中失敗時に作成済みプロセスを削除することを確認する。"""
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

    called: dict[str, str | None] = {"created": None, "deleted": None}

    def fake_post_one_process(
        db_api: str,
        tool_id: str,
        chamber_id: str,
        recipe_id: str,
        process_id: str,
        start_ts: pd.Timestamp,
        end_ts: pd.Timestamp,
        raw_csv_path: str,
    ) -> None:
        """process 作成成功を記録するテストダブル。"""
        called["created"] = process_id

    def fail_step_windows(
        db_api: str,
        process_id: str,
        step_windows: list[tuple[int, pd.Timestamp, pd.Timestamp]],
        source_channel: str,
    ) -> None:
        """step window 投稿失敗を模擬するテストダブル。"""
        raise RuntimeError("forced step window failure")

    def fake_delete_process(db_api: str, process_id: str) -> None:
        """削除呼び出しを記録するテストダブル。"""
        called["deleted"] = process_id

    monkeypatch.setattr(aggregate, "post_one_process", fake_post_one_process)
    monkeypatch.setattr(aggregate, "post_step_windows", fail_step_windows)
    monkeypatch.setattr(aggregate, "delete_process", fake_delete_process)

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
            "--db-api",
            "http://dummy",
        ],
    )

    aggregate.main()

    assert called["created"] is not None
    assert called["deleted"] == called["created"]
