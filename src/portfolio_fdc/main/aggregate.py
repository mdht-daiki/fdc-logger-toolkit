from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
from typing import Any

import pandas as pd
import requests
import yaml


def load_yaml(path: Path) -> dict:
    """YAMLファイルを読み込み、辞書として返す。空の場合は空辞書を返す。"""
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def api_post(db_api: str, path: str, payload: Any) -> dict:
    """DB APIにPOSTリクエストを送り、JSONレスポンスを返す。"""
    r = requests.post(f"{db_api}{path}", json=payload, timeout=30)
    r.raise_for_status()
    return r.json()


def api_delete(db_api: str, path: str, payload: Any) -> dict:
    """DB APIにDELETEリクエストを送り、JSONレスポンスを返す。"""
    r = requests.delete(f"{db_api}{path}", json=payload, timeout=30)
    r.raise_for_status()
    return r.json()


def api_get(db_api: str, path: str, params: dict | None = None) -> dict:
    """DB APIにGETリクエストを送り、JSONレスポンスを返す。"""
    r = requests.get(f"{db_api}{path}", params=params, timeout=30)
    r.raise_for_status()
    return r.json()


def to_dt(s: Any) -> pd.Timestamp:
    """任意の値をPandasの日時型に変換する。失敗時はNaTになる。"""
    return pd.to_datetime(s, errors="coerce")


def make_process_id(
    tool_id: str, chamber_id: str, start_ts: str, end_ts: str, cut_method: str
) -> str:
    """装置・チャンバー・時間範囲・切り出し方式から一意なprocess_idを生成する。"""
    base = f"{tool_id}|{chamber_id}|{start_ts}|{end_ts}|{cut_method}"
    return hashlib.sha1(base.encode("utf-8")).hexdigest()


def ensure_cols(df: pd.DataFrame, cols: list[str]) -> None:
    """DataFrameに必須列が存在するか検証し、不足があれば例外を送出する。"""
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise RuntimeError(f"Missing required columns: {missing}")


def wide_to_long_detail(
    df_seg: pd.DataFrame,
    tool_id: str,
    chamber_id: str,
    process_id: str,
    keep_cols: list[str] | None = None,
) -> pd.DataFrame:
    """ワイド形式の時系列データを詳細保存用のロング形式に変換する。"""
    keep_cols = keep_cols or []
    base_cols = ["timestamp"] + keep_cols
    value_cols = [c for c in df_seg.columns if c not in base_cols]
    out = df_seg.melt(
        id_vars=base_cols, value_vars=value_cols, var_name="parameter", value_name="value"
    )
    out["tool_id"] = tool_id
    out["chamber_id"] = chamber_id
    out["process_id"] = process_id
    return out[["timestamp", "tool_id", "chamber_id", "process_id", "parameter", "value"]]


def save_detail_csv(
    df_long: pd.DataFrame, out_dir: Path, tool_id: str, chamber_id: str, process_id: str
) -> str:
    """ロング形式データを詳細CSVとして保存し、保存パスを返す。"""
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"detail_{tool_id}_{chamber_id}_{process_id}.csv"
    df_long.to_csv(path.as_posix(), index=False)
    return path.as_posix()


def compute_features(
    df_wide: pd.DataFrame,
    step_windows: list[tuple[int, pd.Timestamp, pd.Timestamp]],
    parameters: list[str],
    feature_types: list[str],
) -> list[dict]:
    """各ステップ区間・各パラメータに対して統計特徴量を計算する。"""
    feats: list[dict] = []
    for step_no, s, e in step_windows:
        seg = df_wide[(df_wide["timestamp"] >= s) & (df_wide["timestamp"] <= e)]
        if seg.empty:
            continue
        for p in parameters:
            if p not in seg.columns:
                continue
            x = pd.to_numeric(seg[p], errors="coerce").dropna()
            if x.empty:
                continue
            for ft in feature_types:
                if ft == "mean":
                    v = float(x.mean())
                elif ft == "max":
                    v = float(x.max())
                elif ft == "min":
                    v = float(x.min())
                elif ft == "std":
                    v = float(x.std(ddof=0))
                else:
                    continue
                feats.append(
                    {
                        "parameter": p,
                        "step_no": int(step_no),
                        "feature_type": ft,
                        "feature_value": v,
                    }
                )
    return feats


def detect_edge_windows(
    series: pd.Series, on_th: float, off_th: float, min_on_sec: int, merge_gap_sec: int
) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    """閾値ヒステリシスでON区間を検出し、短区間除外と近接区間マージを行う。"""
    # series index must be timestamp (sorted), values numeric
    s = series.dropna()
    if s.empty:
        return []
    ts = s.index
    v = s.values
    in_on = False
    start = None
    windows: list[tuple[pd.Timestamp, pd.Timestamp]] = []
    for i in range(len(v)):
        if not in_on and v[i] >= on_th:
            in_on = True
            start = ts[i]
        elif in_on and v[i] <= off_th:
            end = ts[i]
            in_on = False
            if start is not None:
                windows.append((start, end))
            start = None
    # if still on at end
    if in_on and start is not None:
        windows.append((start, ts[-1]))
    # filter short
    filtered: list[tuple[pd.Timestamp, pd.Timestamp]] = []
    for a, b in windows:
        if (b - a).total_seconds() >= min_on_sec:
            filtered.append((a, b))
    # merge close gaps
    merged: list[tuple[pd.Timestamp, pd.Timestamp]] = []
    for w in filtered:
        if not merged:
            merged.append(w)
            continue
        prev_a, prev_b = merged[-1]
        gap = (w[0] - prev_b).total_seconds()
        if gap <= merge_gap_sec:
            merged[-1] = (prev_a, max(prev_b, w[1]))
        else:
            merged.append(w)
    return merged


def detect_steppeaks(
    df: pd.DataFrame,
    dc_key: str,
    cl2_key: str,
    dc_on: float,
    dc_off: float,
    cl2_on: float,
) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    """dc_biasのON区間候補を抽出し、cl2_flowが有効な区間のみをsteppeakとして返す。"""
    # minimal: detect "on" windows by dc_bias, but also ensure cl2 is active in window
    ensure_cols(df, ["timestamp", dc_key, cl2_key])
    d = df.copy()
    d["timestamp"] = pd.to_datetime(d["timestamp"], errors="coerce")
    d = d.dropna(subset=["timestamp"]).sort_values("timestamp")
    d = d.set_index("timestamp")
    dc = pd.to_numeric(d[dc_key], errors="coerce")
    raw = detect_edge_windows(dc, on_th=dc_on, off_th=dc_off, min_on_sec=3, merge_gap_sec=2)
    peaks: list[tuple[pd.Timestamp, pd.Timestamp]] = []
    cl2 = pd.to_numeric(d[cl2_key], errors="coerce")
    for a, b in raw:
        seg = cl2[(cl2.index >= a) & (cl2.index <= b)].dropna()
        if seg.empty:
            continue
        if seg.max() >= cl2_on:
            peaks.append((a, b))
    return peaks


def classify_recipe_from_peaks(
    steppeak_queue: list[tuple[pd.Timestamp, pd.Timestamp]], df: pd.DataFrame
) -> str:
    """検出したピーク列からレシピIDを判定する（現状はプレースホルダ）。"""
    # Placeholder
    # your real logic uses dc_bias/cl2_flow levels within each peak, plus conditional rules.
    # For now, return a fixed recipe.
    return "RECIPE_A"


def split_one_peak_into_two(
    peak: tuple[pd.Timestamp, pd.Timestamp], ratio: float
) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    """1つのピーク区間を指定比率で2分割し、2つの区間として返す。"""
    if not (0 < ratio < 1):
        raise ValueError("Ratio must be between 0 and 1")
    a, b = peak
    total = (b - a).total_seconds()
    if total <= 1:
        return [peak]
    cut = a + pd.Timedelta(seconds=int(total * ratio))
    return [(a, cut), (cut, b)]


def build_processes_edge(df: pd.DataFrame, tool_id: str, chamber_id: str, cfg: dict) -> list[dict]:
    """edge方式でプロセス区間を作成し、1区間を1ステップとして返す。"""
    df2 = df.copy()
    df2["timestamp"] = pd.to_datetime(df2["timestamp"], errors="coerce")
    df2 = df2.dropna(subset=["timestamp"]).sort_values("timestamp")
    df2 = df2.set_index("timestamp")
    key = cfg["key_channels"]["dc_bias"]
    on_th = float(cfg["edge"]["on_threshold"])
    off_th = float(cfg["edge"]["off_threshold"])
    min_on = int(cfg["edge"].get("min_on_sec", 5))
    merge_gap = int(cfg["edge"].get("merge_gap_sec", 3))
    s = pd.to_numeric(df2[key], errors="coerce")
    windows = detect_edge_windows(s, on_th, off_th, min_on, merge_gap)
    out: list[dict] = []
    for a, b in windows:
        out.append(
            {
                "cut_method": "edge",
                "recipe_id": "UNKNOWN",
                "process_start": a,
                "process_end": b,
                "step_windows": [(1, a, b)],
            }
        )
    return out


def build_processes_steppeak(
    df: pd.DataFrame, tool_id: str, chamber_id: str, cfg: dict
) -> list[dict]:
    """steppeak方式でピーク列から4ステップ単位のプロセスを構築する。"""
    df2 = df.copy()
    df2["timestamp"] = pd.to_datetime(df2["timestamp"], errors="coerce")
    df2 = df2.dropna(subset=["timestamp"]).sort_values("timestamp")
    dc_key = cfg["key_channels"]["dc_bias"]
    cl2_key = cfg["key_channels"]["cl2_flow"]
    sp = cfg["steppeak"]
    peaks = detect_steppeaks(
        df2,
        dc_key,
        cl2_key,
        float(sp["dc_bias_on"]),
        float(sp["dc_bias_off"]),
        float(sp["cl2_flow_on"]),
    )
    # make queue of latest 4 peaks -> build a process per 4 peaks in sequence
    out: list[dict] = []
    i = 0
    while i + 3 < len(peaks):
        q = peaks[i : i + 4]
        recipe = classify_recipe_from_peaks(q, df2)
        a = q[0][0]
        b = q[3][1]
        step_windows = [
            (1, q[0][0], q[0][1]),
            (2, q[1][0], q[1][1]),
            (3, q[2][0], q[2][1]),
            (4, q[3][0], q[3][1]),
        ]
        out.append(
            {
                "cut_method": "steppeak",
                "recipe_id": recipe,
                "process_start": a,
                "process_end": b,
                "step_windows": step_windows,
            }
        )
        i += 4
    # handle 3-step recipes: placeholder logic (if you detect 3 peaks pattern, split one peak)
    # You can implement: detect 3 peaks then split step2 into 2 using split_ratio to make 4 steps.
    return out


def _to_iso_naive_utc(ts: pd.Timestamp) -> str:
    """タイムゾーンをUTCに変換し、ISOフォーマットの文字列を返す。"""
    t = pd.Timestamp(ts)
    if t.tz is not None:
        t = t.tz_convert("UTC").tz_localize(None)
    return t.isoformat()


def post_one_process(
    db_api: str,
    tool_id: str,
    chamber_id: str,
    recipe_id: str,
    process_id: str,
    start_ts: pd.Timestamp,
    end_ts: pd.Timestamp,
    raw_csv_path: str,
) -> None:
    """1プロセス分のメタ情報をDB APIに登録する。"""
    payload = {
        "process_id": process_id,
        "tool_id": tool_id,
        "chamber_id": chamber_id,
        "recipe_id": recipe_id,
        "start_ts": _to_iso_naive_utc(start_ts),
        "end_ts": _to_iso_naive_utc(end_ts),
        "raw_csv_path": raw_csv_path,
    }
    api_post(db_api, "/processes", payload)


def post_step_windows(
    db_api: str,
    process_id: str,
    step_windows: list[tuple[int, pd.Timestamp, pd.Timestamp]],
    source_channel: str,
) -> None:
    """プロセスの各ステップ時間窓をDB APIへ一括登録する。"""
    payload = [
        {
            "process_id": process_id,
            "step_no": int(no),
            "start_ts": _to_iso_naive_utc(s),
            "end_ts": _to_iso_naive_utc(e),
            "source_channel": source_channel,
        }
        for (no, s, e) in step_windows
    ]
    api_post(db_api, "/step_windows/bulk", payload)


def post_features(db_api: str, process_id: str, feats: list[dict]) -> None:
    """計算済み特徴量をDB APIへ一括登録する。"""
    payload = [
        {
            "process_id": process_id,
            "parameter": f["parameter"],
            "step_no": int(f["step_no"]),
            "feature_type": f["feature_type"],
            "feature_value": f["feature_value"],
        }
        for f in feats
    ]
    if payload:
        api_post(db_api, "/parameters/bulk", payload)


def delete_process(db_api: str, process_id: str) -> None:
    """指定したprocess_idのプロセスと関連データをDBから削除する。"""
    api_delete(db_api, "/processes", {"process_id": process_id})


def main():
    """入力CSVを集約し、プロセス切り出し・詳細保存・特徴量計算・DB登録を実行する。"""
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--input", required=True, help="scrape output CSV (wide, includes tool_id/chamber_id)"
    )
    ap.add_argument("--db-api", default="http://localhost:8000")
    ap.add_argument("--config", default="src/portfolio_fdc/configs/aggregate_tools.yaml")
    ap.add_argument("--detail-out", default="data/detail")
    ap.add_argument("--feature-types", default="mean,max,min,std")
    ap.add_argument("--parameters", default="dc_bias,cl2_flow,apc_pressure")
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="DB APIへはPOSTせず、ローカル処理（切り出し/CSV保存/特徴量計算）のみ実行する",
    )
    args = ap.parse_args()
    cfg = load_yaml(Path(args.config)).get("tools", {})
    df = pd.read_csv(args.input)
    ensure_cols(df, ["timestamp", "tool_id", "chamber_id"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df = df.dropna(subset=["timestamp"]).sort_values("timestamp")
    feature_types = [x.strip() for x in args.feature_types.split(",") if x.strip()]
    parameters = [x.strip() for x in args.parameters.split(",") if x.strip()]
    detail_out = Path(args.detail_out)
    # tool/chamberごとに処理
    for (tool_id, chamber_id), g in df.groupby(["tool_id", "chamber_id"]):
        if tool_id not in cfg:
            # unknown tool: fallback edge on dc_bias if exists
            local = {
                "mode": "edge",
                "key_channels": {"dc_bias": "dc_bias"},
                "edge": {
                    "on_threshold": 0.8,
                    "off_threshold": 0.4,
                    "min_on_sec": 5,
                    "merge_gap_sec": 3,
                },
            }
        else:
            local = cfg[tool_id]
        mode = local.get("mode", "edge")
        g2 = g.copy()
        key_channels = local.get("key_channels", {})
        required = [key_channels.get("dc_bias")]
        if mode == "steppeak":
            required.append(key_channels.get("cl2_flow"))
        missing = [c for c in required if not c or c not in g2.columns]
        if missing:
            print(f"SKIP tool={tool_id} chamber={chamber_id} missing_channels={missing}")
            continue
        if mode == "steppeak":
            procs = build_processes_steppeak(g2, tool_id, chamber_id, local)
            source_ch = key_channels["dc_bias"]
        else:
            procs = build_processes_edge(g2, tool_id, chamber_id, local)
            source_ch = key_channels["dc_bias"]
        for p in procs:
            start_ts = p["process_start"]
            end_ts = p["process_end"]
            cut_method = p["cut_method"]
            recipe_id = p["recipe_id"]
            process_id = make_process_id(
                tool_id,
                chamber_id,
                _to_iso_naive_utc(start_ts),
                _to_iso_naive_utc(end_ts),
                cut_method,
            )
            # extract segment (process range)
            seg = g2[(g2["timestamp"] >= start_ts) & (g2["timestamp"] <= end_ts)].copy()
            if seg.empty:
                continue
            # detail csv: long
            seg_wide = seg.copy()
            seg_wide["timestamp"] = pd.to_datetime(seg_wide["timestamp"], errors="coerce")
            # keep only numeric parameter columns + timestamp
            # keep = ["timestamp"]
            # cols_keep = ["timestamp"]
            cols_param = [c for c in parameters if c in seg_wide.columns]
            seg_wide = seg_wide[["timestamp"] + cols_param]
            long_df = wide_to_long_detail(seg_wide, tool_id, chamber_id, process_id, keep_cols=[])
            raw_csv_path = save_detail_csv(long_df, detail_out, tool_id, chamber_id, process_id)
            # features computed on ORIGINAL wide with parameters
            base = g2.copy()
            base["timestamp"] = pd.to_datetime(base["timestamp"], errors="coerce")
            base = base.dropna(subset=["timestamp"]).sort_values("timestamp")
            feats = compute_features(base, p["step_windows"], cols_param, feature_types)
            if args.dry_run:
                print(
                    f"DRY-RUN: tool={tool_id} chamber={chamber_id} mode={mode} "
                    f"process_id={process_id} recipe={recipe_id} "
                    f"steps={len(p['step_windows'])} features={len(feats)} "
                    f"detail={raw_csv_path}"
                )
            else:
                # write to DB
                created = False
                try:
                    post_one_process(
                        args.db_api,
                        tool_id,
                        chamber_id,
                        recipe_id,
                        process_id,
                        start_ts,
                        end_ts,
                        raw_csv_path,
                    )
                    created = True
                    post_step_windows(args.db_api, process_id, p["step_windows"], source_ch)
                    post_features(args.db_api, process_id, feats)
                    print(
                        f"OK: tool={tool_id} chamber={chamber_id} mode={mode} "
                        f"process_id={process_id} recipe={recipe_id} "
                        f"steps={len(p['step_windows'])} features={len(feats)}"
                    )
                except Exception as e:
                    print(
                        f"ERROR posting process_id={process_id} "
                        f"tool={tool_id} chamber={chamber_id} "
                        f"error={e}"
                    )
                    # attempt cleanup if process was partially created
                    if created:
                        try:
                            delete_process(args.db_api, process_id)
                        except Exception as e2:
                            print(f"ERROR during cleanup of process_id={process_id} error={e2}")


if __name__ == "__main__":
    main()
