"""しきい値ベースでピーク区間を検出するロジック。"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

import numpy as np
import pandas as pd

from .models import StepPeak


@dataclass(frozen=True)
class PeakDetectorConfig:
    """ピーク検出のしきい値・最小長・マージ条件。"""

    rise_threshold: float
    fall_threshold: float
    min_duration_sec: float
    merge_gap_sec: float


class StepPeakDetector:
    def __init__(self, cfg: PeakDetectorConfig):
        """ピーク検出設定を受け取り検出器を初期化する。"""
        self.cfg = cfg

    def detect(self, df: pd.DataFrame, parameter: str) -> list[StepPeak]:
        """対象チャネルの時系列からピーク区間リストを抽出する。

        前提条件:
                - `df` は少なくとも
                    `timestamp`（datetime-like）, `parameter`（str）, `value`（数値）列を持つ。
        - `timestamp` は `pd.Timestamp` / `datetime` 互換、`value` は `float` / `int` 互換を想定。
        - インデックスの単調増加は不要。内部で `timestamp` 昇順に並べ替えて処理する。

        戻り値:
        - `list[StepPeak]` を返す。
        - 各 `StepPeak` は `channel == parameter` を満たし、`start_ts <= end_ts`。
        - `mean/max/min/std` は対応区間の `value` から算出される。
        """
        sub = df[df["parameter"] == parameter].sort_values("timestamp")
        if sub.empty:
            return []

        ts = sub["timestamp"].to_numpy()
        values = sub["value"].to_numpy(dtype=float)

        segments = self._find_segments(ts, values)
        segments = self._merge_close_segments(ts, segments)

        peaks: list[StepPeak] = []
        for i0, i1 in segments:
            start_ts = ts[i0].astype("datetime64[ns]").astype(datetime)
            end_ts = ts[i1].astype("datetime64[ns]").astype(datetime)
            dur = (sub.iloc[i1]["timestamp"] - sub.iloc[i0]["timestamp"]).total_seconds()

            if dur < self.cfg.min_duration_sec:
                continue

            slice_vals = values[i0 : i1 + 1]
            peaks.append(
                StepPeak(
                    channel=parameter,
                    start_ts=start_ts,
                    end_ts=end_ts,
                    duration_sec=float(dur),
                    mean=float(np.mean(slice_vals)),
                    max=float(np.max(slice_vals)),
                    min=float(np.min(slice_vals)),
                    std=float(np.std(slice_vals, ddof=0)),
                )
            )
        return peaks

    def _find_segments(self, ts: np.ndarray, v: np.ndarray) -> list[tuple[int, int]]:
        """上昇/下降しきい値でピーク候補インデックス区間を抽出する。"""
        segs: list[tuple[int, int]] = []
        in_peak = False
        start = 0

        for i in range(len(v)):
            if not in_peak and v[i] >= self.cfg.rise_threshold:
                in_peak = True
                start = i
            elif in_peak and v[i] < self.cfg.fall_threshold:
                end = max(i - 1, start)
                segs.append((start, end))
                in_peak = False

        if in_peak:
            segs.append((start, len(v) - 1))

        return segs

    def _merge_close_segments(
        self, ts: np.ndarray, segs: list[tuple[int, int]]
    ) -> list[tuple[int, int]]:
        """近接するピーク候補区間を `merge_gap_sec` 条件で結合する。

        `segs` の各要素は `(start_index, end_index)`（いずれも両端含む）を表す。
        隣接区間 `(ps, pe)` と `(s, e)` の間隔は
        `gap_sec = (pd.Timestamp(ts[s]) - pd.Timestamp(ts[pe])).total_seconds()`
        で計算し、`gap_sec <= self.cfg.merge_gap_sec` のとき結合する（等号を含む）。
        つまり `if ts[next_start] - ts[prev_end] <= merge_gap_sec then merge` の挙動。
        """
        if not segs:
            return segs
        merged = [segs[0]]
        for s, e in segs[1:]:
            ps, pe = merged[-1]
            gap_sec = (pd.Timestamp(ts[s]) - pd.Timestamp(ts[pe])).total_seconds()
            if gap_sec <= self.cfg.merge_gap_sec:
                merged[-1] = (ps, e)
            else:
                merged.append((s, e))
        return merged
