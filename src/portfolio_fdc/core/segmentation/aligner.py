"""複数チャネルのピークを時間重なりで整列する。"""

from __future__ import annotations

from .models import StepBundle, StepPeak


class MultiChannelPeakAligner:
    """
    Minimal rule:
      - Use dc_bias peaks as the main timeline.
      - For each dc_bias peak, attach the cl2_flow peak with the max overlap.
    """

    def align(self, dc_peaks: list[StepPeak], cl2_peaks: list[StepPeak]) -> list[StepBundle]:
        """`dc_bias` を基準に `cl2_flow` を対応づけて `StepBundle` を作る。"""
        bundles: list[StepBundle] = []
        for idx, dc in enumerate(dc_peaks, start=1):
            best = self._best_overlap(dc, cl2_peaks)
            bundles.append(StepBundle(step_no=idx, dc_bias=dc, cl2_flow=best))
        return bundles

    def _best_overlap(self, dc: StepPeak, cands: list[StepPeak]) -> StepPeak | None:
        """候補の中から重なり秒数が最大のピークを返す。"""
        best = None
        best_ol = 0.0
        for p in cands:
            ol = self._overlap_sec(dc, p)
            if ol > best_ol:
                best_ol = ol
                best = p
        return best

    @staticmethod
    def _overlap_sec(a: StepPeak, b: StepPeak) -> float:
        """2区間の時間重なり秒数を返す。"""
        s = max(a.start_ts, b.start_ts)
        e = min(a.end_ts, b.end_ts)
        return max(0.0, (e - s).total_seconds())
