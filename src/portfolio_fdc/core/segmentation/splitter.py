from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from .models import StepBundle, StepPeak


@dataclass(frozen=True)
class SplitConfig:
    split_step_index: int
    main_ratio: float
    over_ratio: float


class StepSplitter:
    """
    Split one step by time_ratio into two steps.
    Recompute peak stats from the original timeseries df.
    """

    def __init__(self, cfg: SplitConfig):
        self.cfg = cfg

    def split_3_to_4(
        self,
        bundles3: list[StepBundle],
        df_process: pd.DataFrame,
    ) -> list[StepBundle]:
        if len(bundles3) != 3:
            return bundles3

        i = self.cfg.split_step_index - 1
        if i < 0 or i >= 3:
            return bundles3

        target = bundles3[i]
        if target.dc_bias is None:
            return bundles3

        s = target.dc_bias.start_ts
        e = target.dc_bias.end_ts
        total = (e - s).total_seconds()
        main_end = s + pd.Timedelta(seconds=total * self.cfg.main_ratio)

        # create 2 new dc_bias peaks based on recompute
        dc_main = self._recompute_peak(df_process, "dc_bias", s, main_end.to_pydatetime())
        dc_over = self._recompute_peak(df_process, "dc_bias", main_end.to_pydatetime(), e)

        # cl2_flow: simplest approach keeps same attachment (or recompute similarly)
        cl = target.cl2_flow

        out: list[StepBundle] = []
        step_no = 1
        for j, b in enumerate(bundles3):
            if j != i:
                out.append(StepBundle(step_no=step_no, dc_bias=b.dc_bias, cl2_flow=b.cl2_flow))
                step_no += 1
            else:
                out.append(StepBundle(step_no=step_no, dc_bias=dc_main, cl2_flow=cl))
                step_no += 1
                out.append(StepBundle(step_no=step_no, dc_bias=dc_over, cl2_flow=cl))
                step_no += 1
        return out

    def _recompute_peak(self, df: pd.DataFrame, parameter: str, start_ts, end_ts) -> StepPeak:
        sub = df[
            (df["parameter"] == parameter)
            & (df["timestamp"] >= start_ts)
            & (df["timestamp"] <= end_ts)
        ]
        vals = sub["value"].astype(float).to_numpy()
        dur = (end_ts - start_ts).total_seconds()
        return StepPeak(
            channel=parameter,
            start_ts=start_ts,
            end_ts=end_ts,
            duration_sec=dur,
            mean=float(vals.mean()) if len(vals) else 0.0,
            max=float(vals.max()) if len(vals) else 0.0,
            min=float(vals.min()) if len(vals) else 0.0,
            std=float(vals.std()) if len(vals) else 0.0,
        )
