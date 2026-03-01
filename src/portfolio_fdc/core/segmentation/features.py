from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class FeatureRecord:
    process_id: str
    parameter: str
    step_no: int
    feature_type: str
    value: float


class FeatureExtractor:
    """
    Extract features for each (parameter, step) window.
    In the minimal version:
      - Use dc_bias step windows as the segmentation windows.
      - Extract features for dc_bias, cl2_flow, apc_pressure in each step window.
    """

    def extract(
        self,
        df_process: pd.DataFrame,
        process_id: str,
        step_windows: list[tuple[int, object, object]],  # (step_no, start_ts, end_ts)
        parameters: list[str],
        feature_types: list[str] | None = None,
    ) -> list[FeatureRecord]:
        if feature_types is None:
            feature_types = ["mean", "max", "min", "std"]
        out: list[FeatureRecord] = []
        for step_no, s, e in step_windows:
            for p in parameters:
                sub = df_process[
                    (df_process["parameter"] == p)
                    & (df_process["timestamp"] >= s)
                    & (df_process["timestamp"] <= e)
                ]
                vals = pd.to_numeric(sub["value"], errors="coerce").dropna().to_numpy(dtype=float)
                if len(vals) == 0:
                    vals = np.array([0.0], dtype=float)

                for ft in feature_types:
                    if ft == "mean":
                        v = float(np.mean(vals))
                    elif ft == "max":
                        v = float(np.max(vals))
                    elif ft == "min":
                        v = float(np.min(vals))
                    elif ft == "std":
                        v = float(np.std(vals))
                    else:
                        continue

                    out.append(
                        FeatureRecord(
                            process_id=process_id,
                            parameter=p,
                            step_no=step_no,
                            feature_type=ft,
                            value=v,
                        )
                    )
        return out
