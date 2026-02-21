from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class AnomalyConfig:
    mode: str  # offset/drift/spike/variance_increase/step_specific
    level: str  # "warn" or "crit"
    start_ts: pd.Timestamp | None = None
    end_ts: pd.Timestamp | None = None
    magnitude: float = 1.0
    n_spikes: int = 3
    spike_width_sec: int = 1
    step_no: int | None = None  # for step_specific
    seed: int = 42


def _time_mask(
    df: pd.DataFrame, start_ts: pd.Timestamp | None, end_ts: pd.Timestamp | None
) -> pd.Series:
    if start_ts is None and end_ts is None:
        return pd.Series(True, index=df.index)
    t = pd.to_datetime(df["timestamp"], errors="coerce")
    if start_ts is None:
        start_ts = t.min()
    if end_ts is None:
        end_ts = t.max()
    return (t >= start_ts) & (t <= end_ts)


def inject_apc_offset(df: pd.DataFrame, col: str, mask: pd.Series, delta: float) -> pd.DataFrame:
    out = df.copy()
    out.loc[mask, col] = pd.to_numeric(out.loc[mask, col], errors="coerce") + delta
    return out


def inject_apc_drift(
    df: pd.DataFrame, col: str, mask: pd.Series, total_delta: float
) -> pd.DataFrame:
    out = df.copy()
    idx = out.index[mask]
    if len(idx) <= 1:
        return out
    ramp = np.linspace(0.0, total_delta, num=len(idx))
    base = pd.to_numeric(out.loc[idx, col], errors="coerce").to_numpy()
    out.loc[idx, col] = base + ramp
    return out


def inject_apc_spikes(
    df: pd.DataFrame,
    col: str,
    mask: pd.Series,
    spike_delta: float,
    n_spikes: int,
    width_sec: int,
    seed: int,
) -> pd.DataFrame:
    out = df.copy()
    rng = np.random.default_rng(seed)
    idx = out.index[mask]
    if len(idx) == 0:
        return out
    # choose spike centers within masked region
    centers = rng.choice(idx, size=min(n_spikes, len(idx)), replace=False)
    # assume 1Hz sampling; width_sec expands to +/- width range
    for c in centers:
        c_pos = np.where(out.index == c)[0]
        if len(c_pos) == 0:
            continue
        pos = int(c_pos[0])
        lo = max(0, pos - width_sec)
        hi = min(len(out) - 1, pos + width_sec)
        window_idx = out.index[lo : hi + 1]
        base = pd.to_numeric(out.loc[window_idx, col], errors="coerce")
        out.loc[window_idx, col] = base + spike_delta
    return out


def inject_apc_variance_increase(
    df: pd.DataFrame, col: str, mask: pd.Series, add_sigma: float, seed: int
) -> pd.DataFrame:
    out = df.copy()
    rng = np.random.default_rng(seed)
    idx = out.index[mask]
    if len(idx) == 0:
        return out
    base = pd.to_numeric(out.loc[idx, col], errors="coerce").to_numpy()
    noise = rng.normal(0.0, add_sigma, size=len(idx))
    out.loc[idx, col] = base + noise
    return out


def inject_apc_step_specific(
    df: pd.DataFrame,
    col: str,
    step_windows: Sequence[tuple[int, pd.Timestamp, pd.Timestamp]],
    target_step_no: int,
    injector: Callable[[pd.DataFrame, pd.Series], pd.DataFrame],
) -> pd.DataFrame:
    out = df.copy()
    # build mask for the target step
    t = pd.to_datetime(out["timestamp"], errors="coerce")
    masks = []
    for no, s, e in step_windows:
        if int(no) == int(target_step_no):
            masks.append((t >= s) & (t <= e))
    if not masks:
        return out
    mask = masks[0]
    for m in masks[1:]:
        mask = mask | m
    return injector(out, mask)


def inject_apc_anomaly(
    df: pd.DataFrame,
    apc_col: str,
    cfg: AnomalyConfig,
    step_windows: Sequence[tuple[int, pd.Timestamp, pd.Timestamp]] | None = None,
) -> pd.DataFrame:
    # level scaling (tweak freely)
    # assume APC baseline around ~58 (example), change magnitude by level
    if cfg.level == "warn":
        k = 1.0
    elif cfg.level == "crit":
        k = 2.0
    else:
        k = 1.0
    mag = cfg.magnitude * k

    mask = _time_mask(df, cfg.start_ts, cfg.end_ts)

    if cfg.mode == "offset":
        # e.g., +2 (warn), +4 (crit)
        return inject_apc_offset(df, apc_col, mask, delta=mag)

    elif cfg.mode == "drift":
        # e.g., total +3 (warn), +6 (crit)
        return inject_apc_drift(df, apc_col, mask, total_delta=mag)

    elif cfg.mode == "spike":
        # e.g., spikes +5 (warn), +10 spikes (crit)
        return inject_apc_spikes(
            df,
            apc_col,
            mask,
            spike_delta=mag,
            n_spikes=cfg.n_spikes,
            width_sec=cfg.spike_width_sec,
            seed=cfg.seed,
        )

    elif cfg.mode == "variance_increase":
        # e.g., add sigma 0.5 (warn), 1.0 (crit)
        return inject_apc_variance_increase(df, apc_col, mask, add_sigma=mag, seed=cfg.seed)

    elif cfg.mode == "step_specific":
        # requires step_windows
        if step_windows is None or cfg.step_no is None:
            return df

        # choose one of the above injectors for step-only anomaly
        def _inj_offset(d: pd.DataFrame, m: pd.Series) -> pd.DataFrame:
            return inject_apc_offset(d, apc_col, m, delta=mag)

        return inject_apc_step_specific(df, apc_col, step_windows, cfg.step_no, _inj_offset)

    return df
