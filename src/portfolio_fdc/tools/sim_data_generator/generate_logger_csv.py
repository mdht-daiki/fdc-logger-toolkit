from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from portfolio_fdc.tools.sim_data_generator.anomaly import AnomalyConfig, inject_apc_anomaly


@dataclass(frozen=True)
class StepSpec:
    duration_sec: int
    v1: float
    v2: float
    v3: float


@dataclass(frozen=True)
class RecipeSpec:
    recipe_id: str
    steps: list[StepSpec]


def get_tool_parameter_to_sensor_map(tool_id: str) -> dict[str, str]:
    sensor_map_path = Path(__file__).resolve().parents[2] / "configs" / "sensor_map.csv"
    mapping = pd.read_csv(sensor_map_path)
    rows = mapping[mapping["tool_id"] == tool_id]
    if rows.empty:
        raise ValueError(f"sensor mapping not found for tool_id={tool_id}")

    out = {str(r["parameter"]): str(r["sensor"]) for _, r in rows.iterrows()}
    required = {"dc_bias", "cl2_flow", "apc_pressure"}
    if set(out.keys()) != required:
        raise ValueError(f"invalid mapping for tool_id={tool_id}, required={required}")
    if set(out.values()) != {"value01", "value02", "value03"}:
        raise ValueError(f"invalid sensor columns for tool_id={tool_id}")
    return out


def get_apc_sensor_for_tool(tool_id: str) -> str:
    return get_tool_parameter_to_sensor_map(tool_id)["apc_pressure"]


def recipe_specs() -> list[RecipeSpec]:
    """
    value01/value02/value03 are just channels.
    Meaning is defined by tool mapping in scrape config
    """
    return [
        RecipeSpec(
            "RECIPE_A",
            [
                StepSpec(20, 2.0, 15.0, 58.0),
                StepSpec(30, 2.7, 20.0, 58.0),
                StepSpec(25, 2.4, 18.0, 58.0),
                StepSpec(15, 1.8, 12.0, 58.0),
            ],
        ),
        RecipeSpec(
            "RECIPE_3STEP_C",
            [
                StepSpec(20, 1.7, 14.0, 58.0),
                StepSpec(30, 2.2, 18.0, 58.0),
                StepSpec(20, 1.5, 13.0, 58.0),
            ],
        ),
    ]


def build_process_signal(
    start_idx: int, t: np.ndarray, recipe: RecipeSpec
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    # idle baseline
    v1 = np.zeros_like(t, dtype=float)
    v2 = np.zeros_like(t, dtype=float)
    v3 = np.ones_like(t, dtype=float) * 28.0  # baseline pressure
    active = np.zeros_like(t, dtype=bool)

    cur = start_idx
    for s in recipe.steps:
        s_start = cur
        s_end = cur + s.duration_sec - 1
        v1[(t >= s_start) & (t <= s_end)] = s.v1
        v2[(t >= s_start) & (t <= s_end)] = s.v2
        v3[(t >= s_start) & (t <= s_end)] = s.v3
        active[(t >= s_start) & (t <= s_end)] = True
        cur = s_end + 1 + 3

    return v1, v2, v3, active


def add_noise(v: np.ndarray, sigma: float, rng: np.random.Generator) -> np.ndarray:
    return v + rng.normal(0.0, sigma, size=v.shape)


def generate_base_signals(
    start_ts: datetime,
    seconds: int,
    seed: int,
    tool_id: str = "TOOL_A",
) -> pd.DataFrame:
    if start_ts.tzinfo is not None:
        raise ValueError("start_ts must be a naive datetime (no timezone info)")
    rng = np.random.default_rng(seed)
    specs = recipe_specs()
    sensor_map = get_tool_parameter_to_sensor_map(tool_id)

    t = np.arange(seconds, dtype=int)
    dc_bias = np.zeros_like(t, dtype=float)
    cl2_flow = np.zeros_like(t, dtype=float)
    apc_pressure = np.ones_like(t, dtype=float) * 28.0
    process_active = np.zeros_like(t, dtype=bool)

    for i in range(0, seconds, 1800):
        recipe = specs[rng.integers(0, len(specs))]
        p_dc_bias, p_cl2_flow, p_apc_pressure, active = build_process_signal(i + 5, t, recipe)
        dc_bias = np.where(active, p_dc_bias, dc_bias)
        cl2_flow = np.where(active, p_cl2_flow, cl2_flow)
        apc_pressure = np.where(active, p_apc_pressure, apc_pressure)
        process_active = process_active | active

    channels = {
        "value01": np.zeros_like(t, dtype=float),
        "value02": np.zeros_like(t, dtype=float),
        "value03": np.zeros_like(t, dtype=float),
    }
    channels[sensor_map["dc_bias"]] = dc_bias
    channels[sensor_map["cl2_flow"]] = cl2_flow
    channels[sensor_map["apc_pressure"]] = apc_pressure

    for col in ("value01", "value02", "value03"):
        channels[col] = add_noise(channels[col], 0.03, rng)

    ts = [(start_ts + timedelta(seconds=int(i))).replace(tzinfo=None).isoformat() for i in t]
    return pd.DataFrame(
        {
            "timestamp": ts,
            "value01": channels["value01"],
            "value02": channels["value02"],
            "value03": channels["value03"],
            "process_active": process_active,
        }
    )


def inject_anomaly_monitored(
    df: pd.DataFrame,
    scenario: str,
    tool_id: str,
    seed: int,
) -> pd.DataFrame:
    if scenario == "normal":
        return df

    rng = np.random.default_rng(seed)
    pick = scenario
    if scenario == "mix":
        pick = str(rng.choice(["normal", "warn", "crit"], p=[0.7, 0.2, 0.1]))
        if pick == "normal":
            return df

    apc_col = get_apc_sensor_for_tool(tool_id)
    process_mask = df["process_active"].astype(bool)

    if pick == "warn":
        cfg = AnomalyConfig(mode="offset", level="warn", magnitude=0.35, seed=seed)
    elif pick == "crit":
        cfg = AnomalyConfig(mode="offset", level="crit", magnitude=-4.0, seed=seed)
    else:
        return df

    out = inject_apc_anomaly(df, apc_col=apc_col, cfg=cfg)
    out.loc[~process_mask, apc_col] = df.loc[~process_mask, apc_col]
    return out


def write_logger_csv(
    path: Path,
    start_ts: datetime,
    seconds: int,
    scenario: str,
    seed: int,
    append: bool,
    tool_id: str = "TOOL_A",
) -> None:
    df = generate_base_signals(start_ts=start_ts, seconds=seconds, seed=seed, tool_id=tool_id)
    df = inject_anomaly_monitored(df=df, scenario=scenario, tool_id=tool_id, seed=seed)
    df = df.drop(columns=["process_active"], errors="ignore")

    path.parent.mkdir(parents=True, exist_ok=True)
    if not append or not path.exists():
        # write header + DATA + column header
        with path.open("w", encoding="utf-8", newline="\n") as f:
            f.write("LOGGER_CONFIG,Sampling=1s,Channels=3\n")
            f.write("RANGE,value01,0,5\n")
            f.write("RANGE,value02,0,50\n")
            f.write("RANGE,value03,0,100\n")
            f.write("DATA\n")
        df.to_csv(path.as_posix(), mode="a", index=False, lineterminator="\n")
    else:
        # append only data rows (no header line)
        df.to_csv(path.as_posix(), mode="a", index=False, header=False, lineterminator="\n")
    print(f"OK: wrote {len(df)} rows -> {path.as_posix()} (append={append})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/raw/logger_raw.csv")
    ap.add_argument("--start", default="2026-02-19T00:00:00")
    ap.add_argument("--seconds", type=int, default=86400)
    ap.add_argument("--scenario", choices=["normal", "warn", "crit", "mix"], default="mix")
    ap.add_argument("--tool-id", default="TOOL_A")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--append", type=str, default="false")
    args = ap.parse_args()
    append = args.append.lower() == "true"
    write_logger_csv(
        Path(args.out),
        datetime.fromisoformat(args.start),
        args.seconds,
        args.scenario,
        args.seed,
        append,
        args.tool_id,
    )


if __name__ == "__main__":
    main()
