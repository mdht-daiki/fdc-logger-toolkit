from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd


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
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    # idle baseline
    v1 = np.zeros_like(t, dtype=float)
    v2 = np.zeros_like(t, dtype=float)
    v3 = np.ones_like(t, dtype=float) * 28.0  # baseline pressure

    cur = start_idx
    for s in recipe.steps:
        s_start = cur
        s_end = cur + s.duration_sec - 1
        v1[(t >= s_start) & (t <= s_end)] = s.v1
        v2[(t >= s_start) & (t <= s_end)] = s.v2
        v3[(t >= s_start) & (t <= s_end)] = s.v3
        cur = s_end + 1 + 3

    return v1, v2, v3


def inject_scenario(
    recipe: RecipeSpec, v1: np.ndarray, v2: np.ndarray, rng: np.random.Generator, scenario: str
) -> None:
    if scenario == "normal":
        return
    pick = scenario
    if scenario == "mix":
        pick = rng.choice(["normal", "warn", "crit"], p=[0.7, 0.2, 0.1])
    if pick == "warn":
        v1[v1 > 0] += 0.35
    elif pick == "crit":
        v2[v2 > 0] -= 8.0


def add_noise(v: np.array, sigma: float, rng: np.random.Generator) -> np.ndarray:
    return v + rng.normal(0.0, sigma, size=v.shape)


def write_logger_csv(
    path: Path, start_ts: datetime, seconds: int, scenario: str, seed: int, append: bool
) -> None:
    rng = np.random.default_rng(seed)
    specs = recipe_specs()

    # time axis (1-second sampling)
    t = np.arange(seconds, dtype=int)

    # start with all idle
    v1 = np.zeros_like(t, dtype=float)
    v2 = np.zeros_like(t, dtype=float)
    v3 = np.ones_like(t, dtype=float) * 28.0

    # schedule processes every 30 minutes
    for i in range(0, seconds, 1800):
        recipe = rng.choice(specs)
        pv1, pv2, pv3 = build_process_signal(i + 5, t, recipe)

        # overlay: if already non-zero, keep latest (simple)
        v1 = np.where(pv1 > 0, pv1, v1)
        v2 = np.where(pv2 > 0, pv2, v2)
        v3 = np.where(pv3 != 28.0, pv3, v3)
        inject_scenario(recipe, v1, v2, rng, scenario)

    v1 = add_noise(v1, 0.03, rng)
    v2 = add_noise(v2, 0.03, rng)
    v3 = add_noise(v3, 0.03, rng)
    ts = [(start_ts + timedelta(seconds=int(i))).replace(tzinfo=None).isoformat() for i in t]
    df = pd.DataFrame({"timestamp": ts, "value01": v1, "value02": v2, "value03": v3})

    path.parent.mkdir(parents=True, exist_ok=True)
    if not append or not path.exists():
        # write header + DATA + column header
        with path.open("w", encoding="utf-8", newline="\n") as f:
            f.write("LOGGER_CONFIG,Sampling=1s,Channels=3\n")
            f.write("RANGE,value01,0,5\n")
            f.write("RANGE,value02,0,50\n")
            f.write("RANGE,value03,0,100\n")
            f.write("DATA\n")
        df.to_csv(path.as_posix(), mode="a", index=False)
    else:
        # append only data rows (no header line)
        df.to_csv(path.as_posix(), mode="a", index=False, header=False)
    print(f"OK: wrote {len(df)} rows -> {path.as_posix()} (append={append})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/raw/logger_raw.csv")
    ap.add_argument("--start", default="2026-02-19T00:00:00")
    ap.add_argument("--seconds", type=int, default=86400)
    ap.add_argument("--scenario", choices=["normal", "warn", "crit", "mix"], default="mix")
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
    )


if __name__ == "__main__":
    main()
