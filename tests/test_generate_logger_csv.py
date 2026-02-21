from __future__ import annotations

from datetime import datetime
from pathlib import Path

import numpy as np

from portfolio_fdc.tools.sim_data_generator import generate_logger_csv as glc


def test_recipe_specs_shape() -> None:
    specs = glc.recipe_specs()

    assert len(specs) >= 2
    assert all(spec.recipe_id for spec in specs)
    assert all(len(spec.steps) >= 1 for spec in specs)
    assert all(step.duration_sec > 0 for spec in specs for step in spec.steps)


def test_build_process_signal_applies_all_steps() -> None:
    recipe = glc.RecipeSpec(
        recipe_id="R",
        steps=[
            glc.StepSpec(2, 1.0, 10.0, 50.0),
            glc.StepSpec(3, 2.0, 20.0, 60.0),
        ],
    )
    t = np.arange(20, dtype=int)

    v1, v2, v3 = glc.build_process_signal(start_idx=1, t=t, recipe=recipe)

    assert np.count_nonzero(v1 == 1.0) == 2
    assert np.count_nonzero(v2 == 10.0) == 2
    assert np.count_nonzero(v3 == 50.0) == 2

    assert np.count_nonzero(v1 == 2.0) == 3
    assert np.count_nonzero(v2 == 20.0) == 3
    assert np.count_nonzero(v3 == 60.0) == 3


def test_get_apc_sensor_for_tool() -> None:
    assert glc.get_apc_sensor_for_tool("TOOL_A") == "value03"
    assert glc.get_apc_sensor_for_tool("TOOL_B") == "value01"


def test_generate_base_signals_shape_and_columns() -> None:
    df = glc.generate_base_signals(
        start_ts=datetime.fromisoformat("2026-02-19T00:00:00"),
        seconds=10,
        seed=42,
    )
    assert list(df.columns) == ["timestamp", "value01", "value02", "value03"]
    assert len(df) == 10


def test_inject_anomaly_monitored_warn_updates_only_apc_column_for_tool_a() -> None:
    df = glc.generate_base_signals(
        start_ts=datetime.fromisoformat("2026-02-19T00:00:00"),
        seconds=200,
        seed=42,
    )
    out = glc.inject_anomaly_monitored(df, scenario="warn", tool_id="TOOL_A", seed=42)

    assert np.allclose(df["value01"].to_numpy(), out["value01"].to_numpy())
    assert np.allclose(df["value02"].to_numpy(), out["value02"].to_numpy())
    assert not np.allclose(df["value03"].to_numpy(), out["value03"].to_numpy())


def test_inject_anomaly_monitored_warn_updates_only_apc_column_for_tool_b() -> None:
    df = glc.generate_base_signals(
        start_ts=datetime.fromisoformat("2026-02-19T00:00:00"),
        seconds=200,
        seed=42,
        tool_id="TOOL_B",
    )
    out = glc.inject_anomaly_monitored(df, scenario="warn", tool_id="TOOL_B", seed=42)

    assert not np.allclose(df["value01"].to_numpy(), out["value01"].to_numpy())
    assert np.allclose(df["value02"].to_numpy(), out["value02"].to_numpy())
    assert np.allclose(df["value03"].to_numpy(), out["value03"].to_numpy())


def test_add_noise_is_reproducible_with_seed() -> None:
    base = np.array([1.0, 2.0, 3.0])
    r1 = np.random.default_rng(7)
    r2 = np.random.default_rng(7)

    out1 = glc.add_noise(base, sigma=0.03, rng=r1)
    out2 = glc.add_noise(base, sigma=0.03, rng=r2)

    assert out1.shape == base.shape
    assert np.allclose(out1, out2)


def test_write_logger_csv_new_file_layout_and_row_count(tmp_path: Path) -> None:
    out = tmp_path / "logger_raw.csv"

    glc.write_logger_csv(
        path=out,
        start_ts=datetime.fromisoformat("2026-02-19T00:00:00"),
        seconds=10,
        scenario="normal",
        seed=1,
        append=False,
    )

    lines = out.read_text(encoding="utf-8").splitlines()
    assert lines[0] == "LOGGER_CONFIG,Sampling=1s,Channels=3"
    assert lines[1] == "RANGE,value01,0,5"
    assert lines[2] == "RANGE,value02,0,50"
    assert lines[3] == "RANGE,value03,0,100"
    assert lines[4] == "DATA"
    assert lines[5] == "timestamp,value01,value02,value03"

    assert len(lines) == 6 + 10


def test_write_logger_csv_append_mode_has_single_header(tmp_path: Path) -> None:
    out = tmp_path / "logger_raw.csv"

    glc.write_logger_csv(
        path=out,
        start_ts=datetime.fromisoformat("2026-02-19T00:00:00"),
        seconds=3,
        scenario="normal",
        seed=10,
        append=False,
    )
    glc.write_logger_csv(
        path=out,
        start_ts=datetime.fromisoformat("2026-02-19T00:00:03"),
        seconds=2,
        scenario="normal",
        seed=11,
        append=True,
    )

    content = out.read_text(encoding="utf-8")
    assert content.count("LOGGER_CONFIG,Sampling=1s,Channels=3") == 1
    assert content.count("timestamp,value01,value02,value03") == 1

    lines = content.splitlines()
    assert len(lines) == 6 + 5


def test_main_parses_append_true(monkeypatch, tmp_path: Path) -> None:
    captured: dict[str, object] = {}

    def fake_write_logger_csv(
        path: Path,
        start_ts: datetime,
        seconds: int,
        scenario: str,
        seed: int,
        append: bool,
        tool_id: str,
    ) -> None:
        captured["path"] = path
        captured["append"] = append
        captured["seconds"] = seconds
        captured["scenario"] = scenario
        captured["tool_id"] = tool_id

    monkeypatch.setattr(glc, "write_logger_csv", fake_write_logger_csv)
    monkeypatch.setattr(
        "sys.argv",
        [
            "prog",
            "--out",
            str(tmp_path / "x.csv"),
            "--start",
            "2026-02-19T00:00:00",
            "--seconds",
            "12",
            "--scenario",
            "warn",
            "--tool-id",
            "TOOL_B",
            "--seed",
            "123",
            "--append",
            "true",
        ],
    )

    glc.main()

    assert captured["path"] == tmp_path / "x.csv"
    assert captured["append"] is True
    assert captured["seconds"] == 12
    assert captured["scenario"] == "warn"
    assert captured["tool_id"] == "TOOL_B"
