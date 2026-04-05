from __future__ import annotations

from pathlib import Path

import yaml

from portfolio_fdc.tools.sim_data_generator.generate_logger_csv import recipe_specs


def test_charts_seed_matches_recipe_and_tool_contracts() -> None:
    root = Path(__file__).resolve().parents[1]
    charts_path = root / "src" / "portfolio_fdc" / "configs" / "charts_seed.yaml"
    aggregate_tools_path = root / "src" / "portfolio_fdc" / "configs" / "aggregate_tools.yaml"

    charts_payload = yaml.safe_load(charts_path.read_text(encoding="utf-8")) or {}
    tools_payload = yaml.safe_load(aggregate_tools_path.read_text(encoding="utf-8")) or {}

    charts = charts_payload["charts"]
    tools = tools_payload["tools"]
    recipe_step_counts = {recipe.recipe_id: len(recipe.steps) for recipe in recipe_specs()}

    assert charts

    seen_keys: set[tuple[str, str, str, str, int, str]] = set()
    expected_params = {"dc_bias", "cl2_flow", "apc_pressure"}

    for item in charts:
        key = (
            item["tool_id"],
            item["chamber_id"],
            item["recipe_id"],
            item["parameter"],
            int(item["step_no"]),
            item["feature_type"],
        )
        assert key not in seen_keys
        seen_keys.add(key)

        tool_cfg = tools[item["tool_id"]]
        allowed_params = set(tool_cfg.get("key_channels", {}).keys()) | set(
            tool_cfg.get("extra_channels", [])
        )

        assert item["chamber_id"] == tool_cfg["chamber_id"]
        assert item["recipe_id"] in recipe_step_counts
        assert item["parameter"] in allowed_params
        assert item["parameter"] in expected_params
        assert item["feature_type"] == "mean"
        assert 1 <= int(item["step_no"]) <= recipe_step_counts[item["recipe_id"]]
        assert item["crit_low"] <= item["warn_low"] <= item["warn_high"] <= item["crit_high"]

    for recipe_id, step_count in recipe_step_counts.items():
        for step_no in range(1, step_count + 1):
            for parameter in sorted(expected_params):
                assert (
                    "TOOL_A",
                    "CH1",
                    recipe_id,
                    parameter,
                    step_no,
                    "mean",
                ) in seen_keys
