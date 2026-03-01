from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .models import StepBundle


@dataclass(frozen=True)
class Range:
    low: float
    high: float

    def contains(self, x: float) -> bool:
        return self.low <= x <= self.high


class RecipeClassifier:
    def __init__(self, rules: dict[str, Any]):
        """
        rules format: loaded yaml dict (configs/recipe_rules.yaml)
        """
        self.rules = rules.get("recipes", {})

    def classify(self, bundles: list[StepBundle]) -> str:
        for recipe_id, spec in self.rules.items():
            if self._match(recipe_id, spec, bundles):
                return recipe_id
        return "UNKNOWN"

    def _match(self, recipe_id: str, spec: dict[str, Any], bundles: list[StepBundle]) -> bool:
        steps = spec.get("steps", [])
        if len(steps) != len(bundles):
            # 3-step recipe を 4-bundle に暫定一致させるケースのみ許可
            allow_presplit = len(steps) == 3 and len(bundles) == 4
            if not allow_presplit:
                return False

        # compare up to min length
        n = len(steps)
        for i in range(n):
            cond = steps[i]
            b = bundles[i]
            if not self._match_step(cond, b):
                return False
        return True

    def _match_step(self, cond: dict[str, Any], b: StepBundle) -> bool:
        # Example keys: dc_bias_mean, cl2_flow_mean
        for key, rng in cond.items():
            low, high = float(rng[0]), float(rng[1])
            if key == "dc_bias_mean":
                if b.dc_bias is None or not (low <= b.dc_bias.mean <= high):
                    return False
            elif key == "cl2_flow_mean":
                if b.cl2_flow is None or not (low <= b.cl2_flow.mean <= high):
                    return False
            else:
                # unknown condition key -> fail safe
                return False
        return True
