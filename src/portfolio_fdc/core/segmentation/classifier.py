"""ステップ特徴量からレシピIDを推定する分類ロジック。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .models import StepBundle


@dataclass(frozen=True)
class Range:
    """数値レンジ（下限・上限）を表す。"""

    low: float
    high: float

    def contains(self, x: float) -> bool:
        """値がレンジ内に含まれるか判定する。"""
        return self.low <= x <= self.high


class RecipeClassifier:
    def __init__(self, rules: dict[str, Any]):
        """YAML から読み込んだルール辞書で分類器を初期化する。"""
        self.rules = rules.get("recipes", {})

    def classify(self, bundles: list[StepBundle]) -> str:
        """ステップ束列に一致するレシピIDを返す。一致なしは `UNKNOWN`。"""
        for recipe_id, spec in self.rules.items():
            if self._match(recipe_id, spec, bundles):
                return recipe_id
        return "UNKNOWN"

    def _match(self, recipe_id: str, spec: dict[str, Any], bundles: list[StepBundle]) -> bool:
        """1レシピ定義と入力ステップ束列が一致するか判定する。"""
        steps = spec.get("steps", [])
        expanded_steps = steps
        if len(steps) != len(bundles):
            # 3-step recipe を 4-bundle に暫定一致させるケースのみ許可
            allow_presplit = len(steps) == 3 and len(bundles) == 4
            if not allow_presplit:
                return False
            expanded_steps = self._expand_steps_for_presplit(spec, steps)
            if len(expanded_steps) != len(bundles):
                return False

        # compare all bundled steps
        n = len(expanded_steps)
        for i in range(n):
            cond = expanded_steps[i]
            b = bundles[i]
            if not self._match_step(cond, b):
                return False
        return True

    def _expand_steps_for_presplit(
        self, spec: dict[str, Any], steps: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """3-step rule の split 指定がある場合、対象stepを2件に展開して返す。"""
        split = spec.get("split")
        if not isinstance(split, dict):
            return steps
        if split.get("method") != "time_ratio":
            return steps

        split_step = split.get("original_step")
        if not isinstance(split_step, int):
            return steps
        if split_step < 1 or split_step > len(steps):
            return steps

        main_ratio = split.get("main_ratio")
        over_ratio = split.get("over_ratio")
        if main_ratio is None or over_ratio is None:
            return steps

        idx = split_step - 1
        return [*steps[:idx], steps[idx], steps[idx], *steps[idx + 1 :]]

    def _match_step(self, cond: dict[str, Any], b: StepBundle) -> bool:
        """単一ステップ条件と `StepBundle` の一致を評価する。"""
        # Example keys: dc_bias_mean, cl2_flow_mean
        for key, rng in cond.items():
            low, high = float(rng[0]), float(rng[1])
            if key in ("dc_bias_mean", "dc_bias"):
                if b.dc_bias is None or not (low <= b.dc_bias.mean <= high):
                    return False
            elif key in ("cl2_flow_mean", "cl2_flow"):
                if b.cl2_flow is None or not (low <= b.cl2_flow.mean <= high):
                    return False
            else:
                # unknown condition key -> fail safe
                return False
        return True
