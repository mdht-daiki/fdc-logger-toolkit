"""センサ名と論理パラメータ名の対応表を扱う。"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class SensorMap:
    """`(tool_id, sensor) -> parameter` のマッピングを保持する。"""

    mapping: dict[tuple[str, str], str]  # (tool_id, sensor) -> parameter

    @staticmethod
    def from_csv(path: str) -> SensorMap:
        """CSV からセンサマッピングを読み込み `SensorMap` を生成する。"""
        df = pd.read_csv(path)
        required = {"tool_id", "sensor", "parameter"}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"Sensor map CSV '{path}' is missing columns: {missing}")
        m: dict[tuple[str, str], str] = {}
        for _, r in df.iterrows():
            m[(str(r["tool_id"]), str(r["sensor"]))] = str(r["parameter"])
        return SensorMap(mapping=m)

    def resolve(self, tool_id: str, sensor: str) -> str | None:
        """指定ツール/センサに対応するパラメータ名を返す。"""
        return self.mapping.get((tool_id, sensor))
