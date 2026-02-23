from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class SensorMap:
    mapping: dict[tuple[str, str], str]  # (tool_id, sensor) -> parameter

    @staticmethod
    def from_csv(path: str) -> SensorMap:
        df = pd.read_csv(path)
        m: dict[tuple[str, str], str] = {}
        for _, r in df.iterrows():
            m[(str(r["tool_id"]), str(r["sensor"]))] = str(r["parameter"])
        return SensorMap(mapping=m)

    def resolve(self, tool_id: str, sensor: str) -> str | None:
        return self.mapping.get((tool_id, sensor))
