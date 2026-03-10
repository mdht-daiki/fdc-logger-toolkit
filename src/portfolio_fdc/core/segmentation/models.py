"""セグメンテーションで扱うドメインモデル定義。"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime


@dataclass(frozen=True)
class StepPeak:
    """1つのチャネルで検出されたステップピーク区間。"""

    channel: str
    start_ts: datetime
    end_ts: datetime
    duration_sec: float

    mean: float
    max: float
    min: float
    std: float


@dataclass(frozen=True)
class StepBundle:
    """同一ステップ番号に対応づけた複数チャネルのピーク束。"""

    step_no: int
    dc_bias: StepPeak | None
    cl2_flow: StepPeak | None


@dataclass(frozen=True)
class ProcessSegment:
    """切り出し済みプロセス区間のメタ情報。"""

    process_id: str
    tool_id: str
    chamber_id: str

    start_ts: datetime
    end_ts: datetime

    recipe_id: str  # UNKNOWN allowed
