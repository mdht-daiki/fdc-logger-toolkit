from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime


@dataclass(frozen=True)
class StepPeak:
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
    step_no: int
    dc_bias: StepPeak | None
    cl2_flow: StepPeak | None


@dataclass(frozen=True)
class ProcessSegment:
    process_id: str
    tool_id: str
    chamber_id: str

    start_ts: datetime
    end_ts: datetime

    recipe_id: str  # UNKNOWN allowed
