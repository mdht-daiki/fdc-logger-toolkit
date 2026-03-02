from __future__ import annotations

from pydantic import BaseModel


class ProcessInfoIn(BaseModel):
    process_id: str
    tool_id: str
    chamber_id: str
    recipe_id: str
    start_ts: str
    end_ts: str
    raw_csv_path: str


class ProcessDeleteIn(BaseModel):
    process_id: str


class StepWindowIn(BaseModel):
    process_id: str
    step_no: int
    start_ts: str
    end_ts: str
    source_channel: str


class ParameterIn(BaseModel):
    process_id: str
    parameter: str
    step_no: int
    feature_type: str
    feature_value: float
