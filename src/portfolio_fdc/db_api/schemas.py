from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, model_validator


def _parse_iso8601(ts: str) -> datetime:
    return datetime.fromisoformat(ts.replace("Z", "+00:00"))


class ProcessInfoIn(BaseModel):
    process_id: str
    tool_id: str
    chamber_id: str
    recipe_id: str
    start_ts: str
    end_ts: str
    raw_csv_path: str

    @model_validator(mode="after")
    def validate_time_range(self) -> ProcessInfoIn:
        start = _parse_iso8601(self.start_ts)
        end = _parse_iso8601(self.end_ts)
        if (start.tzinfo is None) != (end.tzinfo is None):
            raise ValueError("start_ts and end_ts must use the same timezone format")
        if end < start:
            raise ValueError("end_ts must be greater than or equal to start_ts")
        return self


class ProcessDeleteIn(BaseModel):
    process_id: str


class StepWindowIn(BaseModel):
    process_id: str
    step_no: int
    start_ts: str
    end_ts: str
    source_channel: str

    @model_validator(mode="after")
    def validate_time_range(self) -> StepWindowIn:
        start = _parse_iso8601(self.start_ts)
        end = _parse_iso8601(self.end_ts)
        if (start.tzinfo is None) != (end.tzinfo is None):
            raise ValueError("start_ts and end_ts must use the same timezone format")
        if end < start:
            raise ValueError("end_ts must be greater than or equal to start_ts")
        return self


class ParameterIn(BaseModel):
    process_id: str
    parameter: str
    step_no: int
    feature_type: str
    feature_value: float
