from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, model_validator


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
        start = datetime.fromisoformat(self.start_ts.replace("Z", "+00:00"))
        end = datetime.fromisoformat(self.end_ts.replace("Z", "+00:00"))
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
        start = datetime.fromisoformat(self.start_ts.replace("Z", "+00:00"))
        end = datetime.fromisoformat(self.end_ts.replace("Z", "+00:00"))
        if end < start:
            raise ValueError("end_ts must be greater than or equal to start_ts")
        return self


class ParameterIn(BaseModel):
    process_id: str
    parameter: str
    step_no: int
    feature_type: str
    feature_value: float
