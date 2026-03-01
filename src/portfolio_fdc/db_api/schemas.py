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


class ChartIn(BaseModel):
    tool_id: str
    chamber_id: str
    recipe_id: str
    parameter: str
    step_no: int
    feature_type: str
    warn_low: float | None = None
    warn_high: float | None = None
    crit_low: float | None = None
    crit_high: float | None = None


class JudgementIn(BaseModel):
    process_id: str
    tool_id: str
    chamber_id: str
    recipe_id: str
    status: str
    judged_at: str
    message_json: str


class ChartSetIn(BaseModel):
    name: str
    note: str | None = None
    created_by: str | None = None


class ChartSetActivateIn(BaseModel):
    chart_set_id: int
    updated_by: str | None = None


class ChartUpsertIn(BaseModel):
    tool_id: str
    chamber_id: str
    recipe_id: str
    parameter: str
    step_no: int
    feature_type: str
    warn_low: float | None = None
    warn_high: float | None = None
    crit_low: float | None = None
    crit_high: float | None = None
    updated_by: str | None = None
    update_reason: str | None = None
    update_source: str | None = None
    chart_set_id: int | None = None
