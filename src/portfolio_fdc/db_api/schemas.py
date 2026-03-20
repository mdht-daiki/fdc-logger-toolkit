"""db_api 入力スキーマ（Pydantic モデル）定義。"""

from __future__ import annotations

from datetime import datetime
from math import isfinite

from pydantic import BaseModel, Field, field_validator, model_validator


class ProcessInfoIn(BaseModel):
    """`/processes` に投入するプロセス情報。"""

    process_id: str
    tool_id: str
    chamber_id: str
    recipe_id: str
    start_ts: datetime
    end_ts: datetime
    raw_csv_path: str

    @model_validator(mode="after")
    def validate_time_range(self) -> ProcessInfoIn:
        if (self.start_ts.tzinfo is None) != (self.end_ts.tzinfo is None):
            raise ValueError("start_ts and end_ts must use the same timezone format")
        if self.end_ts < self.start_ts:
            raise ValueError("end_ts must be greater than or equal to start_ts")
        return self


class ProcessDeleteIn(BaseModel):
    """削除対象の `process_id` を受け取る入力モデル。"""

    process_id: str


class StepWindowIn(BaseModel):
    """`/step_windows/bulk` 用のステップ区間入力。"""

    process_id: str
    step_no: int
    start_ts: datetime
    end_ts: datetime
    source_channel: str

    @model_validator(mode="after")
    def validate_time_range(self) -> StepWindowIn:
        if (self.start_ts.tzinfo is None) != (self.end_ts.tzinfo is None):
            raise ValueError("start_ts and end_ts must use the same timezone format")
        if self.end_ts < self.start_ts:
            raise ValueError("end_ts must be greater than or equal to start_ts")
        return self


class ParameterIn(BaseModel):
    """`/parameters/bulk` 用の特徴量入力。"""

    process_id: str
    parameter: str
    step_no: int = Field(ge=0)
    feature_type: str
    feature_value: float

    @field_validator("feature_value")
    @classmethod
    def validate_feature_value_is_finite(cls, value: float) -> float:
        if not isfinite(value):
            raise ValueError("feature_value must be finite")
        return value


class AggregateWriteIn(BaseModel):
    """`/aggregate/write` 用の原子的書き込み入力。"""

    process: ProcessInfoIn
    step_windows: list[StepWindowIn]
    parameters: list[ParameterIn]

    @model_validator(mode="after")
    def validate_process_id_consistency(self) -> AggregateWriteIn:
        pid = self.process.process_id
        if any(item.process_id != pid for item in self.step_windows):
            raise ValueError("step_windows process_id must match process.process_id")
        if any(item.process_id != pid for item in self.parameters):
            raise ValueError("parameters process_id must match process.process_id")
        return self
