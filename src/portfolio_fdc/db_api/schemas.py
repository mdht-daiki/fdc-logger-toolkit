"""db_api 入力スキーマ（Pydantic モデル）定義。"""

from __future__ import annotations

import json
from datetime import datetime
from math import isfinite
from typing import Literal

from pydantic import BaseModel, Field, field_validator, model_validator


def validate_timestamp_range(start_ts: datetime, end_ts: datetime) -> None:
    """start/end のタイムゾーン形式と時系列順序を共通検証する。

    この関数が保証すること:

    - 両者が naive/aware の同一フォーマットであること（片方だけ aware は不可）
    - `end_ts >= start_ts` であること

    aware 同士であればオフセットが異なっていても許可される。
    Python の `datetime` 比較は UTC 正規化後に行われるため、
    例えば `2026-03-20T00:00:00Z` と `2026-03-20T09:00:01+09:00` は
    UTC 基準で前後比較される。
    """
    if (start_ts.tzinfo is None) != (end_ts.tzinfo is None):
        raise ValueError("start_ts and end_ts must use the same timezone format")
    if end_ts < start_ts:
        raise ValueError("end_ts must be greater than or equal to start_ts")


class ProcessInfoIn(BaseModel):
    """`/processes` に投入するプロセス情報。"""

    process_id: str
    tool_id: str
    chamber_id: str
    recipe_id: str
    start_ts: datetime
    end_ts: datetime
    raw_csv_path: str
    lot_id: str | None = None
    wafer_id: str | None = None

    @model_validator(mode="after")
    def validate_time_range(self) -> ProcessInfoIn:
        validate_timestamp_range(self.start_ts, self.end_ts)
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
        validate_timestamp_range(self.start_ts, self.end_ts)
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


class ChangeRequestIn(BaseModel):
    """`/governance/change-requests` POST 用の入力モデル。"""

    chart_id: int = Field(ge=1)
    proposed_by: str = Field(min_length=1, max_length=128)
    change_payload: str = Field(min_length=1)
    expected_version: int = Field(ge=1)
    idempotency_key: str = Field(min_length=1, max_length=128)

    @field_validator("change_payload")
    @classmethod
    def validate_change_payload_is_json(cls, value: str) -> str:
        try:
            json.loads(value)
        except json.JSONDecodeError as exc:
            raise ValueError("change_payload must be valid JSON") from exc
        return value


class ChangeRequestsQuery(BaseModel):
    """`/governance/change-requests` GET 用のクエリ入力モデル。"""

    status: Literal["pending", "approved", "applied", "apply_failed", "rejected"] | None = None
    chart_id: int | None = Field(default=None, ge=1)
    from_ts: datetime | None = None
    to_ts: datetime | None = None
    limit: int = Field(default=100, ge=1, le=500)
    offset: int = Field(default=0, ge=0)

    @model_validator(mode="after")
    def validate_query_time_range(self) -> ChangeRequestsQuery:
        if self.from_ts is not None and self.from_ts.tzinfo is None:
            raise ValueError("from_ts must be timezone-aware")
        if self.to_ts is not None and self.to_ts.tzinfo is None:
            raise ValueError("to_ts must be timezone-aware")
        if self.from_ts is not None and self.to_ts is not None:
            validate_timestamp_range(self.from_ts, self.to_ts)
        return self


class GovernanceAuditEventsQuery(BaseModel):
    """`/governance/audit-events` GET 用のクエリ入力モデル。"""

    event_type: str | None = Field(default=None, min_length=1, max_length=64)
    target_type: str | None = Field(default=None, min_length=1, max_length=64)
    target_id: int | None = Field(default=None, ge=1)
    from_ts: datetime | None = None
    to_ts: datetime | None = None
    limit: int = Field(default=100, ge=1, le=500)
    offset: int = Field(default=0, ge=0)


class ChangeRequestApproveIn(BaseModel):
    """`/governance/change-requests/{request_id}/approve` POST 用の入力モデル。"""

    approved_by: str = Field(min_length=1, max_length=128)
    approved_by_role: str = Field(min_length=1, max_length=64)
    comment: str | None = Field(default=None, max_length=1000)


class ChangeRequestApplyIn(BaseModel):
    """`/governance/change-requests/{request_id}/apply` POST 用の入力モデル。"""

    applied_by: str = Field(min_length=1, max_length=128)
    applied_by_role: str = Field(min_length=1, max_length=64)
    reason: str | None = Field(default=None, max_length=1000)


class EmergencyChangeIn(BaseModel):
    """`/governance/emergency-changes` POST 用の入力モデル。"""

    chart_id: int = Field(ge=1)
    changed_by: str = Field(min_length=1, max_length=128)
    changed_by_role: str = Field(min_length=1, max_length=64)
    reason: str | None = Field(default=None, max_length=1000)
    change_payload: str = Field(min_length=1)

    @field_validator("change_payload")
    @classmethod
    def validate_change_payload_is_json(cls, value: str) -> str:
        try:
            json.loads(value)
        except json.JSONDecodeError as exc:
            raise ValueError("change_payload must be valid JSON") from exc
        return value


class EmergencyChangeRatifyIn(BaseModel):
    """`/governance/emergency-changes/{request_id}/ratify` POST 用の入力モデル。"""

    ratified_by: str = Field(min_length=1, max_length=128)
    ratified_by_role: str = Field(min_length=1, max_length=64)
    ratification_comment: str | None = Field(default=None, max_length=1000)
    related_pr: str | None = Field(default=None, min_length=1, max_length=128)
