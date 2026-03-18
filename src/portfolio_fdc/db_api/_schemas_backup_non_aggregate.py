"""Internal backup-only schema definitions.

This module is not part of the runtime DB API contract.
Do not import from production code.
"""

from __future__ import annotations

from pydantic import BaseModel

__all__: list[str] = []

# このファイルはPRノイズを避けるための退避用です。
# aggregate で使わない schema 群を schemas.py から切り出して保存しています。
# この機能の実装時にコピーする。
#
# 戻し手順メモ（機能ごとに段階復元する）
# 1) charts 基本系を実装する時
#    - schemas.py に ChartIn を戻す
#
# 2) judgements を実装する時
#    - schemas.py に JudgementIn を戻す
#
# 3) chart_sets / charts_v2 / charts_history を実装する時
#    - schemas.py に ChartSetIn, ChartSetActivateIn, ChartUpsertIn を戻す


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
