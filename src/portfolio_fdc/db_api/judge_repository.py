"""Judge result read API 用のリポジトリ層。"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from .datetime_util import to_utc_millis
from .db import MAIN_DB, _connect

_ALLOWED_LEVELS = {"OK", "WARN", "NG"}


@dataclass(frozen=True)
class JudgeResultsQueryCriteria:
    """`GET /judge/results` の検索条件を保持する DTO。"""

    chart_id: str | None = None
    process_id: str | None = None
    lot_id: str | None = None
    recipe_id: str | None = None
    level: str | None = None
    from_ts: str | None = None
    to_ts: str | None = None
    limit: int = 200
    offset: int = 0


@dataclass(frozen=True)
class JudgeResultView:
    """`GET /judge/results` レスポンス 1 件分の DTO。"""

    result_id: str
    chart_id: str | None
    process_id: str
    lot_id: str | None
    wafer_id: str | None
    recipe_id: str
    step_no: int | None
    feature_type: str | None
    feature_value: float | None
    level: str
    judged_at: str
    process_start_ts: str


class JudgeRepository:
    """判定結果一覧を取得するリポジトリ。"""

    _SELECT_SQL = """
        SELECT
            j.id,
            j.process_id,
            p.lot_id,
            p.wafer_id,
            j.recipe_id,
            j.status,
            j.judged_at,
            p.start_ts,
            j.message_json,
            CASE
                WHEN json_valid(j.message_json)
                THEN json_extract(j.message_json, '$.chart_id')
                ELSE NULL
            END AS extracted_chart_id
        FROM JudgementResults j
        INNER JOIN ProcessInfo p
            ON p.process_id = j.process_id
    """

    def find_results(self, criteria: JudgeResultsQueryCriteria) -> list[JudgeResultView]:
        """条件に一致する判定結果一覧を返す。"""
        sql = self._SELECT_SQL
        where_clauses: list[str] = []
        params: list[Any] = []

        self._append_filter_condition(
            criteria.process_id, "j.process_id = ?", where_clauses, params
        )
        self._append_filter_condition(criteria.lot_id, "p.lot_id = ?", where_clauses, params)
        self._append_filter_condition(criteria.recipe_id, "j.recipe_id = ?", where_clauses, params)
        self._append_filter_condition(
            criteria.level, "UPPER(j.status) = UPPER(?)", where_clauses, params
        )
        self._append_filter_condition(
            criteria.chart_id,
            (
                "CASE "
                "WHEN json_valid(j.message_json) = 1 "
                "AND ("
                "json_type(j.message_json, '$.chart_id') IN ('integer', 'real') "
                "OR ("
                "json_type(j.message_json, '$.chart_id') = 'text' "
                "AND json_extract(j.message_json, '$.chart_id') <> '' "
                "AND json_extract(j.message_json, '$.chart_id') NOT GLOB '*[^0-9]*'"
                ")"
                ") "
                "THEN 'CHART_' || CAST(json_extract(j.message_json, '$.chart_id') AS TEXT) "
                "WHEN json_valid(j.message_json) = 1 "
                "THEN json_extract(j.message_json, '$.chart_id') "
                "ELSE NULL END = ?"
            ),
            where_clauses,
            params,
        )
        self._append_filter_condition(
            criteria.from_ts,
            "julianday(j.judged_at) >= julianday(?)",
            where_clauses,
            params,
        )
        self._append_filter_condition(
            criteria.to_ts,
            "julianday(j.judged_at) <= julianday(?)",
            where_clauses,
            params,
        )

        if where_clauses:
            sql += " WHERE " + " AND ".join(where_clauses)

        sql += " ORDER BY julianday(j.judged_at) DESC, j.id DESC LIMIT ? OFFSET ?"
        params.extend([criteria.limit, criteria.offset])

        con = _connect(MAIN_DB)
        try:
            rows = con.execute(sql, tuple(params)).fetchall()
        finally:
            con.close()

        views: list[JudgeResultView] = []
        for row in rows:
            view = self._to_judge_result_view(row)
            if view is not None:
                views.append(view)
        return views

    @staticmethod
    def _append_filter_condition(
        value: Any,
        sql_fragment: str,
        where_clauses: list[str],
        params: list[Any],
    ) -> None:
        """値が存在する場合のみ WHERE 条件とパラメータを追加する。"""
        if value is None:
            return
        where_clauses.append(sql_fragment)
        params.append(value)

    @staticmethod
    def _to_judge_result_view(row: tuple[Any, ...]) -> JudgeResultView | None:
        """DB 行を `JudgeResultView` へ変換する。"""
        (
            result_pk,
            process_id,
            lot_id,
            wafer_id,
            recipe_id,
            status,
            judged_at,
            process_start_ts,
            message_json,
            extracted_chart_id,
        ) = row

        normalized_level = _normalize_level(status)
        if normalized_level is None:
            return None

        payload = _parse_message_json(message_json)
        chart_id = _extract_chart_id(payload, extracted_chart_id)
        step_no = _to_int_or_none(payload.get("step_no"))
        feature_type = _to_str_or_none(payload.get("feature_type"))
        feature_value = _to_float_or_none(payload.get("feature_value"))

        return JudgeResultView(
            result_id=f"JR_{int(result_pk)}",
            chart_id=chart_id,
            process_id=str(process_id),
            lot_id=_to_str_or_none(lot_id),
            wafer_id=_to_str_or_none(wafer_id),
            recipe_id=str(recipe_id),
            step_no=step_no,
            feature_type=feature_type,
            feature_value=feature_value,
            level=normalized_level,
            judged_at=to_utc_millis(str(judged_at)),
            process_start_ts=to_utc_millis(str(process_start_ts)),
        )


def _parse_message_json(raw: Any) -> dict[str, Any]:
    """message_json を辞書へ安全に変換する。"""
    if not isinstance(raw, str):
        return {}
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        return {}
    if not isinstance(parsed, dict):
        return {}
    return parsed


def _extract_chart_id(payload: dict[str, Any], extracted_chart_id: Any) -> str | None:
    """message_json 由来の chart_id を正規化して返す。"""
    candidate = payload.get("chart_id")
    if candidate is None:
        candidate = extracted_chart_id

    # Exclude bool (bool is a subclass of int in Python)
    if isinstance(candidate, bool):
        return None
    # Handle numeric types (int, float)
    if isinstance(candidate, (int, float)):
        return f"CHART_{int(candidate)}"
    if isinstance(candidate, str):
        if not candidate:
            return None
        if candidate.startswith("CHART_"):
            return candidate
        if candidate.isdigit():
            return f"CHART_{candidate}"
        return candidate
    return None


def _normalize_level(raw: Any) -> str | None:
    """status を API 契約上の level 値へ正規化する。"""
    if not isinstance(raw, str):
        return None
    normalized = raw.upper()
    if normalized not in _ALLOWED_LEVELS:
        return None
    return normalized


def _to_str_or_none(value: Any) -> str | None:
    """値を文字列または None へ変換する。"""
    if value is None:
        return None
    return str(value)


def _to_int_or_none(value: Any) -> int | None:
    """値を int または None へ変換する。変換失敗時は None を返す。"""
    if value is None:
        return None
    try:
        return int(value)
    except (ValueError, TypeError):
        return None


def _to_float_or_none(value: Any) -> float | None:
    """値を float または None へ変換する。変換失敗時は None を返す。"""
    if value is None:
        return None
    try:
        return float(value)
    except (ValueError, TypeError):
        return None
