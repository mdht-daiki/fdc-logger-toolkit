"""Judge result read API 用のリポジトリ層。"""

from __future__ import annotations

import json
import logging
import math
import re
from dataclasses import dataclass
from typing import Any

from .datetime_util import to_utc_millis
from .db import MAIN_DB, _connect

_ALLOWED_LEVELS = {"OK", "WARN", "NG"}
_LOGGER = logging.getLogger(__name__)
_CHART_ID_PATTERN = re.compile(r"^CHART_[0-9]+$")


def _allowed_levels_sql() -> str:
    """許可された level 値の SQL IN 句用フラグメントを返す。"""
    return ", ".join(f"'{level}'" for level in sorted(_ALLOWED_LEVELS))


class JudgeDataCorruptionError(RuntimeError):
    """DB 行が存在するのに API モデルへ変換できない場合の例外。"""


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


@dataclass(frozen=True)
class JudgeResultDetailView:
    """`GET /judge/results/{result_id}` レスポンス DTO。"""

    result_id: str
    chart_id: str | None
    process_id: str
    lot_id: str | None
    wafer_id: str | None
    tool_id: str
    chamber_id: str
    recipe_id: str
    parameter: str | None
    step_no: int | None
    feature_type: str | None
    feature_value: float | None
    warning_lcl: float | None
    warning_ucl: float | None
    critical_lcl: float | None
    critical_ucl: float | None
    level: str
    judged_at: str
    process_start_ts: str
    stop_api_called: bool
    stop_api_status: str


class JudgeRepository:
    """判定結果一覧を取得するリポジトリ。"""

    _ALLOWED_LEVELS_SQL = _allowed_levels_sql()

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
        -- INNER JOIN enforces data integrity: JudgementResults and ProcessInfo are deleted
        -- together after their 1-year retention period. Orphaned JudgementResults
        -- (where ProcessInfo was deleted) should not occur by design.
        -- If this changes, reconsider LEFT JOIN + NULL handling.
        INNER JOIN ProcessInfo p
            ON p.process_id = j.process_id
    """

    _SELECT_DETAIL_SQL = """
        SELECT
            j.id,
            j.process_id,
            p.lot_id,
            p.wafer_id,
            j.tool_id,
            j.chamber_id,
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
                WHERE j.id = ?
                    AND UPPER(j.status) IN ({allowed_levels})
    """

    _SELECT_CHART_THRESHOLDS_SQL = """
        SELECT
            c.parameter,
            c.step_no,
            c.feature_type,
            c.warn_low,
            c.warn_high,
            c.crit_low,
            c.crit_high
        FROM ChartsV2 c
        WHERE c.id = ?
    """

    def find_results(self, criteria: JudgeResultsQueryCriteria) -> list[JudgeResultView]:
        """条件に一致する判定結果一覧を返す。"""
        sql = self._SELECT_SQL
        # Keep paging stable by excluding invalid rows at SQL level.
        where_clauses: list[str] = [
            f"UPPER(j.status) IN ({_allowed_levels_sql()})",
            "j.judged_at IS NOT NULL",
            "p.start_ts IS NOT NULL",
            "julianday(j.judged_at) IS NOT NULL",
            "julianday(p.start_ts) IS NOT NULL",
        ]
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
                "AND json_type(j.message_json, '$.chart_id') IN ('integer', 'real') "
                "AND json_extract(j.message_json, '$.chart_id') >= 0 "
                "THEN 'CHART_' || CAST(json_extract(j.message_json, '$.chart_id') AS INTEGER) "
                "WHEN json_valid(j.message_json) = 1 "
                "AND json_type(j.message_json, '$.chart_id') = 'text' "
                "AND json_extract(j.message_json, '$.chart_id') <> '' "
                "AND json_extract(j.message_json, '$.chart_id') NOT GLOB '*[^0-9]*' "
                "THEN 'CHART_' || "
                "CASE "
                "WHEN ltrim(json_extract(j.message_json, '$.chart_id'), '0') = '' THEN '0' "
                "ELSE ltrim(json_extract(j.message_json, '$.chart_id'), '0') "
                "END "
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

    def find_result_by_id(self, result_pk: int) -> JudgeResultDetailView | None:
        """`result_pk` に一致する判定結果詳細を返す。"""
        con = _connect(MAIN_DB)
        try:
            sql = self._SELECT_DETAIL_SQL.format(allowed_levels=self._ALLOWED_LEVELS_SQL)
            row = con.execute(sql, (result_pk,)).fetchone()
            if row is None:
                return None
            detail = self._to_judge_result_detail_view(con, row)
            if detail is None:
                raise JudgeDataCorruptionError(
                    f"Failed to convert judge result detail row for result_pk={result_pk}"
                )
            return detail
        finally:
            con.close()

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
        result_id = f"JR_{int(result_pk)}"

        try:
            judged_at_utc = to_utc_millis(str(judged_at))
        except ValueError:
            _LOGGER.warning(
                "Skipping judge result due to invalid judged_at: result_id=%s judged_at=%r",
                result_id,
                judged_at,
                exc_info=True,
            )
            return None

        try:
            process_start_ts_utc = to_utc_millis(str(process_start_ts))
        except ValueError:
            _LOGGER.warning(
                "Skipping judge result due to invalid process_start_ts: "
                "result_id=%s process_start_ts=%r",
                result_id,
                process_start_ts,
                exc_info=True,
            )
            return None

        payload = _parse_message_json(message_json)
        chart_id = _extract_chart_id(payload, extracted_chart_id)
        step_no = _to_int_or_none(payload.get("step_no"))
        feature_type = _to_str_value_or_none(payload.get("feature_type"))
        feature_value = _to_float_or_none(payload.get("feature_value"))

        return JudgeResultView(
            result_id=result_id,
            chart_id=chart_id,
            process_id=str(process_id),
            lot_id=_to_str_or_none(lot_id),
            wafer_id=_to_str_or_none(wafer_id),
            recipe_id=str(recipe_id),
            step_no=step_no,
            feature_type=feature_type,
            feature_value=feature_value,
            level=normalized_level,
            judged_at=judged_at_utc,
            process_start_ts=process_start_ts_utc,
        )

    def _to_judge_result_detail_view(
        self,
        con: Any,
        row: tuple[Any, ...],
    ) -> JudgeResultDetailView | None:
        """DB 行を `JudgeResultDetailView` へ変換する。"""
        (
            result_pk,
            process_id,
            lot_id,
            wafer_id,
            tool_id,
            chamber_id,
            recipe_id,
            status,
            judged_at,
            process_start_ts,
            message_json,
            extracted_chart_id,
        ) = row

        normalized_level = _normalize_level(status)
        result_id = f"JR_{int(result_pk)}"

        try:
            judged_at_utc = to_utc_millis(str(judged_at))
        except ValueError:
            _LOGGER.warning(
                "Skipping judge result detail due to invalid judged_at: result_id=%s judged_at=%r",
                result_id,
                judged_at,
                exc_info=True,
            )
            return None

        try:
            process_start_ts_utc = to_utc_millis(str(process_start_ts))
        except ValueError:
            _LOGGER.warning(
                "Skipping judge result detail due to invalid process_start_ts: "
                "result_id=%s process_start_ts=%r",
                result_id,
                process_start_ts,
                exc_info=True,
            )
            return None

        payload = _parse_message_json(message_json)
        chart_id = _extract_chart_id(payload, extracted_chart_id)

        parameter = _to_str_value_or_none(payload.get("parameter"))
        step_no = _to_int_or_none(payload.get("step_no"))
        feature_type = _to_str_value_or_none(payload.get("feature_type"))
        feature_value = _to_float_or_none(payload.get("feature_value"))

        warning_lcl: float | None = None
        warning_ucl: float | None = None
        critical_lcl: float | None = None
        critical_ucl: float | None = None

        chart_pk = _chart_pk_from_chart_id(chart_id)
        if chart_pk is not None:
            chart_row = con.execute(self._SELECT_CHART_THRESHOLDS_SQL, (chart_pk,)).fetchone()
            if chart_row is not None:
                (
                    chart_parameter,
                    chart_step_no,
                    chart_feature_type,
                    warning_lcl,
                    warning_ucl,
                    critical_lcl,
                    critical_ucl,
                ) = chart_row
                if parameter is None:
                    parameter = _to_str_or_none(chart_parameter)
                if step_no is None:
                    step_no = _to_int_or_none(chart_step_no)
                if feature_type is None:
                    feature_type = _to_str_or_none(chart_feature_type)

        stop_api_called = _to_bool_or_default(payload.get("stop_api_called"), default=False)
        stop_api_status = _to_stop_api_status_or_default(
            payload.get("stop_api_status"),
            default="NOT_CALLED",
        )

        return JudgeResultDetailView(
            result_id=result_id,
            chart_id=chart_id,
            process_id=str(process_id),
            lot_id=_to_str_or_none(lot_id),
            wafer_id=_to_str_or_none(wafer_id),
            tool_id=str(tool_id),
            chamber_id=str(chamber_id),
            recipe_id=str(recipe_id),
            parameter=parameter,
            step_no=step_no,
            feature_type=feature_type,
            feature_value=feature_value,
            warning_lcl=_to_float_or_none(warning_lcl),
            warning_ucl=_to_float_or_none(warning_ucl),
            critical_lcl=_to_float_or_none(critical_lcl),
            critical_ucl=_to_float_or_none(critical_ucl),
            level=normalized_level,
            judged_at=judged_at_utc,
            process_start_ts=process_start_ts_utc,
            stop_api_called=stop_api_called,
            stop_api_status=stop_api_status,
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
        # Reject non-finite floats (NaN, inf, -inf)
        if not math.isfinite(candidate):
            return None
        if candidate < 0:
            return None
        try:
            return f"CHART_{int(candidate)}"
        except (ValueError, OverflowError):
            return None
    if isinstance(candidate, str):
        if not candidate:
            return None
        if candidate.startswith("CHART_"):
            if _CHART_ID_PATTERN.fullmatch(candidate) is None:
                return None
            return candidate
        if candidate.isascii() and candidate.isdigit():
            # Mirror SQL-side numeric normalization (e.g. "001" -> "CHART_1").
            try:
                numeric_candidate = int(candidate)
            except (ValueError, OverflowError):
                return None
            return f"CHART_{numeric_candidate}"
        return None
    return None


def _normalize_level(raw: Any) -> str:
    """status を API 契約上の level 値へ正規化する。"""
    if isinstance(raw, str):
        return raw.upper()
    return str(raw).upper()


def _to_str_or_none(value: Any) -> str | None:
    """値を文字列または None へ変換する。"""
    if value is None:
        return None
    return str(value)


def _to_str_value_or_none(value: Any) -> str | None:
    """値が文字列の場合のみそのまま返し、それ以外は None を返す。"""
    if isinstance(value, str):
        return value
    return None


def _to_int_or_none(value: Any) -> int | None:
    """値を int または None へ変換する。変換失敗時は None を返す。"""
    if value is None:
        return None
    if isinstance(value, bool):
        return None
    if isinstance(value, float):
        if not math.isfinite(value):
            return None
        if not value.is_integer():
            return None
        return int(value)
    try:
        return int(value)
    except (ValueError, TypeError, OverflowError):
        return None


def _to_float_or_none(value: Any) -> float | None:
    """値を float または None へ変換する。変換失敗時は None を返す。"""
    if value is None:
        return None
    if isinstance(value, bool):
        return None
    try:
        result = float(value)
        # Reject non-finite floats (NaN, inf, -inf)
        if not math.isfinite(result):
            return None
        return result
    except (ValueError, TypeError, OverflowError):
        return None


def _chart_pk_from_chart_id(chart_id: str | None) -> int | None:
    """`CHART_<id>` を chart PK へ変換する。"""
    if chart_id is None or _CHART_ID_PATTERN.fullmatch(chart_id) is None:
        return None
    try:
        chart_pk = int(chart_id.split("_", maxsplit=1)[1])
    except (ValueError, IndexError, OverflowError):
        return None
    if chart_pk < 0:
        return None
    return chart_pk


def _to_bool_or_default(value: Any, *, default: bool) -> bool:
    """bool 値として安全に解釈できる場合のみその値を返す。"""
    if isinstance(value, bool):
        return value
    return default


def _to_stop_api_status_or_default(value: Any, *, default: str) -> str:
    """stop_api_status を文字列として返し、未指定/不正時は既定値へフォールバックする。"""
    if isinstance(value, str):
        normalized = value.strip()
        if normalized:
            return normalized
    return default
