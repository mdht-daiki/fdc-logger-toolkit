"""Chart read API 用のリポジトリ層。"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

from .db import MAIN_DB, _connect


@dataclass(frozen=True)
class ChartsQueryCriteria:
    """`GET /charts` の検索条件を保持する DTO。"""

    tool_id: str | None = None
    chamber_id: str | None = None
    recipe_id: str | None = None
    parameter: str | None = None
    step_no: int | None = None
    feature_type: str | None = None
    active_only: bool = False


@dataclass(frozen=True)
class ChartView:
    """`GET /charts` レスポンス 1 件分の DTO。"""

    chart_id: str
    chart_set_id: int
    tool_id: str
    chamber_id: str
    recipe_id: str
    parameter: str
    step_no: int
    feature_type: str
    lcl: float | None
    ucl: float | None
    warning_lcl: float | None
    warning_ucl: float | None
    critical_lcl: float | None
    critical_ucl: float | None
    updated_at: str
    version: int
    is_active: bool


class ChartRepository:
    """ChartsV2 から chart 一覧を取得するリポジトリ。"""

    _FILTERED_CHARTS_CTE = """
        WITH filtered_charts AS (
            SELECT
                c.id,
                c.chart_set_id,
                c.tool_id,
                c.chamber_id,
                c.recipe_id,
                c.parameter,
                c.step_no,
                c.feature_type,
                c.warn_low,
                c.warn_high,
                c.crit_low,
                c.crit_high,
                c.updated_at
            FROM ChartsV2 c
    """

    _HISTORY_VERSIONS_CTE = """
        ),
        history_versions AS (
            SELECT
                h.chart_set_id,
                h.tool_id,
                h.chamber_id,
                h.recipe_id,
                h.parameter,
                h.step_no,
                h.feature_type,
                COUNT(*) + 1 AS version
            FROM ChartsHistory h
            INNER JOIN filtered_charts fc
                ON h.chart_set_id = fc.chart_set_id
               AND h.tool_id = fc.tool_id
               AND h.chamber_id = fc.chamber_id
               AND h.recipe_id = fc.recipe_id
               AND h.parameter = fc.parameter
               AND h.step_no = fc.step_no
               AND h.feature_type = fc.feature_type
            GROUP BY
                h.chart_set_id,
                h.tool_id,
                h.chamber_id,
                h.recipe_id,
                h.parameter,
                h.step_no,
                h.feature_type
        )
    """

    _SELECT_SQL = """
        SELECT
            fc.id,
            fc.chart_set_id,
            fc.tool_id,
            fc.chamber_id,
            fc.recipe_id,
            fc.parameter,
            fc.step_no,
            fc.feature_type,
            fc.warn_low,
            fc.warn_high,
            fc.crit_low,
            fc.crit_high,
            fc.updated_at,
            CASE WHEN a.chart_set_id = fc.chart_set_id THEN 1 ELSE 0 END AS is_active,
            COALESCE(hv.version, 1) AS version
        FROM filtered_charts fc
        LEFT JOIN ActiveChartSet a ON a.id = 1
        LEFT JOIN history_versions hv
            ON hv.chart_set_id = fc.chart_set_id
           AND hv.tool_id = fc.tool_id
           AND hv.chamber_id = fc.chamber_id
           AND hv.recipe_id = fc.recipe_id
           AND hv.parameter = fc.parameter
           AND hv.step_no = fc.step_no
           AND hv.feature_type = fc.feature_type
    """

    def find_charts(self, criteria: ChartsQueryCriteria) -> list[ChartView]:
        """条件に一致する chart 一覧を返す。"""
        sql = self._FILTERED_CHARTS_CTE
        where_clauses: list[str] = []
        params: list[Any] = []

        if criteria.tool_id is not None:
            where_clauses.append("c.tool_id = ?")
            params.append(criteria.tool_id)
        if criteria.chamber_id is not None:
            where_clauses.append("c.chamber_id = ?")
            params.append(criteria.chamber_id)
        if criteria.recipe_id is not None:
            where_clauses.append("c.recipe_id = ?")
            params.append(criteria.recipe_id)
        if criteria.parameter is not None:
            where_clauses.append("c.parameter = ?")
            params.append(criteria.parameter)
        if criteria.step_no is not None:
            where_clauses.append("c.step_no = ?")
            params.append(criteria.step_no)
        if criteria.feature_type is not None:
            where_clauses.append("c.feature_type = ?")
            params.append(criteria.feature_type)
        if criteria.active_only:
            where_clauses.append(
                "EXISTS ("
                "SELECT 1 FROM ActiveChartSet active "
                "WHERE active.id = 1 AND active.chart_set_id = c.chart_set_id"
                ")"
            )

        if where_clauses:
            sql += " WHERE " + " AND ".join(where_clauses)

        sql += self._HISTORY_VERSIONS_CTE
        sql += self._SELECT_SQL
        sql += (
            " ORDER BY fc.chart_set_id, fc.tool_id, fc.chamber_id, fc.recipe_id,"
            " fc.parameter, fc.step_no, fc.feature_type"
        )

        con = _connect(MAIN_DB)
        try:
            rows = con.execute(sql, tuple(params)).fetchall()
        finally:
            con.close()

        return [self._to_chart_view(row) for row in rows]

    @staticmethod
    def _to_chart_view(row: tuple[Any, ...]) -> ChartView:
        """DB 行を `ChartView` へ変換する。"""
        (
            chart_pk,
            chart_set_id,
            tool_id,
            chamber_id,
            recipe_id,
            parameter,
            step_no,
            feature_type,
            warn_low,
            warn_high,
            crit_low,
            crit_high,
            updated_at,
            is_active,
            version,
        ) = row

        return ChartView(
            chart_id=f"CHART_{int(chart_pk)}",
            chart_set_id=int(chart_set_id),
            tool_id=str(tool_id),
            chamber_id=str(chamber_id),
            recipe_id=str(recipe_id),
            parameter=str(parameter),
            step_no=int(step_no),
            feature_type=str(feature_type),
            lcl=_to_float_or_none(crit_low),
            ucl=_to_float_or_none(crit_high),
            warning_lcl=_to_float_or_none(warn_low),
            warning_ucl=_to_float_or_none(warn_high),
            critical_lcl=_to_float_or_none(crit_low),
            critical_ucl=_to_float_or_none(crit_high),
            updated_at=_to_utc_millis(str(updated_at)),
            version=int(version),
            is_active=bool(is_active),
        )


def _to_float_or_none(value: Any) -> float | None:
    """SQLite 値を float または None へ変換する。"""
    if value is None:
        return None
    return float(value)


def _to_utc_millis(raw: str) -> str:
    """任意の ISO 8601 文字列を UTC ミリ秒固定の文字列へ正規化する。"""
    normalized = raw.replace("Z", "+00:00")
    dt = datetime.fromisoformat(normalized)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=UTC)
    else:
        dt = dt.astimezone(UTC)
    return dt.strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"
