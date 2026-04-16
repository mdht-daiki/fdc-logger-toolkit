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
class ActiveChartsQueryCriteria:
    """`GET /charts/active` の検索条件を保持する DTO。"""

    tool_id: str | None = None
    chamber_id: str | None = None
    recipe_id: str | None = None


@dataclass(frozen=True)
class ChartsHistoryQueryCriteria:
    """`GET /charts/history` の検索条件を保持する DTO。"""

    chart_pk: int | None = None
    chart_set_id: int | None = None
    change_source: str | None = None
    from_ts: str | None = None
    to_ts: str | None = None
    limit: int = 100
    offset: int = 0


@dataclass(frozen=True)
class ChartHistoryFilterKey:
    """ChartsHistory を一意に絞り込む複合キー。"""

    chart_set_id: int
    tool_id: str
    chamber_id: str
    recipe_id: str
    parameter: str
    step_no: int
    feature_type: str


@dataclass(frozen=True)
class ChartView:
    """`GET /charts` レスポンス 1 件分の DTO。

    `lcl/ucl` は後方互換のために維持している互換フィールドで、
    現在は `critical_lcl/critical_ucl` と同値を返す。
    新規利用側は warning/critical の明示フィールド参照を優先する。
    """

    chart_id: str
    chart_set_id: int
    tool_id: str
    chamber_id: str
    recipe_id: str
    parameter: str
    step_no: int
    feature_type: str
    lcl: float | None  # Compatibility alias of critical_lcl.
    ucl: float | None  # Compatibility alias of critical_ucl.
    warning_lcl: float | None
    warning_ucl: float | None
    critical_lcl: float | None
    critical_ucl: float | None
    updated_at: str
    version: int
    is_active: bool


@dataclass(frozen=True)
class ActiveChartView:
    """`GET /charts/active` の charts 配列 1 件分の DTO。"""

    chart_id: str
    parameter: str
    step_no: int
    feature_type: str
    warning_lcl: float | None
    warning_ucl: float | None
    critical_lcl: float | None
    critical_ucl: float | None


@dataclass(frozen=True)
class ActiveChartSetView:
    """`GET /charts/active` レスポンス data の DTO。"""

    active_chart_set_id: int | None
    activated_at: str | None
    charts: list[ActiveChartView]


@dataclass(frozen=True)
class ChartThresholdSnapshot:
    """履歴上のしきい値スナップショット。"""

    warning_lcl: float | None
    warning_ucl: float | None
    critical_lcl: float | None
    critical_ucl: float | None


@dataclass(frozen=True)
class ChartHistoryView:
    """`GET /charts/history` レスポンス 1 件分の DTO。"""

    history_id: str
    chart_id: str | None
    chart_set_id: int
    change_source: str | None
    change_reason: str | None
    before: ChartThresholdSnapshot
    after: ChartThresholdSnapshot
    changed_by: str | None
    changed_at: str


class ChartRepository:
    """ChartsV2 から chart 一覧を取得するリポジトリ。"""

    _ACTIVE_CHART_SET_SQL = """
        SELECT chart_set_id, updated_at
        FROM ActiveChartSet
        WHERE id = 1
    """

    _ACTIVE_CHARTS_SQL = """
        SELECT
            c.id,
            c.parameter,
            c.step_no,
            c.feature_type,
            c.warn_low,
            c.warn_high,
            c.crit_low,
            c.crit_high
        FROM ChartsV2 c
        WHERE c.chart_set_id = ?
    """

    _CHARTS_HISTORY_SQL = """
        SELECT
            h.id,
            h.chart_set_id,
            c.id,
            h.change_source,
            h.change_reason,
            h.old_warn_low,
            h.old_warn_high,
            h.old_crit_low,
            h.old_crit_high,
            h.new_warn_low,
            h.new_warn_high,
            h.new_crit_low,
            h.new_crit_high,
            h.changed_by,
            h.changed_at
        FROM ChartsHistory h
        LEFT JOIN ChartsV2 c
            ON c.chart_set_id = h.chart_set_id
           AND c.tool_id = h.tool_id
           AND c.chamber_id = h.chamber_id
           AND c.recipe_id = h.recipe_id
           AND c.parameter = h.parameter
           AND c.step_no = h.step_no
           AND c.feature_type = h.feature_type
    """

    _CHART_HISTORY_FILTER_KEY_SQL = """
        SELECT
            chart_set_id,
            tool_id,
            chamber_id,
            recipe_id,
            parameter,
            step_no,
            feature_type
        FROM ChartsV2
        WHERE id = ?
    """

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
                c.updated_at,
                CASE
                    WHEN EXISTS (
                        SELECT 1
                        FROM ActiveChartSet active
                        WHERE active.id = 1
                          AND active.chart_set_id = c.chart_set_id
                    ) THEN 1
                    ELSE 0
                END AS is_active
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
            fc.is_active,
            COALESCE(hv.version, 1) AS version
        FROM filtered_charts fc
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

        self._append_filter_condition(
            criteria.tool_id,
            "c.tool_id = ?",
            where_clauses,
            params,
        )
        self._append_filter_condition(
            criteria.chamber_id,
            "c.chamber_id = ?",
            where_clauses,
            params,
        )
        self._append_filter_condition(
            criteria.recipe_id,
            "c.recipe_id = ?",
            where_clauses,
            params,
        )
        self._append_filter_condition(
            criteria.parameter,
            "c.parameter = ?",
            where_clauses,
            params,
        )
        self._append_filter_condition(
            criteria.step_no,
            "c.step_no = ?",
            where_clauses,
            params,
        )
        self._append_filter_condition(
            criteria.feature_type,
            "c.feature_type = ?",
            where_clauses,
            params,
        )
        if criteria.active_only:
            where_clauses.append("is_active = 1")

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

    def find_active_chart_set(self, criteria: ActiveChartsQueryCriteria) -> ActiveChartSetView:
        """active chart set とその charts 一覧を返す。"""
        con = _connect(MAIN_DB)
        try:
            active_row = con.execute(self._ACTIVE_CHART_SET_SQL).fetchone()
            if active_row is None:
                return ActiveChartSetView(
                    active_chart_set_id=None,
                    activated_at=None,
                    charts=[],
                )

            active_chart_set_id = int(active_row[0])
            activated_at = _to_utc_millis(str(active_row[1]))

            sql = self._ACTIVE_CHARTS_SQL
            where_clauses: list[str] = []
            params: list[Any] = [active_chart_set_id]

            self._append_filter_condition(
                criteria.tool_id,
                "c.tool_id = ?",
                where_clauses,
                params,
            )
            self._append_filter_condition(
                criteria.chamber_id,
                "c.chamber_id = ?",
                where_clauses,
                params,
            )
            self._append_filter_condition(
                criteria.recipe_id,
                "c.recipe_id = ?",
                where_clauses,
                params,
            )

            if where_clauses:
                sql += " AND " + " AND ".join(where_clauses)

            sql += " ORDER BY c.parameter, c.step_no, c.feature_type"
            rows = con.execute(sql, tuple(params)).fetchall()
        finally:
            con.close()

        return ActiveChartSetView(
            active_chart_set_id=active_chart_set_id,
            activated_at=activated_at,
            charts=[self._to_active_chart_view(row) for row in rows],
        )

    def find_chart_history(self, criteria: ChartsHistoryQueryCriteria) -> list[ChartHistoryView]:
        """条件に一致する chart 閾値変更履歴を返す。"""
        sql = self._CHARTS_HISTORY_SQL
        where_clauses: list[str] = []
        params: list[Any] = []

        chart_filter_key = self._find_chart_history_filter_key(criteria.chart_pk)
        if criteria.chart_pk is not None and chart_filter_key is None:
            return []

        if chart_filter_key is not None:
            self._append_filter_condition(
                chart_filter_key.chart_set_id,
                "h.chart_set_id = ?",
                where_clauses,
                params,
            )
            self._append_filter_condition(
                chart_filter_key.tool_id,
                "h.tool_id = ?",
                where_clauses,
                params,
            )
            self._append_filter_condition(
                chart_filter_key.chamber_id,
                "h.chamber_id = ?",
                where_clauses,
                params,
            )
            self._append_filter_condition(
                chart_filter_key.recipe_id,
                "h.recipe_id = ?",
                where_clauses,
                params,
            )
            self._append_filter_condition(
                chart_filter_key.parameter,
                "h.parameter = ?",
                where_clauses,
                params,
            )
            self._append_filter_condition(
                chart_filter_key.step_no,
                "h.step_no = ?",
                where_clauses,
                params,
            )
            self._append_filter_condition(
                chart_filter_key.feature_type,
                "h.feature_type = ?",
                where_clauses,
                params,
            )
        self._append_filter_condition(
            criteria.chart_set_id,
            "h.chart_set_id = ?",
            where_clauses,
            params,
        )
        self._append_filter_condition(
            criteria.change_source,
            "h.change_source = ?",
            where_clauses,
            params,
        )
        self._append_filter_condition(
            criteria.from_ts,
            "datetime(h.changed_at) >= datetime(?)",
            where_clauses,
            params,
        )
        self._append_filter_condition(
            criteria.to_ts,
            "datetime(h.changed_at) <= datetime(?)",
            where_clauses,
            params,
        )

        if where_clauses:
            sql += " WHERE " + " AND ".join(where_clauses)

        sql += " ORDER BY datetime(h.changed_at) DESC, h.id DESC LIMIT ? OFFSET ?"
        params.extend([criteria.limit, criteria.offset])

        con = _connect(MAIN_DB)
        try:
            rows = con.execute(sql, tuple(params)).fetchall()
        finally:
            con.close()

        return [self._to_chart_history_view(row) for row in rows]

    def _find_chart_history_filter_key(self, chart_pk: int | None) -> ChartHistoryFilterKey | None:
        """chart_id から現在の ChartsV2 上の複合キーを解決する。"""
        if chart_pk is None:
            return None

        con = _connect(MAIN_DB)
        try:
            row = con.execute(self._CHART_HISTORY_FILTER_KEY_SQL, (chart_pk,)).fetchone()
        finally:
            con.close()

        if row is None:
            return None

        return ChartHistoryFilterKey(
            chart_set_id=int(row[0]),
            tool_id=str(row[1]),
            chamber_id=str(row[2]),
            recipe_id=str(row[3]),
            parameter=str(row[4]),
            step_no=int(row[5]),
            feature_type=str(row[6]),
        )

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

        # Keep lcl/ucl as backward-compatible aliases of critical bounds.
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

    @staticmethod
    def _to_active_chart_view(row: tuple[Any, ...]) -> ActiveChartView:
        """DB 行を `ActiveChartView` へ変換する。"""
        (
            chart_pk,
            parameter,
            step_no,
            feature_type,
            warn_low,
            warn_high,
            crit_low,
            crit_high,
        ) = row

        return ActiveChartView(
            chart_id=f"CHART_{int(chart_pk)}",
            parameter=str(parameter),
            step_no=int(step_no),
            feature_type=str(feature_type),
            warning_lcl=_to_float_or_none(warn_low),
            warning_ucl=_to_float_or_none(warn_high),
            critical_lcl=_to_float_or_none(crit_low),
            critical_ucl=_to_float_or_none(crit_high),
        )

    @staticmethod
    def _to_chart_history_view(row: tuple[Any, ...]) -> ChartHistoryView:
        """DB 行を `ChartHistoryView` へ変換する。"""
        (
            history_pk,
            chart_set_id,
            chart_pk,
            change_source,
            change_reason,
            old_warn_low,
            old_warn_high,
            old_crit_low,
            old_crit_high,
            new_warn_low,
            new_warn_high,
            new_crit_low,
            new_crit_high,
            changed_by,
            changed_at,
        ) = row

        return ChartHistoryView(
            history_id=f"HIS_{int(history_pk)}",
            chart_id=None if chart_pk is None else f"CHART_{int(chart_pk)}",
            chart_set_id=int(chart_set_id),
            change_source=None if change_source is None else str(change_source),
            change_reason=None if change_reason is None else str(change_reason),
            before=ChartThresholdSnapshot(
                warning_lcl=_to_float_or_none(old_warn_low),
                warning_ucl=_to_float_or_none(old_warn_high),
                critical_lcl=_to_float_or_none(old_crit_low),
                critical_ucl=_to_float_or_none(old_crit_high),
            ),
            after=ChartThresholdSnapshot(
                warning_lcl=_to_float_or_none(new_warn_low),
                warning_ucl=_to_float_or_none(new_warn_high),
                critical_lcl=_to_float_or_none(new_crit_low),
                critical_ucl=_to_float_or_none(new_crit_high),
            ),
            changed_by=None if changed_by is None else str(changed_by),
            changed_at=_to_utc_millis(str(changed_at)),
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
