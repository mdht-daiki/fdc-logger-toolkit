from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import asdict
from datetime import datetime

from fastapi import APIRouter, Depends, HTTPException, Path, Query, Request

from ..api_common import (
    CHART_ID_PATTERN,
    CHARTS_FILTER_MAX_LENGTH,
    CHARTS_FILTER_PATTERN,
    JUDGE_LEVEL_PATTERN,
    RESULT_ID_PATTERN,
    normalize_query_datetime,
    not_found_error_response,
    parse_chart_pk,
    parse_result_pk,
    raise_api_error,
    validate_query_datetime_range,
)
from ..chart_repository import (
    ActiveChartsQueryCriteria,
    ChartRepository,
    ChartsHistoryQueryCriteria,
    ChartsQueryCriteria,
)
from ..db import MAIN_DB, _connect_readonly
from ..judge_repository import JudgeDataCorruptionError, JudgeRepository, JudgeResultsQueryCriteria
from ..schemas import ChangeRequestsQuery, GovernanceAuditEventsQuery
from ..task_runner import DBTaskRunner

logger = logging.getLogger("portfolio_fdc.db_api.app")


class QueryRouter:
    """参照系エンドポイントをまとめるルータークラス。"""

    def __init__(
        self,
        *,
        chart_repository: ChartRepository,
        judge_repository: JudgeRepository,
        governance_change_request_repository,
        get_runner: Callable[[Request], DBTaskRunner],
        build_waveform_preview: Callable[[str, int], dict[str, object]],
    ) -> None:
        self._chart_repository = chart_repository
        self._judge_repository = judge_repository
        self._governance_change_request_repository = governance_change_request_repository
        self._get_runner = get_runner
        self._build_waveform_preview = build_waveform_preview
        self.router = APIRouter()
        self._register_routes()

    def _register_routes(self) -> None:
        @self.router.get("/charts")
        def get_charts(
            tool_id: str | None = Query(
                default=None,
                min_length=1,
                max_length=CHARTS_FILTER_MAX_LENGTH,
                pattern=CHARTS_FILTER_PATTERN,
            ),
            chamber_id: str | None = Query(
                default=None,
                min_length=1,
                max_length=CHARTS_FILTER_MAX_LENGTH,
                pattern=CHARTS_FILTER_PATTERN,
            ),
            recipe_id: str | None = Query(
                default=None,
                min_length=1,
                max_length=CHARTS_FILTER_MAX_LENGTH,
                pattern=CHARTS_FILTER_PATTERN,
            ),
            parameter: str | None = Query(
                default=None,
                min_length=1,
                max_length=CHARTS_FILTER_MAX_LENGTH,
                pattern=CHARTS_FILTER_PATTERN,
            ),
            step_no: int | None = Query(default=None, ge=0),
            feature_type: str | None = Query(
                default=None,
                min_length=1,
                max_length=CHARTS_FILTER_MAX_LENGTH,
                pattern=CHARTS_FILTER_PATTERN,
            ),
            active_only: bool = False,
        ):
            criteria = ChartsQueryCriteria(
                tool_id=tool_id,
                chamber_id=chamber_id,
                recipe_id=recipe_id,
                parameter=parameter,
                step_no=step_no,
                feature_type=feature_type,
                active_only=active_only,
            )
            try:
                rows = self._chart_repository.find_charts(criteria)
                return {"ok": True, "data": [asdict(row) for row in rows]}
            except Exception as e:
                raise_api_error(operation="GET /charts", error=e)

        @self.router.get("/charts/active")
        def get_active_charts(
            tool_id: str | None = Query(
                default=None,
                min_length=1,
                max_length=CHARTS_FILTER_MAX_LENGTH,
                pattern=CHARTS_FILTER_PATTERN,
            ),
            chamber_id: str | None = Query(
                default=None,
                min_length=1,
                max_length=CHARTS_FILTER_MAX_LENGTH,
                pattern=CHARTS_FILTER_PATTERN,
            ),
            recipe_id: str | None = Query(
                default=None,
                min_length=1,
                max_length=CHARTS_FILTER_MAX_LENGTH,
                pattern=CHARTS_FILTER_PATTERN,
            ),
        ):
            criteria = ActiveChartsQueryCriteria(
                tool_id=tool_id,
                chamber_id=chamber_id,
                recipe_id=recipe_id,
            )
            try:
                data = self._chart_repository.find_active_chart_set(criteria)
                return {"ok": True, "data": asdict(data)}
            except Exception as e:
                raise_api_error(operation="GET /charts/active", error=e)

        @self.router.get("/charts/history")
        def get_charts_history(
            chart_id: str | None = Query(
                default=None, min_length=1, max_length=64, pattern=CHART_ID_PATTERN
            ),
            chart_set_id: int | None = Query(default=None, ge=1),
            change_source: str | None = Query(
                default=None,
                min_length=1,
                max_length=CHARTS_FILTER_MAX_LENGTH,
                pattern=CHARTS_FILTER_PATTERN,
            ),
            from_ts: datetime | None = Query(default=None),  # noqa: B008
            to_ts: datetime | None = Query(default=None),  # noqa: B008
            limit: int = Query(default=100, ge=1, le=500),
            offset: int = Query(default=0, ge=0),
        ):
            validate_query_datetime_range(from_ts, to_ts, require_pair=False)
            chart_pk = parse_chart_pk(chart_id)
            criteria = ChartsHistoryQueryCriteria(
                chart_pk=chart_pk,
                chart_set_id=chart_set_id,
                change_source=change_source,
                from_ts=normalize_query_datetime(from_ts),
                to_ts=normalize_query_datetime(to_ts),
                limit=limit,
                offset=offset,
            )

            try:
                rows = self._chart_repository.find_chart_history(criteria)
                return {"ok": True, "data": [asdict(row) for row in rows]}
            except Exception as e:
                raise_api_error(operation="GET /charts/history", error=e)

        @self.router.get("/charts/{chart_id}/points")
        def get_chart_points(
            chart_id: str = Path(min_length=1, max_length=64, pattern=CHART_ID_PATTERN),
            limit: int = Query(default=50, ge=1, le=500),
        ):
            chart_pk = parse_chart_pk(chart_id)
            if chart_pk is None:
                raise HTTPException(status_code=400, detail="Invalid chart_id")

            try:
                rows = self._chart_repository.find_chart_points(chart_pk, limit)
                return {"ok": True, "data": [asdict(row) for row in rows]}
            except Exception as e:
                raise_api_error(operation="GET /charts/{chart_id}/points", error=e)

        @self.router.get("/processes/{process_id}/waveform-preview")
        def get_process_waveform_preview(
            process_id: str = Path(
                min_length=1,
                max_length=CHARTS_FILTER_MAX_LENGTH,
                pattern=CHARTS_FILTER_PATTERN,
            ),
            limit: int = Query(default=300, ge=10, le=2000),
        ):
            try:
                data = self._build_waveform_preview(process_id, limit)
                return {"ok": True, "data": data}
            except HTTPException:
                raise
            except Exception as e:
                raise_api_error(operation="GET /processes/{process_id}/waveform-preview", error=e)

        @self.router.get("/judge/results")
        def get_judge_results(
            chart_id: str | None = Query(
                default=None, min_length=1, max_length=64, pattern=CHART_ID_PATTERN
            ),
            process_id: str | None = Query(
                default=None,
                min_length=1,
                max_length=CHARTS_FILTER_MAX_LENGTH,
                pattern=CHARTS_FILTER_PATTERN,
            ),
            lot_id: str | None = Query(
                default=None,
                min_length=1,
                max_length=CHARTS_FILTER_MAX_LENGTH,
                pattern=CHARTS_FILTER_PATTERN,
            ),
            recipe_id: str | None = Query(
                default=None,
                min_length=1,
                max_length=CHARTS_FILTER_MAX_LENGTH,
                pattern=CHARTS_FILTER_PATTERN,
            ),
            level: str | None = Query(default=None, pattern=JUDGE_LEVEL_PATTERN),
            from_ts: datetime | None = Query(default=None),  # noqa: B008
            to_ts: datetime | None = Query(default=None),  # noqa: B008
            limit: int = Query(default=200, ge=1, le=1000),
            offset: int = Query(default=0, ge=0),
        ):
            validate_query_datetime_range(from_ts, to_ts, require_pair=True)
            criteria = JudgeResultsQueryCriteria(
                chart_id=chart_id,
                process_id=process_id,
                lot_id=lot_id,
                recipe_id=recipe_id,
                level=level,
                from_ts=normalize_query_datetime(from_ts),
                to_ts=normalize_query_datetime(to_ts),
                limit=limit,
                offset=offset,
            )

            try:
                rows = self._judge_repository.find_results(criteria)
                return {"ok": True, "data": [asdict(row) for row in rows]}
            except Exception as e:
                raise_api_error(operation="GET /judge/results", error=e)

        @self.router.get("/judge/results/{result_id}")
        def get_judge_result_by_id(
            result_id: str = Path(min_length=1, max_length=64, pattern=RESULT_ID_PATTERN),
        ):
            result_pk = parse_result_pk(result_id)
            try:
                row = self._judge_repository.find_result_by_id(result_pk)
                if row is None:
                    return not_found_error_response(
                        message="judge result not found",
                        details={"result_id": result_id},
                    )
                return {"ok": True, "data": asdict(row)}
            except JudgeDataCorruptionError as e:
                logger.error(
                    "JUDGE_DATA_CORRUPTION: GET /judge/results/{result_id} failed "
                    "(requested_result_id=%s, result_pk=%s)",
                    result_id,
                    result_pk,
                    exc_info=True,
                )
                raise HTTPException(status_code=500, detail="Internal server error") from e
            except Exception as e:
                raise_api_error(operation="GET /judge/results/{result_id}", error=e)

        @self.router.get("/governance/change-requests")
        def get_governance_change_requests(query: ChangeRequestsQuery = Depends()):  # noqa: B008
            con = _connect_readonly(MAIN_DB)
            try:
                rows = self._governance_change_request_repository.list(
                    con,
                    status=query.status,
                    chart_id=query.chart_id,
                    from_ts=normalize_query_datetime(query.from_ts),
                    to_ts=normalize_query_datetime(query.to_ts),
                    limit=query.limit,
                    offset=query.offset,
                )
                return {"ok": True, "data": [asdict(row) for row in rows]}
            except Exception as e:
                raise_api_error(operation="GET /governance/change-requests", error=e)
            finally:
                con.close()

        @self.router.get("/governance/audit-events")
        def get_governance_audit_events(query: GovernanceAuditEventsQuery = Depends()):  # noqa: B008
            con = _connect_readonly(MAIN_DB)
            try:
                validate_query_datetime_range(query.from_ts, query.to_ts, require_pair=False)

                sql = """
                    SELECT
                        id, event_type, actor, actor_role, target_type, target_id,
                        occurred_at, before_json, after_json, correlation_id
                    FROM GovernanceAuditEvents
                """
                where_clauses: list[str] = []
                params: list[object] = []

                if query.event_type is not None:
                    where_clauses.append("event_type = ?")
                    params.append(query.event_type)
                if query.target_type is not None:
                    where_clauses.append("target_type = ?")
                    params.append(query.target_type)
                if query.target_id is not None:
                    where_clauses.append("target_id = ?")
                    params.append(query.target_id)
                if query.from_ts is not None:
                    where_clauses.append("occurred_at >= ?")
                    params.append(normalize_query_datetime(query.from_ts))
                if query.to_ts is not None:
                    where_clauses.append("occurred_at <= ?")
                    params.append(normalize_query_datetime(query.to_ts))

                if where_clauses:
                    sql += " WHERE " + " AND ".join(where_clauses)

                sql += " ORDER BY occurred_at DESC, id DESC LIMIT ? OFFSET ?"
                params.extend((query.limit, query.offset))

                rows = con.execute(sql, params).fetchall()
                data = [
                    {
                        "id": int(row[0]),
                        "event_type": str(row[1]),
                        "actor": str(row[2]),
                        "actor_role": str(row[3]),
                        "target_type": str(row[4]),
                        "target_id": int(row[5]),
                        "occurred_at": str(row[6]),
                        "before_json": row[7],
                        "after_json": row[8],
                        "correlation_id": row[9],
                    }
                    for row in rows
                ]
                return {"ok": True, "data": data}
            except HTTPException:
                raise
            except Exception as e:
                raise_api_error(operation="GET /governance/audit-events", error=e)
            finally:
                con.close()
