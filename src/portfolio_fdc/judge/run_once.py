from __future__ import annotations

import argparse
import json
import logging
import sqlite3
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from portfolio_fdc.db_api.datetime_util import to_utc_millis
from portfolio_fdc.db_api.db import MAIN_DB, _connect, _init_schema

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class JudgeRunSummary:
    """run_once の集計結果。

    Note:
        written は「論理レコード件数」ではなく DB 書き込み操作回数。
        NG の場合は PENDING INSERT と最終 UPDATE の 2 操作になる。
    """

    evaluated: int = 0
    written: int = 0
    notifications_attempted: int = 0
    notifications_failed: int = 0
    stop_api_attempted: int = 0
    stop_api_failed: int = 0
    skipped_without_chart: int = 0


@dataclass(frozen=True)
class _ProcessRow:
    process_id: str
    tool_id: str
    chamber_id: str
    recipe_id: str
    start_ts: str


@dataclass(frozen=True)
class _ParameterRow:
    parameter: str
    step_no: int
    feature_type: str
    feature_value: float


@dataclass(frozen=True)
class _ChartRow:
    chart_id: str
    tool_id: str
    chamber_id: str
    recipe_id: str
    parameter: str
    step_no: int
    feature_type: str
    warning_lcl: float | None
    warning_ucl: float | None
    critical_lcl: float | None
    critical_ucl: float | None


NotificationSink = Callable[[dict[str, Any]], None]
StopApiHook = Callable[[dict[str, Any]], None]


class JudgeEngine:
    """ProcessInfo/Parameters を読み、JudgementResults へ書き込む最小 judge エンジン。"""

    def __init__(
        self,
        *,
        db_path: Path = MAIN_DB,
        notification_sink: NotificationSink | None = None,
        stop_api_hook: StopApiHook | None = None,
    ) -> None:
        self._db_path = db_path
        self._notification_sink = notification_sink or (lambda _payload: None)
        self._stop_api_hook = stop_api_hook or (lambda _payload: None)

    def run_once(self, *, process_id: str | None = None) -> JudgeRunSummary:
        """1 回分の判定を実行し、集計を返す。"""
        _init_schema(self._db_path)
        con = _connect(self._db_path)
        try:
            process_rows = self._load_process_rows(con, process_id=process_id)
            if not process_rows:
                return JudgeRunSummary()

            chart_lookup = self._load_active_chart_lookup(con)
            summary = JudgeRunSummary()

            for process_row in process_rows:
                parameter_rows = self._load_parameter_rows(con, process_row.process_id)
                for parameter_row in parameter_rows:
                    chart_row = chart_lookup.get(
                        self._chart_key(
                            process_row.tool_id,
                            process_row.chamber_id,
                            process_row.recipe_id,
                            parameter_row.parameter,
                            parameter_row.step_no,
                            parameter_row.feature_type,
                        )
                    )
                    if chart_row is None:
                        summary = self._replace_summary(
                            summary,
                            skipped_without_chart=summary.skipped_without_chart + 1,
                        )
                        continue

                    judged_at = to_utc_millis(datetime.now(UTC).isoformat())
                    level = self._evaluate_level(parameter_row.feature_value, chart_row)
                    notification_called = False
                    notification_status = "NOT_CALLED"
                    stop_api_called = False
                    stop_api_status = "NOT_CALLED"

                    if level in {"WARN", "NG"}:
                        notification_called = True
                        summary = self._replace_summary(
                            summary,
                            notifications_attempted=summary.notifications_attempted + 1,
                        )
                        try:
                            self._notification_sink(
                                self._build_message_json(
                                    process_row=process_row,
                                    parameter_row=parameter_row,
                                    chart_row=chart_row,
                                    level=level,
                                    judged_at=judged_at,
                                    notification_called=True,
                                    notification_status=notification_status,
                                    stop_api_called=False,
                                    stop_api_status="NOT_CALLED",
                                )
                            )
                            notification_status = "CALLED"
                        except Exception:
                            notification_status = "FAILED"
                            summary = self._replace_summary(
                                summary,
                                notifications_failed=summary.notifications_failed + 1,
                            )
                            logger.exception(
                                "Notification sink failed for process_id=%s chart_id=%s",
                                process_row.process_id,
                                chart_row.chart_id,
                            )

                    if level == "NG":
                        # NG は PENDING INSERT -> stop API -> 最終 UPDATE の 2 操作。
                        pending_payload = self._build_message_json(
                            process_row=process_row,
                            parameter_row=parameter_row,
                            chart_row=chart_row,
                            level=level,
                            judged_at=judged_at,
                            notification_called=notification_called,
                            notification_status=notification_status,
                            stop_api_called=False,
                            stop_api_status="PENDING",
                        )
                        result_id = self._insert_result(
                            con,
                            process_row=process_row,
                            level=level,
                            judged_at=judged_at,
                            payload=pending_payload,
                        )
                        summary = self._replace_summary(
                            summary,
                            evaluated=summary.evaluated + 1,
                            written=summary.written + 1,
                            stop_api_attempted=summary.stop_api_attempted + 1,
                        )
                        try:
                            self._stop_api_hook(
                                {
                                    "process_id": process_row.process_id,
                                    "chart_id": chart_row.chart_id,
                                    "level": level,
                                    "judged_at": judged_at,
                                    "feature_value": parameter_row.feature_value,
                                }
                            )
                            stop_api_called = True
                            stop_api_status = "CALLED"
                        except Exception:
                            stop_api_called = True
                            stop_api_status = "FAILED"
                            summary = self._replace_summary(
                                summary,
                                stop_api_failed=summary.stop_api_failed + 1,
                            )
                            logger.exception(
                                "Stop API hook failed for process_id=%s chart_id=%s",
                                process_row.process_id,
                                chart_row.chart_id,
                            )
                        final_payload = self._build_message_json(
                            process_row=process_row,
                            parameter_row=parameter_row,
                            chart_row=chart_row,
                            level=level,
                            judged_at=judged_at,
                            notification_called=notification_called,
                            notification_status=notification_status,
                            stop_api_called=stop_api_called,
                            stop_api_status=stop_api_status,
                        )
                        self._update_result(
                            con,
                            result_id=result_id,
                            level=level,
                            judged_at=judged_at,
                            payload=final_payload,
                        )
                        summary = self._replace_summary(
                            summary,
                            written=summary.written + 1,
                        )
                        continue

                    payload = self._build_message_json(
                        process_row=process_row,
                        parameter_row=parameter_row,
                        chart_row=chart_row,
                        level=level,
                        judged_at=judged_at,
                        notification_called=notification_called,
                        notification_status=notification_status,
                        stop_api_called=stop_api_called,
                        stop_api_status=stop_api_status,
                    )
                    self._insert_result(
                        con,
                        process_row=process_row,
                        level=level,
                        judged_at=judged_at,
                        payload=payload,
                    )
                    summary = self._replace_summary(
                        summary,
                        evaluated=summary.evaluated + 1,
                        written=summary.written + 1,
                    )

            con.commit()
            return summary
        finally:
            con.close()

    @staticmethod
    def _replace_summary(summary: JudgeRunSummary, **changes: int) -> JudgeRunSummary:
        # written は DB 操作回数を表す。
        return JudgeRunSummary(
            evaluated=changes.get("evaluated", summary.evaluated),
            written=changes.get("written", summary.written),
            notifications_attempted=changes.get(
                "notifications_attempted",
                summary.notifications_attempted,
            ),
            notifications_failed=changes.get("notifications_failed", summary.notifications_failed),
            stop_api_attempted=changes.get("stop_api_attempted", summary.stop_api_attempted),
            stop_api_failed=changes.get("stop_api_failed", summary.stop_api_failed),
            skipped_without_chart=changes.get(
                "skipped_without_chart",
                summary.skipped_without_chart,
            ),
        )

    def _load_process_rows(
        self,
        con: sqlite3.Connection,
        *,
        process_id: str | None,
    ) -> list[_ProcessRow]:
        if process_id is not None:
            rows = con.execute(
                """
                SELECT process_id, tool_id, chamber_id, recipe_id, start_ts
                FROM ProcessInfo pi
                WHERE pi.process_id = ?
                AND NOT EXISTS (
                    SELECT 1
                    FROM JudgementResults jr
                    WHERE jr.process_id = pi.process_id
                )
                """,
                (process_id,),
            ).fetchall()
        else:
            rows = con.execute(
                """
                SELECT pi.process_id, pi.tool_id, pi.chamber_id, pi.recipe_id, pi.start_ts
                FROM ProcessInfo pi
                WHERE EXISTS (
                    SELECT 1
                    FROM Parameters p
                    WHERE p.process_id = pi.process_id
                )
                AND NOT EXISTS (
                    SELECT 1
                    FROM JudgementResults jr
                    WHERE jr.process_id = pi.process_id
                )
                ORDER BY julianday(pi.start_ts) DESC, pi.process_id DESC
                """,
            ).fetchall()

        return [
            _ProcessRow(
                process_id=str(row[0]),
                tool_id=str(row[1]),
                chamber_id=str(row[2]),
                recipe_id=str(row[3]),
                start_ts=str(row[4]),
            )
            for row in rows
        ]

    def _load_parameter_rows(self, con: sqlite3.Connection, process_id: str) -> list[_ParameterRow]:
        rows = con.execute(
            """
            SELECT parameter, step_no, feature_type, feature_value
            FROM Parameters
            WHERE process_id = ?
            ORDER BY parameter, step_no, feature_type, id
            """,
            (process_id,),
        ).fetchall()
        return [
            _ParameterRow(
                parameter=str(row[0]),
                step_no=int(row[1]),
                feature_type=str(row[2]),
                feature_value=float(row[3]),
            )
            for row in rows
        ]

    def _load_active_chart_lookup(
        self,
        con: sqlite3.Connection,
    ) -> dict[tuple[str, str, str, str, int, str], _ChartRow]:
        rows = con.execute(
            """
            SELECT
                c.id,
                c.tool_id,
                c.chamber_id,
                c.recipe_id,
                c.parameter,
                c.step_no,
                c.feature_type,
                c.warn_low,
                c.warn_high,
                c.crit_low,
                c.crit_high
            FROM ChartsV2 c
            INNER JOIN ActiveChartSet active
                ON active.id = 1
               AND active.chart_set_id = c.chart_set_id
            """,
        ).fetchall()

        lookup: dict[tuple[str, str, str, str, int, str], _ChartRow] = {}
        for row in rows:
            chart_row = _ChartRow(
                chart_id=f"CHART_{int(row[0])}",
                tool_id=str(row[1]),
                chamber_id=str(row[2]),
                recipe_id=str(row[3]),
                parameter=str(row[4]),
                step_no=int(row[5]),
                feature_type=str(row[6]),
                warning_lcl=None if row[7] is None else float(row[7]),
                warning_ucl=None if row[8] is None else float(row[8]),
                critical_lcl=None if row[9] is None else float(row[9]),
                critical_ucl=None if row[10] is None else float(row[10]),
            )
            lookup[
                self._chart_key(
                    chart_row.tool_id,
                    chart_row.chamber_id,
                    chart_row.recipe_id,
                    chart_row.parameter,
                    chart_row.step_no,
                    chart_row.feature_type,
                )
            ] = chart_row
        return lookup

    @staticmethod
    def _chart_key(
        tool_id: str,
        chamber_id: str,
        recipe_id: str,
        parameter: str,
        step_no: int,
        feature_type: str,
    ) -> tuple[str, str, str, str, int, str]:
        return (tool_id, chamber_id, recipe_id, parameter, step_no, feature_type)

    @staticmethod
    def _evaluate_level(feature_value: float, chart_row: _ChartRow) -> str:
        if chart_row.critical_lcl is not None and feature_value < chart_row.critical_lcl:
            return "NG"
        if chart_row.critical_ucl is not None and feature_value > chart_row.critical_ucl:
            return "NG"
        if chart_row.warning_lcl is not None and feature_value < chart_row.warning_lcl:
            return "WARN"
        if chart_row.warning_ucl is not None and feature_value > chart_row.warning_ucl:
            return "WARN"
        return "OK"

    def _build_message_json(
        self,
        *,
        process_row: _ProcessRow,
        parameter_row: _ParameterRow,
        chart_row: _ChartRow,
        level: str,
        judged_at: str,
        notification_called: bool,
        notification_status: str,
        stop_api_called: bool,
        stop_api_status: str,
    ) -> dict[str, Any]:
        return {
            "chart_id": chart_row.chart_id,
            "process_id": process_row.process_id,
            "tool_id": process_row.tool_id,
            "chamber_id": process_row.chamber_id,
            "recipe_id": process_row.recipe_id,
            "start_ts": to_utc_millis(process_row.start_ts),
            "parameter": parameter_row.parameter,
            "step_no": parameter_row.step_no,
            "feature_type": parameter_row.feature_type,
            "feature_value": parameter_row.feature_value,
            "warning_lcl": chart_row.warning_lcl,
            "warning_ucl": chart_row.warning_ucl,
            "critical_lcl": chart_row.critical_lcl,
            "critical_ucl": chart_row.critical_ucl,
            "level": level,
            "judged_at": judged_at,
            "notification_called": notification_called,
            "notification_status": notification_status,
            "stop_api_called": stop_api_called,
            "stop_api_status": stop_api_status,
        }

    def _insert_result(
        self,
        con: sqlite3.Connection,
        *,
        process_row: _ProcessRow,
        level: str,
        judged_at: str,
        payload: dict[str, Any],
    ) -> int:
        payload_str = json.dumps(payload, ensure_ascii=False)
        cur = con.execute(
            """
            INSERT INTO JudgementResults(
                process_id, tool_id, chamber_id, recipe_id, status, judged_at, message_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                process_row.process_id,
                process_row.tool_id,
                process_row.chamber_id,
                process_row.recipe_id,
                level,
                judged_at,
                payload_str,
            ),
        )
        result_id = cur.lastrowid
        if result_id is None:
            raise RuntimeError("failed to get inserted judgement result id")
        return int(result_id)

    def _update_result(
        self,
        con: sqlite3.Connection,
        *,
        result_id: int,
        level: str,
        judged_at: str,
        payload: dict[str, Any],
    ) -> None:
        payload_str = json.dumps(payload, ensure_ascii=False)
        con.execute(
            """
            UPDATE JudgementResults
            SET status = ?, judged_at = ?, message_json = ?
            WHERE id = ?
            """,
            (level, judged_at, payload_str, result_id),
        )


def run_once(
    *,
    db_path: Path = MAIN_DB,
    process_id: str | None = None,
    notification_sink: NotificationSink | None = None,
    stop_api_hook: StopApiHook | None = None,
) -> JudgeRunSummary:
    """judge MVP の 1 回実行を行う。"""
    engine = JudgeEngine(
        db_path=db_path,
        notification_sink=notification_sink,
        stop_api_hook=stop_api_hook,
    )
    return engine.run_once(process_id=process_id)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run judge MVP once")
    parser.add_argument("--db-path", type=Path, default=MAIN_DB, help="SQLite DB path")
    parser.add_argument(
        "--process-id",
        default=None,
        help="Target process_id. If omitted, all processes with Parameters are judged.",
    )
    parser.add_argument(
        "--db-api",
        default=None,
        help="Reserved for future remote integration. The MVP uses the local DB path.",
    )
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)
    if args.db_api:
        logger.info("--db-api is reserved in MVP and will be ignored: %s", args.db_api)

    summary = run_once(db_path=args.db_path, process_id=args.process_id)
    print(
        json.dumps(
            {
                "ok": True,
                "data": {
                    "evaluated": summary.evaluated,
                    "written": summary.written,
                    "notifications_attempted": summary.notifications_attempted,
                    "notifications_failed": summary.notifications_failed,
                    "stop_api_attempted": summary.stop_api_attempted,
                    "stop_api_failed": summary.stop_api_failed,
                    "skipped_without_chart": summary.skipped_without_chart,
                },
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
