from __future__ import annotations

import json
import sqlite3
from collections.abc import Callable
from datetime import UTC, datetime

from fastapi import APIRouter, Path, Request
from fastapi.responses import JSONResponse

from ..api_common import (
    NOTIFICATION_RETRY_BACKOFF_MINUTES,
    GovernanceApplyValidationError,
    bad_request_error_response,
    compute_notification_next_retry_at,
    conflict_error_response,
    duplicate_idempotency_error_response,
    not_found_error_response,
    parse_threshold_patch,
    raise_api_error,
    thresholds_equal,
    validate_threshold_consistency,
    validation_error_response,
)
from ..audit_event_writer import AuditEventWriter
from ..datetime_util import to_utc_millis
from ..db import MAIN_DB, _connect
from ..governance_repository import (
    GovernanceApprovalsRepository,
    GovernanceChangeRequestRepository,
    GovernanceEmergencyChangesRepository,
    GovernanceNotFoundError,
    GovernanceRatificationsRepository,
)
from ..schemas import (
    ChangeRequestApplyIn,
    ChangeRequestApproveIn,
    ChangeRequestIn,
    EmergencyChangeIn,
    EmergencyChangeRatifyIn,
)
from ..task_runner import DBTaskRunner


class _GovernanceChangeRequestIdempotencyConflict(Exception):
    pass


class _GovernanceChangeRequestChartFkViolation(Exception):
    pass


class _GovernanceApproveAlreadyApproved(Exception):
    pass


class _GovernanceApproveInvalidState(Exception):
    pass


class _GovernanceApplyInvalidState(Exception):
    pass


class _GovernanceApplyChartNotFound(Exception):
    pass


class _GovernanceApplyVersionConflict(Exception):
    def __init__(
        self,
        *,
        current_version: int,
        current_updated_at: str,
        chart_id: int,
        current_status: str,
    ) -> None:
        self.current_version = current_version
        self.current_updated_at = current_updated_at
        self.chart_id = chart_id
        self.current_status = current_status


class _GovernanceNotificationNotFound(Exception):
    pass


class _GovernanceNotificationInvalidState(Exception):
    def __init__(self, *, status: str) -> None:
        self.status = status


class _GovernanceNotificationRetryLimitExceeded(Exception):
    def __init__(self, *, retry_count: int) -> None:
        self.retry_count = retry_count


class _GovernanceNotificationConcurrentModification(Exception):
    pass


class _GovernanceEmergencyChangeChartNotFound(Exception):
    pass


class _GovernanceEmergencyRatificationConflict(Exception):
    def __init__(self, *, ec_id: int) -> None:
        self.ec_id = ec_id


def _is_duplicate_change_request_idempotency_error(error: sqlite3.IntegrityError) -> bool:
    message = str(error)
    return (
        "GovernanceChangeRequests.idempotency_key" in message
        or "idx_change_requests_idempotency" in message
    )


def _is_governance_change_request_chart_fk_error(error: sqlite3.IntegrityError) -> bool:
    return "foreign key constraint failed" in str(error).lower()


def _is_governance_ratification_ec_unique_error(error: sqlite3.IntegrityError) -> bool:
    message = str(error)
    return (
        "GovernanceRatifications.ec_id" in message
        or "UNIQUE constraint failed: GovernanceRatifications.ec_id" in message
    )


class GovernanceRouter:
    """ガバナンス書き込み系エンドポイントをまとめるルータークラス。"""

    def __init__(
        self,
        *,
        governance_change_request_repository: GovernanceChangeRequestRepository,
        governance_approvals_repository: GovernanceApprovalsRepository,
        governance_emergency_changes_repository: GovernanceEmergencyChangesRepository,
        governance_ratifications_repository: GovernanceRatificationsRepository,
        audit_event_writer: AuditEventWriter,
        get_runner: Callable[[Request], DBTaskRunner],
    ) -> None:
        self._governance_change_request_repository = governance_change_request_repository
        self._governance_approvals_repository = governance_approvals_repository
        self._governance_emergency_changes_repository = governance_emergency_changes_repository
        self._governance_ratifications_repository = governance_ratifications_repository
        self._audit_event_writer = audit_event_writer
        self._get_runner = get_runner
        self.router = APIRouter()
        self._register_routes()

    def _register_routes(self) -> None:
        @self.router.post("/governance/emergency-changes")
        def create_governance_emergency_change(payload: EmergencyChangeIn, request: Request):
            changed_at = to_utc_millis(datetime.now(UTC).isoformat())
            runner = self._get_runner(request)

            def _write() -> dict[str, object]:
                con = _connect(MAIN_DB)
                try:
                    con.execute("BEGIN")
                    row = con.execute(
                        """
                        SELECT
                            id, chart_set_id, tool_id, chamber_id, recipe_id, parameter,
                            step_no, feature_type, warn_low, warn_high, crit_low, crit_high,
                            version
                        FROM ChartsV2
                        WHERE id = ?
                        """,
                        (payload.chart_id,),
                    ).fetchone()
                    if row is None:
                        raise _GovernanceEmergencyChangeChartNotFound

                    (
                        chart_id,
                        chart_set_id,
                        tool_id,
                        chamber_id,
                        recipe_id,
                        parameter,
                        step_no,
                        feature_type,
                        old_warn_low,
                        old_warn_high,
                        old_crit_low,
                        old_crit_high,
                        current_version,
                    ) = row

                    try:
                        patch = parse_threshold_patch(payload.change_payload)
                    except json.JSONDecodeError as e:
                        raise GovernanceApplyValidationError(
                            message="change_payload must be valid JSON"
                        ) from e
                    if not any(
                        k in patch for k in ["warn_low", "warn_high", "crit_low", "crit_high"]
                    ):
                        raise GovernanceApplyValidationError(
                            message="change_payload must contain at least one of: "
                            "warn_low, warn_high, crit_low, crit_high"
                        )

                    new_warn_low = patch.get("warn_low", old_warn_low)
                    new_warn_high = patch.get("warn_high", old_warn_high)
                    new_crit_low = patch.get("crit_low", old_crit_low)
                    new_crit_high = patch.get("crit_high", old_crit_high)

                    validate_threshold_consistency(
                        warn_low=new_warn_low,
                        warn_high=new_warn_high,
                        crit_low=new_crit_low,
                        crit_high=new_crit_high,
                    )

                    is_noop = (
                        thresholds_equal(old_warn_low, new_warn_low)
                        and thresholds_equal(old_warn_high, new_warn_high)
                        and thresholds_equal(old_crit_low, new_crit_low)
                        and thresholds_equal(old_crit_high, new_crit_high)
                    )

                    resulting_version = int(current_version)
                    before_json = json.dumps(
                        {
                            "warn_low": old_warn_low,
                            "warn_high": old_warn_high,
                            "crit_low": old_crit_low,
                            "crit_high": old_crit_high,
                        }
                    )
                    after_json = json.dumps(
                        {
                            "warn_low": new_warn_low,
                            "warn_high": new_warn_high,
                            "crit_low": new_crit_low,
                            "crit_high": new_crit_high,
                        }
                    )

                    if not is_noop:
                        resulting_version = int(current_version) + 1
                        con.execute(
                            """
                            UPDATE ChartsV2
                            SET warn_low = ?, warn_high = ?, crit_low = ?, crit_high = ?,
                                version = ?, updated_at = ?, updated_by = ?,
                                update_reason = ?, update_source = ?
                            WHERE id = ?
                            """,
                            (
                                new_warn_low,
                                new_warn_high,
                                new_crit_low,
                                new_crit_high,
                                resulting_version,
                                changed_at,
                                payload.changed_by,
                                payload.reason,
                                "emergency_manual",
                                chart_id,
                            ),
                        )
                        con.execute(
                            """
                            INSERT INTO ChartsHistory(
                                chart_set_id, tool_id, chamber_id, recipe_id, parameter,
                                step_no, feature_type,
                                old_warn_low, old_warn_high, old_crit_low, old_crit_high,
                                new_warn_low, new_warn_high, new_crit_low, new_crit_high,
                                changed_at, changed_by, change_reason, change_source, chart_id
                            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                            """,
                            (
                                chart_set_id,
                                tool_id,
                                chamber_id,
                                recipe_id,
                                parameter,
                                step_no,
                                feature_type,
                                old_warn_low,
                                old_warn_high,
                                old_crit_low,
                                old_crit_high,
                                new_warn_low,
                                new_warn_high,
                                new_crit_low,
                                new_crit_high,
                                changed_at,
                                payload.changed_by,
                                payload.reason,
                                "emergency_manual",
                                chart_id,
                            ),
                        )

                    emergency_change_id = self._governance_emergency_changes_repository.create(
                        con,
                        chart_id=int(chart_id),
                        changed_by=payload.changed_by,
                        changed_by_role=payload.changed_by_role,
                        changed_at=changed_at,
                        reason=payload.reason,
                        before_json=before_json,
                        after_json=after_json,
                        resulting_version=resulting_version,
                        related_issue_or_pr=None,
                    )
                    audit_event_id = self._audit_event_writer.write(
                        con,
                        event_type="emergency_changed",
                        actor=payload.changed_by,
                        actor_role=payload.changed_by_role,
                        target_type="emergency_change",
                        target_id=emergency_change_id,
                        occurred_at=changed_at,
                        before_json=before_json,
                        after_json=after_json,
                        correlation_id=f"emergency:{emergency_change_id}",
                    )
                    con.execute(
                        """
                        INSERT INTO GovernanceNotificationOutbox(event_id, status)
                        VALUES (?, 'pending')
                        """,
                        (audit_event_id,),
                    )
                    con.commit()
                    return {
                        "request_id": emergency_change_id,
                        "status": "applied",
                        "resulting_version": resulting_version,
                        "noop": is_noop,
                    }
                except Exception:
                    con.rollback()
                    raise
                finally:
                    con.close()

            try:
                data = runner.submit("write", _write)
                return {"ok": True, "data": data}
            except _GovernanceEmergencyChangeChartNotFound:
                return not_found_error_response(
                    code="CHART_NOT_FOUND",
                    message="target chart not found",
                    details={"chart_id": str(payload.chart_id)},
                )
            except GovernanceApplyValidationError as e:
                return validation_error_response(
                    issues=[
                        {
                            "loc": ["body", "change_payload"],
                            "msg": e.message,
                            "type": "value_error",
                        }
                    ]
                )
            except Exception as e:
                raise_api_error(operation="POST /governance/emergency-changes", error=e)

        @self.router.post("/governance/emergency-changes/{request_id}/ratify")
        def ratify_governance_emergency_change(
            payload: EmergencyChangeRatifyIn,
            request: Request,
            request_id: int = Path(ge=1),
        ):
            ratified_at = to_utc_millis(datetime.now(UTC).isoformat())
            runner = self._get_runner(request)

            def _write() -> dict[str, object]:
                con = _connect(MAIN_DB)
                try:
                    con.execute("BEGIN")
                    self._governance_emergency_changes_repository.find_by_id(con, request_id)

                    try:
                        self._governance_ratifications_repository.create(
                            con,
                            ec_id=request_id,
                            ratified_by_role=payload.ratified_by_role,
                            ratified_at=ratified_at,
                            ratification_comment=payload.ratification_comment,
                            related_pr=payload.related_pr,
                        )
                    except sqlite3.IntegrityError as e:
                        if _is_governance_ratification_ec_unique_error(e):
                            raise _GovernanceEmergencyRatificationConflict(ec_id=request_id) from e
                        if "foreign key constraint failed" in str(e).lower():
                            raise GovernanceNotFoundError(
                                f"GovernanceEmergencyChanges id={request_id} not found"
                            ) from e
                        raise

                    if payload.related_pr is not None:
                        con.execute(
                            """
                            UPDATE GovernanceEmergencyChanges
                            SET related_issue_or_pr = ?
                            WHERE id = ?
                            """,
                            (payload.related_pr, request_id),
                        )

                    self._audit_event_writer.write(
                        con,
                        event_type="emergency_ratified",
                        actor=payload.ratified_by,
                        actor_role=payload.ratified_by_role,
                        target_type="emergency_change",
                        target_id=request_id,
                        occurred_at=ratified_at,
                        correlation_id=f"emergency:{request_id}",
                    )
                    con.commit()
                    return {"request_id": request_id, "status": "ratified"}
                except Exception:
                    con.rollback()
                    raise
                finally:
                    con.close()

            try:
                data = runner.submit("write", _write)
                return {"ok": True, "data": data}
            except GovernanceNotFoundError:
                return not_found_error_response(
                    message="emergency change not found",
                    details={"request_id": str(request_id)},
                )
            except _GovernanceEmergencyRatificationConflict:
                return conflict_error_response(
                    code="ALREADY_RATIFIED",
                    message="emergency change is already ratified",
                    details={"request_id": str(request_id)},
                )
            except Exception as e:
                raise_api_error(
                    operation="POST /governance/emergency-changes/{request_id}/ratify",
                    error=e,
                )

        @self.router.post("/governance/notifications/{event_id}/retry")
        def retry_governance_notification(request: Request, event_id: int = Path(ge=1)):
            runner = self._get_runner(request)

            def _retry_failed_notification_outbox() -> dict[str, object]:
                con = _connect(MAIN_DB)
                try:
                    con.execute("BEGIN")
                    row = con.execute(
                        """
                        SELECT id, status, retry_count
                        FROM GovernanceNotificationOutbox
                        WHERE event_id = ?
                        """,
                        (event_id,),
                    ).fetchone()
                    if row is None:
                        raise _GovernanceNotificationNotFound

                    outbox_id = int(row[0])
                    status = str(row[1])
                    retry_count = int(row[2])

                    if status != "failed":
                        raise _GovernanceNotificationInvalidState(status=status)
                    if retry_count >= len(NOTIFICATION_RETRY_BACKOFF_MINUTES):
                        raise _GovernanceNotificationRetryLimitExceeded(retry_count=retry_count)

                    now = datetime.now(UTC)
                    now_iso = to_utc_millis(now.isoformat())
                    next_retry_at = compute_notification_next_retry_at(
                        base_time=now,
                        retry_count=retry_count + 1,
                    )

                    update_cur = con.execute(
                        """
                        UPDATE GovernanceNotificationOutbox
                        SET status = 'pending',
                            retry_count = retry_count + 1,
                            next_retry_at = ?,
                            last_attempt_at = ?,
                            last_error = NULL
                        WHERE id = ?
                          AND status = 'failed'
                          AND retry_count = ?
                        """,
                        (next_retry_at, now_iso, outbox_id, retry_count),
                    )
                    if update_cur.rowcount == 0:
                        latest = con.execute(
                            """
                            SELECT status, retry_count
                            FROM GovernanceNotificationOutbox
                            WHERE id = ?
                            """,
                            (outbox_id,),
                        ).fetchone()
                        if latest is None:
                            raise _GovernanceNotificationNotFound

                        latest_status = str(latest[0])
                        latest_retry_count = int(latest[1])
                        if latest_status != "failed":
                            raise _GovernanceNotificationInvalidState(status=latest_status)
                        if latest_retry_count >= len(NOTIFICATION_RETRY_BACKOFF_MINUTES):
                            raise _GovernanceNotificationRetryLimitExceeded(
                                retry_count=latest_retry_count
                            )
                        raise _GovernanceNotificationConcurrentModification

                    self._audit_event_writer.write(
                        con,
                        event_type="notification_queued",
                        actor="ops",
                        actor_role="ops",
                        target_type="notification",
                        target_id=outbox_id,
                        occurred_at=now_iso,
                        correlation_id=f"event:{event_id}",
                    )
                    con.commit()
                    return {
                        "event_id": event_id,
                        "status": "pending",
                        "retry_count": retry_count + 1,
                        "next_retry_at": next_retry_at,
                    }
                except Exception:
                    con.rollback()
                    raise
                finally:
                    con.close()

            try:
                data = runner.submit("write", _retry_failed_notification_outbox)
                return {"ok": True, "data": data}
            except _GovernanceNotificationNotFound:
                return not_found_error_response(
                    message="notification outbox not found",
                    details={"event_id": str(event_id)},
                )
            except _GovernanceNotificationInvalidState as e:
                return bad_request_error_response(
                    code="INVALID_RETRY_TARGET",
                    message="only failed notification can be retried",
                    details={"event_id": str(event_id), "current_status": e.status},
                )
            except _GovernanceNotificationRetryLimitExceeded as e:
                return conflict_error_response(
                    code="RETRY_LIMIT_EXCEEDED",
                    message="retry_count has reached the maximum",
                    details={
                        "event_id": str(event_id),
                        "retry_count": str(e.retry_count),
                        "max_retry_count": str(len(NOTIFICATION_RETRY_BACKOFF_MINUTES)),
                    },
                )
            except _GovernanceNotificationConcurrentModification:
                return conflict_error_response(
                    code="CONCURRENT_MODIFICATION",
                    message="notification outbox was modified concurrently",
                    details={"event_id": str(event_id)},
                )
            except Exception as e:
                raise_api_error(
                    operation="POST /governance/notifications/{event_id}/retry", error=e
                )

        @self.router.post("/governance/change-requests")
        def create_governance_change_request(payload: ChangeRequestIn, request: Request):
            proposed_at = to_utc_millis(datetime.now(UTC).isoformat())
            runner = self._get_runner(request)

            def _write() -> dict[str, int | str]:
                con = _connect(MAIN_DB)
                try:
                    con.execute("BEGIN")
                    try:
                        request_id = self._governance_change_request_repository.create(
                            con,
                            chart_id=payload.chart_id,
                            proposed_by=payload.proposed_by,
                            proposed_at=proposed_at,
                            change_payload=payload.change_payload,
                            expected_version=payload.expected_version,
                            idempotency_key=payload.idempotency_key,
                        )
                    except sqlite3.IntegrityError as e:
                        if _is_duplicate_change_request_idempotency_error(e):
                            raise _GovernanceChangeRequestIdempotencyConflict from e
                        if _is_governance_change_request_chart_fk_error(e):
                            raise _GovernanceChangeRequestChartFkViolation from e
                        raise

                    self._audit_event_writer.write(
                        con,
                        event_type="change_requested",
                        actor=payload.proposed_by,
                        actor_role="requester",
                        target_type="change_request",
                        target_id=request_id,
                        occurred_at=proposed_at,
                        correlation_id=payload.idempotency_key,
                    )

                    row = self._governance_change_request_repository.find_by_id(con, request_id)
                    con.commit()
                    return {"request_id": request_id, "status": row.status}
                except Exception:
                    con.rollback()
                    raise
                finally:
                    con.close()

            try:
                data = runner.submit("write", _write)
                return {"ok": True, "data": data}
            except _GovernanceChangeRequestIdempotencyConflict:
                return duplicate_idempotency_error_response(idempotency_key=payload.idempotency_key)
            except _GovernanceChangeRequestChartFkViolation:
                return validation_error_response(
                    issues=[
                        {
                            "loc": ["body", "chart_id"],
                            "msg": "chart_id must reference an existing chart",
                            "type": "value_error",
                        }
                    ]
                )
            except Exception as e:
                raise_api_error(operation="POST /governance/change-requests", error=e)

        @self.router.post("/governance/change-requests/{request_id}/approve")
        def approve_governance_change_request(
            payload: ChangeRequestApproveIn,
            request: Request,
            request_id: int = Path(ge=1),
        ):
            approved_at = to_utc_millis(datetime.now(UTC).isoformat())
            runner = self._get_runner(request)

            def _write() -> dict[str, int | str]:
                con = _connect(MAIN_DB)
                try:
                    con.execute("BEGIN")
                    row = self._governance_change_request_repository.find_by_id(con, request_id)

                    if row.status == "approved":
                        raise _GovernanceApproveAlreadyApproved
                    if row.status != "pending":
                        raise _GovernanceApproveInvalidState

                    self._governance_approvals_repository.create(
                        con,
                        request_id=request_id,
                        approved_by=payload.approved_by,
                        approved_by_role=payload.approved_by_role,
                        approved_at=approved_at,
                        comment=payload.comment,
                    )
                    self._governance_change_request_repository.update_status(
                        con,
                        record_id=request_id,
                        new_status="approved",
                    )
                    self._audit_event_writer.write(
                        con,
                        event_type="change_approved",
                        actor=payload.approved_by,
                        actor_role=payload.approved_by_role,
                        target_type="change_request",
                        target_id=request_id,
                        occurred_at=approved_at,
                        correlation_id=f"request:{request_id}",
                    )
                    con.commit()
                    return {"request_id": request_id, "status": "approved"}
                except Exception:
                    con.rollback()
                    raise
                finally:
                    con.close()

            try:
                data = runner.submit("write", _write)
                return {
                    "ok": True,
                    "data": data,
                    "timestamp": to_utc_millis(datetime.now(UTC).isoformat()),
                }
            except GovernanceNotFoundError:
                return not_found_error_response(
                    message="change request not found",
                    details={"request_id": str(request_id)},
                )
            except _GovernanceApproveAlreadyApproved:
                return conflict_error_response(
                    code="ALREADY_APPROVED",
                    message="change request is already approved",
                    details={"request_id": str(request_id)},
                )
            except _GovernanceApproveInvalidState:
                return conflict_error_response(
                    code="INVALID_STATUS_TRANSITION",
                    message="only pending change request can be approved",
                    details={"request_id": str(request_id)},
                )
            except Exception as e:
                raise_api_error(
                    operation="POST /governance/change-requests/{request_id}/approve",
                    error=e,
                )

        @self.router.post("/governance/change-requests/{request_id}/apply")
        def apply_governance_change_request(
            payload: ChangeRequestApplyIn,
            request: Request,
            request_id: int = Path(ge=1),
        ):
            applied_at = to_utc_millis(datetime.now(UTC).isoformat())
            runner = self._get_runner(request)

            def _write() -> dict[str, object]:
                con = _connect(MAIN_DB)
                try:
                    con.execute("BEGIN")
                    try:
                        req = self._governance_change_request_repository.find_by_id(con, request_id)
                        if req.status != "approved":
                            raise _GovernanceApplyInvalidState

                        chart_row = con.execute(
                            """
                            SELECT
                                id, chart_set_id, tool_id, chamber_id, recipe_id, parameter,
                                step_no, feature_type, warn_low, warn_high, crit_low, crit_high,
                                version, updated_at
                            FROM ChartsV2
                            WHERE id = ?
                            """,
                            (req.chart_id,),
                        ).fetchone()
                        if chart_row is None:
                            raise _GovernanceApplyChartNotFound

                        (
                            chart_id,
                            chart_set_id,
                            tool_id,
                            chamber_id,
                            recipe_id,
                            parameter,
                            step_no,
                            feature_type,
                            old_warn_low,
                            old_warn_high,
                            old_crit_low,
                            old_crit_high,
                            current_version,
                            current_updated_at,
                        ) = chart_row

                        if int(current_version) != int(req.expected_version):
                            raise _GovernanceApplyVersionConflict(
                                current_version=int(current_version),
                                current_updated_at=str(current_updated_at),
                                chart_id=int(chart_id),
                                current_status=str(getattr(req, "status", "unknown") or "unknown"),
                            )

                        patch = parse_threshold_patch(req.change_payload)
                        new_warn_low = patch.get("warn_low", old_warn_low)
                        new_warn_high = patch.get("warn_high", old_warn_high)
                        new_crit_low = patch.get("crit_low", old_crit_low)
                        new_crit_high = patch.get("crit_high", old_crit_high)

                        validate_threshold_consistency(
                            warn_low=new_warn_low,
                            warn_high=new_warn_high,
                            crit_low=new_crit_low,
                            crit_high=new_crit_high,
                        )

                        is_noop = (
                            thresholds_equal(old_warn_low, new_warn_low)
                            and thresholds_equal(old_warn_high, new_warn_high)
                            and thresholds_equal(old_crit_low, new_crit_low)
                            and thresholds_equal(old_crit_high, new_crit_high)
                        )

                        resulting_version = int(current_version)
                        if not is_noop:
                            resulting_version = int(current_version) + 1
                            con.execute(
                                """
                                UPDATE ChartsV2
                                SET warn_low = ?, warn_high = ?, crit_low = ?, crit_high = ?,
                                    version = ?, updated_at = ?, updated_by = ?,
                                    update_reason = ?, update_source = ?
                                WHERE id = ?
                                """,
                                (
                                    new_warn_low,
                                    new_warn_high,
                                    new_crit_low,
                                    new_crit_high,
                                    resulting_version,
                                    applied_at,
                                    payload.applied_by,
                                    payload.reason,
                                    "governance_apply",
                                    chart_id,
                                ),
                            )
                            con.execute(
                                """
                                INSERT INTO ChartsHistory(
                                    chart_set_id, tool_id, chamber_id, recipe_id, parameter,
                                    step_no, feature_type,
                                    old_warn_low, old_warn_high, old_crit_low, old_crit_high,
                                    new_warn_low, new_warn_high, new_crit_low, new_crit_high,
                                    changed_at, changed_by, change_reason, change_source, chart_id
                                ) VALUES (
                                    ?, ?, ?, ?, ?, ?, ?, ?, ?, ?,
                                    ?, ?, ?, ?, ?, ?, ?, ?, ?, ?
                                )
                                """,
                                (
                                    chart_set_id,
                                    tool_id,
                                    chamber_id,
                                    recipe_id,
                                    parameter,
                                    step_no,
                                    feature_type,
                                    old_warn_low,
                                    old_warn_high,
                                    old_crit_low,
                                    old_crit_high,
                                    new_warn_low,
                                    new_warn_high,
                                    new_crit_low,
                                    new_crit_high,
                                    applied_at,
                                    payload.applied_by,
                                    payload.reason,
                                    "governance_apply",
                                    chart_id,
                                ),
                            )

                        con.execute(
                            """
                            INSERT OR REPLACE INTO GovernanceApplyResults(
                                request_id, applied_at, success, resulting_version,
                                error_code, error_message
                            ) VALUES (?, ?, 1, ?, NULL, NULL)
                            """,
                            (request_id, applied_at, resulting_version),
                        )
                        self._governance_change_request_repository.update_status(
                            con,
                            record_id=request_id,
                            new_status="applied",
                        )
                        self._audit_event_writer.write(
                            con,
                            event_type="change_applied",
                            actor=payload.applied_by,
                            actor_role=payload.applied_by_role,
                            target_type="change_request",
                            target_id=request_id,
                            occurred_at=applied_at,
                            correlation_id=f"request:{request_id}",
                        )
                        con.commit()
                        return {
                            "request_id": request_id,
                            "status": "applied",
                            "resulting_version": resulting_version,
                            "noop": is_noop,
                        }
                    except (
                        _GovernanceApplyInvalidState,
                        GovernanceApplyValidationError,
                        _GovernanceApplyVersionConflict,
                    ):
                        self._audit_event_writer.write(
                            con,
                            event_type="change_apply_failed",
                            actor=payload.applied_by,
                            actor_role=payload.applied_by_role,
                            target_type="change_request",
                            target_id=request_id,
                            occurred_at=applied_at,
                            correlation_id=f"request:{request_id}",
                        )
                        con.commit()
                        raise
                except Exception:
                    con.rollback()
                    raise
                finally:
                    con.close()

            try:
                data = runner.submit("write", _write)
                return {
                    "ok": True,
                    "data": data,
                    "timestamp": to_utc_millis(datetime.now(UTC).isoformat()),
                }
            except GovernanceNotFoundError:
                return not_found_error_response(
                    message="change request not found",
                    details={"request_id": str(request_id)},
                )
            except _GovernanceApplyInvalidState:
                return conflict_error_response(
                    code="INVALID_STATUS_TRANSITION",
                    message="only approved change request can be applied",
                    details={"request_id": str(request_id)},
                )
            except _GovernanceApplyChartNotFound:
                return bad_request_error_response(
                    code="CHART_NOT_FOUND",
                    message="target chart not found",
                    details={"request_id": str(request_id)},
                )
            except _GovernanceApplyVersionConflict as e:
                return JSONResponse(
                    status_code=409,
                    content={
                        "ok": False,
                        "error": {
                            "code": "STALE_EXPECTED_VERSION",
                            "message": "expected_version does not match current version",
                            "details": {
                                "request_id": str(request_id),
                                "current": {
                                    "chart_id": e.chart_id,
                                    "version": e.current_version,
                                    "updated_at": e.current_updated_at,
                                    "status": e.current_status,
                                },
                            },
                        },
                    },
                )
            except GovernanceApplyValidationError as e:
                return validation_error_response(
                    issues=[
                        {
                            "loc": ["body", "change_payload"],
                            "msg": e.message,
                            "type": "value_error",
                        }
                    ]
                )
            except Exception as e:
                raise_api_error(
                    operation="POST /governance/change-requests/{request_id}/apply",
                    error=e,
                )
