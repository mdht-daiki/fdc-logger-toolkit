from __future__ import annotations

from datetime import UTC, datetime, timedelta
from uuid import uuid4

from fastapi.testclient import TestClient

from portfolio_fdc.db_api.db import MAIN_DB, _connect


def _insert_audit_event_for_outbox(correlation_id: str) -> int:
    con = _connect(MAIN_DB)
    try:
        cur = con.execute(
            """
            INSERT INTO GovernanceAuditEvents(
                event_type, actor, actor_role, target_type, target_id,
                occurred_at, before_json, after_json, correlation_id
            ) VALUES (?, ?, ?, ?, ?, ?, NULL, NULL, ?)
            """,
            (
                "notification_emitted",
                "system",
                "system",
                "change_request",
                1,
                "2026-05-10T00:00:00.000Z",
                correlation_id,
            ),
        )
        con.commit()
        row_id = cur.lastrowid
        if row_id is None:
            raise RuntimeError("Failed to insert audit event")
        return int(row_id)
    finally:
        con.close()


def _insert_outbox(
    *, event_id: int, status: str, retry_count: int, last_error: str | None = None
) -> int:
    con = _connect(MAIN_DB)
    try:
        cur = con.execute(
            """
            INSERT INTO GovernanceNotificationOutbox(
                event_id, status, retry_count, next_retry_at,
                last_attempt_at, last_error, delivered_at
            ) VALUES (?, ?, ?, ?, ?, ?, NULL)
            """,
            (
                event_id,
                status,
                retry_count,
                "2026-05-10T00:01:00.000Z",
                "2026-05-10T00:00:00.000Z",
                last_error,
            ),
        )
        con.commit()
        row_id = cur.lastrowid
        if row_id is None:
            raise RuntimeError("Failed to insert notification outbox")
        return int(row_id)
    finally:
        con.close()


def _find_outbox_by_event_id(event_id: int) -> tuple[str, int, str | None, str | None]:
    con = _connect(MAIN_DB)
    try:
        row = con.execute(
            """
            SELECT status, retry_count, next_retry_at, last_error
            FROM GovernanceNotificationOutbox
            WHERE event_id = ?
            """,
            (event_id,),
        ).fetchone()
        if row is None:
            raise RuntimeError("outbox not found")
        return (str(row[0]), int(row[1]), row[2], row[3])
    finally:
        con.close()


def _delete_seeded_notification_records(event_id: int, correlation_id: str) -> None:
    con = _connect(MAIN_DB)
    try:
        con.execute(
            "DELETE FROM GovernanceNotificationOutbox WHERE event_id = ?",
            (event_id,),
        )
        con.execute(
            "DELETE FROM GovernanceAuditEvents WHERE id = ?",
            (event_id,),
        )
        con.execute(
            "DELETE FROM GovernanceAuditEvents WHERE correlation_id = ?",
            (correlation_id,),
        )
        con.execute(
            "DELETE FROM GovernanceAuditEvents WHERE correlation_id = ?",
            (f"event:{event_id}",),
        )
        con.commit()
    finally:
        con.close()


def _count_notification_queued_events(event_id: int) -> int:
    con = _connect(MAIN_DB)
    try:
        row = con.execute(
            """
                        SELECT COUNT(*)
                        FROM GovernanceAuditEvents AS ae
                        JOIN GovernanceNotificationOutbox AS outbox
                            ON outbox.id = ae.target_id
                        WHERE outbox.event_id = ?
                            AND ae.target_type = 'notification'
                            AND ae.event_type = 'notification_queued'
                        """,
            (event_id,),
        ).fetchone()
        return int(row[0])
    finally:
        con.close()


def _parse_utc_millis(ts: str) -> datetime:
    return datetime.fromisoformat(ts.replace("Z", "+00:00"))


def test_post_notifications_retry_success_updates_failed_to_pending(
    client: TestClient,
) -> None:
    correlation_id = f"notif-retry-success-{uuid4().hex[:8]}"
    event_id = _insert_audit_event_for_outbox(correlation_id)
    _insert_outbox(event_id=event_id, status="failed", retry_count=1, last_error="smtp timeout")

    try:
        before_call = datetime.now(UTC)
        before_audit_count = _count_notification_queued_events(event_id)
        res = client.post(f"/governance/notifications/{event_id}/retry")
        after_call = datetime.now(UTC)

        assert res.status_code == 200
        body = res.json()
        assert body["ok"] is True
        assert body["data"]["event_id"] == event_id
        assert body["data"]["status"] == "pending"
        assert body["data"]["retry_count"] == 2
        assert body["data"]["next_retry_at"] is not None

        status, retry_count, next_retry_at, last_error = _find_outbox_by_event_id(event_id)
        assert status == "pending"
        assert retry_count == 2
        assert next_retry_at == body["data"]["next_retry_at"]
        assert last_error is None

        actual_next_retry_at = _parse_utc_millis(body["data"]["next_retry_at"])
        lower_bound = before_call + timedelta(minutes=5) - timedelta(seconds=2)
        upper_bound = after_call + timedelta(minutes=5) + timedelta(seconds=2)
        assert lower_bound <= actual_next_retry_at <= upper_bound
        assert _count_notification_queued_events(event_id) == before_audit_count + 1
    finally:
        _delete_seeded_notification_records(event_id, correlation_id)


def test_post_notifications_retry_returns_404_when_event_not_found(
    client: TestClient,
) -> None:
    res = client.post("/governance/notifications/9223372036854775807/retry")

    assert res.status_code == 404
    body = res.json()
    assert body["ok"] is False
    assert body["error"]["code"] == "NOT_FOUND"


def test_post_notifications_retry_returns_400_when_status_is_pending(
    client: TestClient,
) -> None:
    correlation_id = f"notif-retry-pending-{uuid4().hex[:8]}"
    event_id = _insert_audit_event_for_outbox(correlation_id)
    _insert_outbox(event_id=event_id, status="pending", retry_count=1)

    try:
        seeded_status, seeded_retry_count, _, _ = _find_outbox_by_event_id(event_id)
        res = client.post(f"/governance/notifications/{event_id}/retry")

        assert res.status_code == 400
        body = res.json()
        assert body["ok"] is False
        assert body["error"]["code"] == "INVALID_RETRY_TARGET"
        assert body["error"]["details"]["current_status"] == "pending"
        status, retry_count, _, _ = _find_outbox_by_event_id(event_id)
        assert status == seeded_status
        assert retry_count == seeded_retry_count
    finally:
        _delete_seeded_notification_records(event_id, correlation_id)


def test_post_notifications_retry_returns_400_when_status_is_sent(
    client: TestClient,
) -> None:
    correlation_id = f"notif-retry-sent-{uuid4().hex[:8]}"
    event_id = _insert_audit_event_for_outbox(correlation_id)
    _insert_outbox(event_id=event_id, status="sent", retry_count=1)

    try:
        seeded_status, seeded_retry_count, _, _ = _find_outbox_by_event_id(event_id)
        res = client.post(f"/governance/notifications/{event_id}/retry")

        assert res.status_code == 400
        body = res.json()
        assert body["ok"] is False
        assert body["error"]["code"] == "INVALID_RETRY_TARGET"
        assert body["error"]["details"]["current_status"] == "sent"
        status, retry_count, _, _ = _find_outbox_by_event_id(event_id)
        assert status == seeded_status
        assert retry_count == seeded_retry_count
    finally:
        _delete_seeded_notification_records(event_id, correlation_id)


def test_post_notifications_retry_returns_409_when_retry_limit_exceeded(
    client: TestClient,
) -> None:
    correlation_id = f"notif-retry-limit-{uuid4().hex[:8]}"
    event_id = _insert_audit_event_for_outbox(correlation_id)
    _insert_outbox(event_id=event_id, status="failed", retry_count=3)

    try:
        seeded_status, seeded_retry_count, _, _ = _find_outbox_by_event_id(event_id)
        res = client.post(f"/governance/notifications/{event_id}/retry")

        assert res.status_code == 409
        body = res.json()
        assert body["ok"] is False
        assert body["error"]["code"] == "RETRY_LIMIT_EXCEEDED"
        assert body["error"]["details"]["retry_count"] == "3"
        status, retry_count, _, _ = _find_outbox_by_event_id(event_id)
        assert status == seeded_status
        assert retry_count == seeded_retry_count
    finally:
        _delete_seeded_notification_records(event_id, correlation_id)
