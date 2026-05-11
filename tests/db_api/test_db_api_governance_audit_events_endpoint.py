from __future__ import annotations

from datetime import UTC, datetime
from uuid import uuid4

from fastapi.testclient import TestClient

from portfolio_fdc.db_api.db import MAIN_DB, _connect
from tests.utils.test_utils import assert_validation_error_envelope


def _insert_audit_event(
    *,
    event_type: str,
    actor: str,
    actor_role: str,
    target_type: str,
    target_id: int,
    occurred_at: str,
    correlation_id: str,
) -> int:
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
                event_type,
                actor,
                actor_role,
                target_type,
                target_id,
                occurred_at,
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


def _delete_audit_events(correlation_id: str) -> None:
    con = _connect(MAIN_DB)
    try:
        con.execute(
            "DELETE FROM GovernanceAuditEvents WHERE correlation_id = ?",
            (correlation_id,),
        )
        con.commit()
    finally:
        con.close()


def test_get_governance_audit_events_returns_empty_list_when_no_match(
    client: TestClient,
) -> None:
    res = client.get(
        "/governance/audit-events",
        params={"event_type": "no_such_event_type"},
    )

    assert res.status_code == 200
    body = res.json()
    assert body["ok"] is True
    assert body["data"] == []


def test_get_governance_audit_events_filters_by_event_type_and_target(
    client: TestClient,
) -> None:
    correlation_id = f"audit-filter-{uuid4().hex[:8]}"
    now = datetime.now(UTC)
    try:
        target_id = 22001
        expected_id = _insert_audit_event(
            event_type="change_applied",
            actor="ops-a",
            actor_role="ops",
            target_type="change_request",
            target_id=target_id,
            occurred_at=now.isoformat(),
            correlation_id=correlation_id,
        )
        _insert_audit_event(
            event_type="change_approved",
            actor="ops-b",
            actor_role="ops",
            target_type="change_request",
            target_id=target_id,
            occurred_at=now.isoformat(),
            correlation_id=correlation_id,
        )
        _insert_audit_event(
            event_type="change_applied",
            actor="ops-c",
            actor_role="ops",
            target_type="change_request",
            target_id=22002,
            occurred_at=now.isoformat(),
            correlation_id=correlation_id,
        )

        res = client.get(
            "/governance/audit-events",
            params={
                "event_type": "change_applied",
                "target_type": "change_request",
                "target_id": target_id,
            },
        )

        assert res.status_code == 200
        body = res.json()
        assert body["ok"] is True
        assert [row["id"] for row in body["data"]] == [expected_id]
    finally:
        _delete_audit_events(correlation_id)


def test_get_governance_audit_events_filters_by_from_to(
    client: TestClient,
) -> None:
    correlation_id = f"audit-time-{uuid4().hex[:8]}"
    try:
        _insert_audit_event(
            event_type="change_requested",
            actor="ops-a",
            actor_role="ops",
            target_type="change_request",
            target_id=23001,
            occurred_at="2026-05-01T00:00:00.000Z",
            correlation_id=correlation_id,
        )
        id_mid = _insert_audit_event(
            event_type="change_approved",
            actor="ops-b",
            actor_role="ops",
            target_type="change_request",
            target_id=23002,
            occurred_at="2026-05-02T00:00:00.000Z",
            correlation_id=correlation_id,
        )
        id_new = _insert_audit_event(
            event_type="change_applied",
            actor="ops-c",
            actor_role="ops",
            target_type="change_request",
            target_id=23003,
            occurred_at="2026-05-03T00:00:00.000Z",
            correlation_id=correlation_id,
        )

        res = client.get(
            "/governance/audit-events",
            params={
                "from_ts": "2026-05-02T00:00:00Z",
                "to_ts": "2026-05-03T00:00:00Z",
            },
        )

        assert res.status_code == 200
        body = res.json()
        assert body["ok"] is True
        assert [row["id"] for row in body["data"] if row["correlation_id"] == correlation_id] == [
            id_new,
            id_mid,
        ]
    finally:
        _delete_audit_events(correlation_id)


def test_get_governance_audit_events_supports_limit_and_offset(
    client: TestClient,
) -> None:
    correlation_id = f"audit-page-{uuid4().hex[:8]}"
    event_type = f"page_event_{uuid4().hex[:8]}"
    try:
        _insert_audit_event(
            event_type=event_type,
            actor="ops-a",
            actor_role="ops",
            target_type="change_request",
            target_id=24001,
            occurred_at="2026-05-01T00:00:00.000Z",
            correlation_id=correlation_id,
        )
        id_second = _insert_audit_event(
            event_type=event_type,
            actor="ops-b",
            actor_role="ops",
            target_type="change_request",
            target_id=24002,
            occurred_at="2026-05-02T00:00:00.000Z",
            correlation_id=correlation_id,
        )
        _insert_audit_event(
            event_type=event_type,
            actor="ops-c",
            actor_role="ops",
            target_type="change_request",
            target_id=24003,
            occurred_at="2026-05-03T00:00:00.000Z",
            correlation_id=correlation_id,
        )

        res = client.get(
            "/governance/audit-events",
            params={"event_type": event_type, "limit": 1, "offset": 1},
        )

        assert res.status_code == 200
        body = res.json()
        assert body["ok"] is True
        assert [row["id"] for row in body["data"]] == [id_second]
    finally:
        _delete_audit_events(correlation_id)


def test_get_governance_audit_events_returns_400_for_naive_from_ts(
    client: TestClient,
) -> None:
    res = client.get(
        "/governance/audit-events",
        params={
            "from_ts": "2026-05-01T00:00:00",
            "to_ts": "2026-05-02T00:00:00Z",
        },
    )

    assert res.status_code == 400
    assert "timezone" in res.json()["detail"]


def test_get_governance_audit_events_returns_400_for_naive_to_ts_only(
    client: TestClient,
) -> None:
    res = client.get(
        "/governance/audit-events",
        params={
            "to_ts": "2026-05-02T00:00:00",
        },
    )

    assert res.status_code == 400
    assert "timezone" in res.json()["detail"]


def test_get_governance_audit_events_returns_400_when_from_ts_is_after_to_ts(
    client: TestClient,
) -> None:
    res = client.get(
        "/governance/audit-events",
        params={
            "from_ts": "2026-05-03T00:00:00Z",
            "to_ts": "2026-05-02T00:00:00Z",
        },
    )

    assert res.status_code == 400
    assert "end_ts must be greater than or equal to start_ts" in res.json()["detail"]


def test_get_governance_audit_events_returns_422_for_invalid_limit(
    client: TestClient,
) -> None:
    res = client.get(
        "/governance/audit-events",
        params={
            "limit": 0,
        },
    )

    assert res.status_code == 422
    assert_validation_error_envelope(res.json(), expected_loc_fragment="limit")


def test_get_governance_audit_events_returns_422_for_invalid_offset(
    client: TestClient,
) -> None:
    res = client.get(
        "/governance/audit-events",
        params={
            "offset": -1,
        },
    )

    assert res.status_code == 422
    assert_validation_error_envelope(res.json(), expected_loc_fragment="offset")
