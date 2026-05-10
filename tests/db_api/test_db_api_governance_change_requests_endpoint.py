from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from datetime import UTC, datetime
from uuid import uuid4

import pytest
from fastapi.testclient import TestClient

from portfolio_fdc.db_api.db import MAIN_DB, _connect, _init_schema
from tests.utils.test_utils import assert_validation_error_envelope


@dataclass(frozen=True)
class SeededChangeRequestsContext:
    chart_1_id: int
    chart_2_id: int
    request_pending_chart_1: int
    request_approved_chart_1: int
    request_pending_chart_2: int


def _insert_chart_set(now: str, suffix: str) -> int:
    con = _connect(MAIN_DB)
    try:
        con.execute(
            "INSERT INTO ChartSet(name, note, created_at, created_by) VALUES (?, ?, ?, ?)",
            (f"governance_req_set_{suffix}", "test", now, "test"),
        )
        chart_set_id = int(con.execute("SELECT last_insert_rowid()").fetchone()[0])
        con.commit()
        return chart_set_id
    finally:
        con.close()


def _insert_chart(chart_set_id: int, suffix: str, parameter: str) -> int:
    con = _connect(MAIN_DB)
    try:
        cur = con.execute(
            """
            INSERT INTO ChartsV2(
                chart_set_id, tool_id, chamber_id, recipe_id, parameter,
                step_no, feature_type, warn_low, warn_high, crit_low, crit_high,
                updated_at, updated_by, update_reason, update_source
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                chart_set_id,
                f"TOOL_{suffix}",
                "CH1",
                f"RECIPE_{suffix}",
                parameter,
                1,
                "mean",
                1.0,
                2.0,
                0.8,
                2.2,
                "2026-05-01T00:00:00.000Z",
                "tester",
                "seed",
                "test",
            ),
        )
        con.commit()
        row_id = cur.lastrowid
        if row_id is None:
            raise RuntimeError("Failed to insert chart")
        return int(row_id)
    finally:
        con.close()


def _insert_change_request(
    *,
    chart_id: int,
    proposed_by: str,
    proposed_at: str,
    idempotency_key: str,
    change_payload: str = "{}",
    expected_version: int = 1,
) -> int:
    con = _connect(MAIN_DB)
    try:
        cur = con.execute(
            """
            INSERT INTO GovernanceChangeRequests(
                chart_id, proposed_by, proposed_at, change_payload,
                expected_version, idempotency_key
            ) VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                chart_id,
                proposed_by,
                proposed_at,
                change_payload,
                expected_version,
                idempotency_key,
            ),
        )
        con.commit()
        row_id = cur.lastrowid
        if row_id is None:
            raise RuntimeError("Failed to insert change request")
        return int(row_id)
    finally:
        con.close()


def _update_change_request_status(request_id: int, status: str) -> None:
    con = _connect(MAIN_DB)
    try:
        con.execute(
            "UPDATE GovernanceChangeRequests SET status = ? WHERE id = ?",
            (status, request_id),
        )
        con.commit()
    finally:
        con.close()


def _cleanup_seeded(chart_set_id: int | None, request_ids: list[int]) -> None:
    con = _connect(MAIN_DB)
    try:
        if request_ids:
            con.executemany(
                "DELETE FROM GovernanceApprovals WHERE request_id = ?",
                [(request_id,) for request_id in request_ids],
            )
            con.executemany(
                "DELETE FROM GovernanceApplyResults WHERE request_id = ?",
                [(request_id,) for request_id in request_ids],
            )
            con.executemany(
                "DELETE FROM GovernanceChangeRequests WHERE id = ?",
                [(request_id,) for request_id in request_ids],
            )
        if chart_set_id is not None:
            con.execute("DELETE FROM ChartsV2 WHERE chart_set_id = ?", (chart_set_id,))
            con.execute("DELETE FROM ChartSet WHERE chart_set_id = ?", (chart_set_id,))
        con.commit()
    finally:
        con.close()


def _delete_change_request_by_idempotency(idempotency_key: str) -> None:
    con = _connect(MAIN_DB)
    try:
        request_ids = [
            int(row[0])
            for row in con.execute(
                "SELECT id FROM GovernanceChangeRequests WHERE idempotency_key = ?",
                (idempotency_key,),
            ).fetchall()
        ]
        if request_ids:
            con.executemany(
                "DELETE FROM GovernanceApprovals WHERE request_id = ?",
                [(request_id,) for request_id in request_ids],
            )
            con.executemany(
                "DELETE FROM GovernanceApplyResults WHERE request_id = ?",
                [(request_id,) for request_id in request_ids],
            )
            con.executemany(
                "DELETE FROM GovernanceAuditEvents WHERE target_type = ? AND target_id = ?",
                [("change_request", request_id) for request_id in request_ids],
            )
        con.execute(
            "DELETE FROM GovernanceAuditEvents WHERE correlation_id = ?",
            (idempotency_key,),
        )
        con.execute(
            "DELETE FROM GovernanceChangeRequests WHERE idempotency_key = ?",
            (idempotency_key,),
        )
        con.commit()
    finally:
        con.close()


def _count_audit_events_by_correlation_id(correlation_id: str) -> int:
    con = _connect(MAIN_DB)
    try:
        row = con.execute(
            "SELECT COUNT(*) FROM GovernanceAuditEvents WHERE correlation_id = ?",
            (correlation_id,),
        ).fetchone()
        return int(row[0])
    finally:
        con.close()


def _find_latest_audit_event_by_correlation_id(correlation_id: str) -> tuple[str, str, int] | None:
    con = _connect(MAIN_DB)
    try:
        row = con.execute(
            """
            SELECT event_type, target_type, target_id
            FROM GovernanceAuditEvents
            WHERE correlation_id = ?
            ORDER BY id DESC
            LIMIT 1
            """,
            (correlation_id,),
        ).fetchone()
        if row is None:
            return None
        return (str(row[0]), str(row[1]), int(row[2]))
    finally:
        con.close()


def _find_change_request_status(request_id: int) -> str:
    con = _connect(MAIN_DB)
    try:
        row = con.execute(
            "SELECT status FROM GovernanceChangeRequests WHERE id = ?",
            (request_id,),
        ).fetchone()
        if row is None:
            raise RuntimeError("Change request not found")
        return str(row[0])
    finally:
        con.close()


def _find_approval_count(request_id: int) -> int:
    con = _connect(MAIN_DB)
    try:
        row = con.execute(
            "SELECT COUNT(*) FROM GovernanceApprovals WHERE request_id = ?",
            (request_id,),
        ).fetchone()
        return int(row[0])
    finally:
        con.close()


def _count_chart_history(chart_id: int) -> int:
    con = _connect(MAIN_DB)
    try:
        row = con.execute(
            "SELECT COUNT(*) FROM ChartsHistory WHERE chart_id = ?",
            (chart_id,),
        ).fetchone()
        return int(row[0])
    finally:
        con.close()


def _update_chart_version(chart_id: int, version: int) -> None:
    con = _connect(MAIN_DB)
    try:
        con.execute("UPDATE ChartsV2 SET version = ? WHERE id = ?", (version, chart_id))
        con.commit()
    finally:
        con.close()


def _force_change_request_chart_id(request_id: int, chart_id: int) -> None:
    con = _connect(MAIN_DB)
    con.execute("PRAGMA foreign_keys = OFF")
    try:
        con.execute(
            "UPDATE GovernanceChangeRequests SET chart_id = ? WHERE id = ?",
            (chart_id, request_id),
        )
        con.commit()
    finally:
        con.execute("PRAGMA foreign_keys = ON")
        con.close()


@pytest.fixture
def seeded_change_requests_context() -> Iterator[SeededChangeRequestsContext]:
    chart_set_id: int | None = None
    chart_1: int | None = None
    chart_2: int | None = None
    created_request_ids: list[int] = []

    try:
        _init_schema(MAIN_DB)
        suffix = uuid4().hex[:10]
        now = datetime.now(UTC).isoformat()
        chart_set_id = _insert_chart_set(now, suffix)
        chart_1 = _insert_chart(chart_set_id, suffix, "param_a")
        chart_2 = _insert_chart(chart_set_id, suffix, "param_b")

        req_pending_chart_1 = _insert_change_request(
            chart_id=chart_1,
            proposed_by="alice",
            proposed_at="2026-05-01T00:00:00.000Z",
            idempotency_key=f"{suffix}-k1",
        )
        created_request_ids.append(req_pending_chart_1)
        req_approved_chart_1 = _insert_change_request(
            chart_id=chart_1,
            proposed_by="bob",
            proposed_at="2026-05-02T00:00:00.000Z",
            idempotency_key=f"{suffix}-k2",
        )
        created_request_ids.append(req_approved_chart_1)
        req_pending_chart_2 = _insert_change_request(
            chart_id=chart_2,
            proposed_by="carol",
            proposed_at="2026-05-03T00:00:00.000Z",
            idempotency_key=f"{suffix}-k3",
        )
        created_request_ids.append(req_pending_chart_2)
        _update_change_request_status(req_approved_chart_1, "approved")

        yield SeededChangeRequestsContext(
            chart_1_id=chart_1,
            chart_2_id=chart_2,
            request_pending_chart_1=req_pending_chart_1,
            request_approved_chart_1=req_approved_chart_1,
            request_pending_chart_2=req_pending_chart_2,
        )
    finally:
        _cleanup_seeded(chart_set_id, created_request_ids)


def test_get_change_requests_returns_empty_list_when_no_match(client: TestClient) -> None:
    res = client.get(
        "/governance/change-requests",
        params={"chart_id": 9223372036854775807},
    )

    assert res.status_code == 200
    body = res.json()
    assert body["ok"] is True
    assert body["data"] == []


def test_get_change_requests_returns_envelope_contract(
    client: TestClient,
    seeded_change_requests_context: SeededChangeRequestsContext,
) -> None:
    seeded = seeded_change_requests_context

    res = client.get(
        "/governance/change-requests",
        params={"chart_id": seeded.chart_1_id},
    )

    assert res.status_code == 200
    body = res.json()
    assert body["ok"] is True
    assert isinstance(body["data"], list)
    assert body["data"]
    proposed_at_values = [row["proposed_at"] for row in body["data"]]
    assert "2026-05-02T00:00:00.000Z" in proposed_at_values
    for proposed_at in proposed_at_values:
        assert isinstance(proposed_at, str)
        parsed = datetime.fromisoformat(proposed_at.replace("Z", "+00:00"))
        assert parsed.tzinfo is not None


def test_get_change_requests_filters_by_status(
    client: TestClient,
    seeded_change_requests_context: SeededChangeRequestsContext,
) -> None:
    seeded = seeded_change_requests_context

    res = client.get(
        "/governance/change-requests",
        params={"status": "approved", "chart_id": seeded.chart_1_id},
    )

    assert res.status_code == 200
    rows = res.json()["data"]
    assert [row["id"] for row in rows] == [seeded.request_approved_chart_1]
    assert all(row["status"] == "approved" for row in rows)


def test_get_change_requests_filters_by_chart_id(
    client: TestClient,
    seeded_change_requests_context: SeededChangeRequestsContext,
) -> None:
    seeded = seeded_change_requests_context

    res = client.get(
        "/governance/change-requests",
        params={"chart_id": seeded.chart_2_id},
    )

    assert res.status_code == 200
    rows = res.json()["data"]
    assert [row["id"] for row in rows] == [seeded.request_pending_chart_2]


def test_get_change_requests_filters_by_from_to(
    client: TestClient,
    seeded_change_requests_context: SeededChangeRequestsContext,
) -> None:
    seeded = seeded_change_requests_context

    res = client.get(
        "/governance/change-requests",
        params={
            "from_ts": "2026-05-02T00:00:00Z",
            "to_ts": "2026-05-03T00:00:00Z",
        },
    )

    assert res.status_code == 200
    rows = res.json()["data"]
    assert [row["id"] for row in rows] == [
        seeded.request_pending_chart_2,
        seeded.request_approved_chart_1,
    ]


def test_get_change_requests_invalid_datetime_returns_422(client: TestClient) -> None:
    res = client.get(
        "/governance/change-requests",
        params={
            "from_ts": "invalid-date",
            "to_ts": "2026-13-01T00:00:00Z",
        },
    )

    assert res.status_code == 422
    assert_validation_error_envelope(
        res.json(),
        expected_loc_fragment="from_ts",
    )


def test_get_change_requests_applies_limit(
    client: TestClient,
    seeded_change_requests_context: SeededChangeRequestsContext,
) -> None:
    seeded = seeded_change_requests_context

    res = client.get(
        "/governance/change-requests",
        params={"chart_id": seeded.chart_1_id, "limit": 1},
    )

    assert res.status_code == 200
    rows = res.json()["data"]
    assert len(rows) == 1


def test_get_change_requests_applies_offset(
    client: TestClient,
    seeded_change_requests_context: SeededChangeRequestsContext,
) -> None:
    seeded = seeded_change_requests_context

    res = client.get(
        "/governance/change-requests",
        params={"chart_id": seeded.chart_1_id, "offset": 1},
    )

    assert res.status_code == 200
    rows = res.json()["data"]
    assert [row["id"] for row in rows] == [seeded.request_pending_chart_1]


def test_get_change_requests_limit_exceeds_max(
    client: TestClient,
    seeded_change_requests_context: SeededChangeRequestsContext,
) -> None:
    seeded = seeded_change_requests_context

    res = client.get(
        "/governance/change-requests",
        params={"chart_id": seeded.chart_1_id, "limit": 501},
    )

    assert res.status_code == 422
    assert_validation_error_envelope(
        res.json(),
        expected_loc_fragment="limit",
        expected_message_fragment="less than or equal to 500",
    )


def test_get_change_requests_negative_offset(
    client: TestClient,
    seeded_change_requests_context: SeededChangeRequestsContext,
) -> None:
    seeded = seeded_change_requests_context

    res = client.get(
        "/governance/change-requests",
        params={"chart_id": seeded.chart_1_id, "offset": -1},
    )

    assert res.status_code == 422
    assert_validation_error_envelope(
        res.json(),
        expected_loc_fragment="offset",
        expected_message_fragment="greater than or equal to 0",
    )


def test_post_change_requests_success_returns_request_id_and_pending_status(
    client: TestClient,
    seeded_change_requests_context: SeededChangeRequestsContext,
) -> None:
    seeded = seeded_change_requests_context
    idempotency_key = f"post-ok-{uuid4().hex[:12]}"
    try:
        res = client.post(
            "/governance/change-requests",
            json={
                "chart_id": seeded.chart_1_id,
                "proposed_by": "tester",
                "change_payload": '{"warning_ucl": 2.5}',
                "expected_version": 1,
                "idempotency_key": idempotency_key,
            },
        )

        assert res.status_code == 200
        body = res.json()
        assert body["ok"] is True
        assert isinstance(body["data"]["request_id"], int)
        assert body["data"]["status"] == "pending"
    finally:
        _delete_change_request_by_idempotency(idempotency_key)


def test_post_approve_change_requests_success_transitions_to_approved(
    client: TestClient,
    seeded_change_requests_context: SeededChangeRequestsContext,
) -> None:
    seeded = seeded_change_requests_context
    request_id = seeded.request_pending_chart_1

    res = client.post(
        f"/governance/change-requests/{request_id}/approve",
        json={
            "approved_by": "ops-user",
            "approved_by_role": "ops",
            "comment": "looks good",
        },
    )

    assert res.status_code == 200
    body = res.json()
    assert body["ok"] is True
    assert body["data"]["request_id"] == request_id
    assert body["data"]["status"] == "approved"
    assert _find_change_request_status(request_id) == "approved"
    assert _find_approval_count(request_id) == 1


def test_post_approve_change_requests_returns_404_when_request_not_found(
    client: TestClient,
) -> None:
    request_id = 9223372036854775807

    res = client.post(
        f"/governance/change-requests/{request_id}/approve",
        json={
            "approved_by": "ops-user",
            "approved_by_role": "ops",
        },
    )

    assert res.status_code == 404
    body = res.json()
    assert body["ok"] is False
    assert body["error"]["code"] == "NOT_FOUND"
    assert body["error"]["details"]["request_id"] == str(request_id)


def test_post_approve_change_requests_returns_409_for_duplicate_approval(
    client: TestClient,
    seeded_change_requests_context: SeededChangeRequestsContext,
) -> None:
    seeded = seeded_change_requests_context
    request_id = seeded.request_approved_chart_1

    res = client.post(
        f"/governance/change-requests/{request_id}/approve",
        json={
            "approved_by": "ops-user",
            "approved_by_role": "ops",
            "comment": "duplicate approval",
        },
    )

    assert res.status_code == 409
    body = res.json()
    assert body["ok"] is False
    assert body["error"]["code"] == "ALREADY_APPROVED"
    assert body["error"]["details"]["request_id"] == str(request_id)


def test_post_approve_change_requests_success_envelope_contract(
    client: TestClient,
    seeded_change_requests_context: SeededChangeRequestsContext,
) -> None:
    seeded = seeded_change_requests_context
    idempotency_key = f"approve-envelope-{uuid4().hex[:12]}"

    request_id = _insert_change_request(
        chart_id=seeded.chart_2_id,
        proposed_by="ops",
        proposed_at="2026-05-04T00:00:00.000Z",
        idempotency_key=idempotency_key,
        change_payload="{}",
        expected_version=1,
    )
    try:
        res = client.post(
            f"/governance/change-requests/{request_id}/approve",
            json={
                "approved_by": "tester",
                "approved_by_role": "ops",
                "comment": "test approve contract",
            },
        )

        assert res.status_code == 200
        body = res.json()
        assert body["ok"] is True
        assert "data" in body
        assert "timestamp" in body
    finally:
        _delete_change_request_by_idempotency(idempotency_key)


def test_post_apply_change_requests_success_noop_does_not_add_history(
    client: TestClient,
    seeded_change_requests_context: SeededChangeRequestsContext,
) -> None:
    seeded = seeded_change_requests_context
    request_id = seeded.request_approved_chart_1
    chart_id = seeded.chart_1_id
    before_history = _count_chart_history(chart_id)

    res = client.post(
        f"/governance/change-requests/{request_id}/apply",
        json={
            "applied_by": "ops-user",
            "applied_by_role": "ops",
            "reason": "apply no-op",
        },
    )

    assert res.status_code == 200
    body = res.json()
    assert body["ok"] is True
    assert body["data"]["request_id"] == request_id
    assert body["data"]["status"] == "applied"
    assert body["data"]["noop"] is True
    assert _find_change_request_status(request_id) == "applied"
    assert _count_chart_history(chart_id) == before_history


def test_post_apply_change_requests_success_with_threshold_change(
    client: TestClient,
    seeded_change_requests_context: SeededChangeRequestsContext,
) -> None:
    seeded = seeded_change_requests_context
    chart_id = seeded.chart_1_id
    idempotency_key = f"test-threshold-change-{uuid4().hex[:8]}"
    previous_version = 1  # ChartsV2 version DEFAULT 1

    request_id = _insert_change_request(
        chart_id=chart_id,
        proposed_by="test-user",
        proposed_at="2026-05-10T00:00:00.000Z",
        idempotency_key=idempotency_key,
        change_payload='{"warn_low": 1.5}',
        expected_version=previous_version,
    )
    _update_change_request_status(request_id, "approved")

    try:
        before_history = _count_chart_history(chart_id)

        res = client.post(
            f"/governance/change-requests/{request_id}/apply",
            json={
                "applied_by": "ops-user",
                "applied_by_role": "ops",
                "reason": "apply threshold change",
            },
        )

        assert res.status_code == 200
        body = res.json()
        assert body["ok"] is True
        assert body["data"]["request_id"] == request_id
        assert body["data"]["noop"] is False
        assert body["data"]["status"] == "applied"
        assert _find_change_request_status(request_id) == "applied"
        assert _count_chart_history(chart_id) == before_history + 1
        assert body["data"]["resulting_version"] > previous_version
    finally:
        con = _connect(MAIN_DB)
        try:
            con.execute("DELETE FROM ChartsHistory WHERE chart_id = ?", (chart_id,))
            con.commit()
        finally:
            con.close()
        _cleanup_seeded(None, [request_id])


def test_post_apply_change_requests_returns_404_when_request_not_found(
    client: TestClient,
) -> None:
    request_id = 9223372036854775807
    res = client.post(
        f"/governance/change-requests/{request_id}/apply",
        json={
            "applied_by": "ops-user",
            "applied_by_role": "ops",
        },
    )

    assert res.status_code == 404
    body = res.json()
    assert body["ok"] is False
    assert body["error"]["code"] == "NOT_FOUND"
    assert body["error"]["details"]["request_id"] == str(request_id)


def test_post_apply_change_requests_returns_409_when_not_approved(
    client: TestClient,
    seeded_change_requests_context: SeededChangeRequestsContext,
) -> None:
    seeded = seeded_change_requests_context
    request_id = seeded.request_pending_chart_1
    res = client.post(
        f"/governance/change-requests/{request_id}/apply",
        json={
            "applied_by": "ops-user",
            "applied_by_role": "ops",
        },
    )

    assert res.status_code == 409
    body = res.json()
    assert body["ok"] is False
    assert body["error"]["code"] == "INVALID_STATUS_TRANSITION"


def test_post_apply_change_requests_returns_409_for_stale_expected_version(
    client: TestClient,
    seeded_change_requests_context: SeededChangeRequestsContext,
) -> None:
    seeded = seeded_change_requests_context
    _update_chart_version(seeded.chart_1_id, 3)
    before_history = _count_chart_history(seeded.chart_1_id)

    request_id = seeded.request_approved_chart_1
    res = client.post(
        f"/governance/change-requests/{request_id}/apply",
        json={
            "applied_by": "ops-user",
            "applied_by_role": "ops",
        },
    )

    assert res.status_code == 409
    body = res.json()
    assert body["ok"] is False
    assert body["error"]["code"] == "STALE_EXPECTED_VERSION"
    current = body["error"]["details"]["current"]
    assert current["chart_id"] == seeded.chart_1_id
    assert current["version"] == 3
    assert _count_chart_history(seeded.chart_1_id) == before_history


def test_post_apply_change_requests_returns_422_for_invalid_threshold_consistency(
    client: TestClient,
    seeded_change_requests_context: SeededChangeRequestsContext,
) -> None:
    seeded = seeded_change_requests_context
    request_id = _insert_change_request(
        chart_id=seeded.chart_2_id,
        proposed_by="ops",
        proposed_at="2026-05-04T00:00:00.000Z",
        idempotency_key=f"apply-invalid-{uuid4().hex[:12]}",
        change_payload='{"warn_low": 5.0, "warn_high": 2.0}',
        expected_version=1,
    )
    _update_change_request_status(request_id, "approved")
    try:
        res = client.post(
            f"/governance/change-requests/{request_id}/apply",
            json={
                "applied_by": "ops-user",
                "applied_by_role": "ops",
            },
        )
        assert res.status_code == 422
        assert_validation_error_envelope(
            res.json(),
            expected_loc_fragment="change_payload",
            expected_message_fragment="warn_low",
        )
    finally:
        con = _connect(MAIN_DB)
        try:
            con.execute("DELETE FROM GovernanceApplyResults WHERE request_id = ?", (request_id,))
            con.execute("DELETE FROM GovernanceApprovals WHERE request_id = ?", (request_id,))
            con.execute(
                "DELETE FROM GovernanceAuditEvents WHERE target_type = ? AND target_id = ?",
                ("change_request", request_id),
            )
            con.execute("DELETE FROM GovernanceChangeRequests WHERE id = ?", (request_id,))
            con.commit()
        finally:
            con.close()


def test_post_apply_change_requests_returns_400_when_chart_missing(
    client: TestClient,
    seeded_change_requests_context: SeededChangeRequestsContext,
) -> None:
    seeded = seeded_change_requests_context
    request_id = seeded.request_approved_chart_1
    _force_change_request_chart_id(request_id, 9223372036854775701)

    res = client.post(
        f"/governance/change-requests/{request_id}/apply",
        json={
            "applied_by": "ops-user",
            "applied_by_role": "ops",
        },
    )

    assert res.status_code == 400
    body = res.json()
    assert body["ok"] is False
    assert body["error"]["code"] == "CHART_NOT_FOUND"


def test_post_apply_change_requests_success_envelope_contract(
    client: TestClient,
    seeded_change_requests_context: SeededChangeRequestsContext,
) -> None:
    seeded = seeded_change_requests_context
    idempotency_key = f"apply-envelope-{uuid4().hex[:12]}"

    request_id = _insert_change_request(
        chart_id=seeded.chart_1_id,
        proposed_by="ops",
        proposed_at="2026-05-04T00:00:00.000Z",
        idempotency_key=idempotency_key,
        change_payload='{"warn_low": 1.0}',
        expected_version=1,
    )
    _update_change_request_status(request_id, "approved")
    try:
        res = client.post(
            f"/governance/change-requests/{request_id}/apply",
            json={
                "applied_by": "tester",
                "applied_by_role": "ops",
            },
        )

        assert res.status_code == 200
        body = res.json()
        assert body["ok"] is True
        assert "data" in body
        assert "timestamp" in body
    finally:
        _delete_change_request_by_idempotency(idempotency_key)


def test_post_change_requests_success_envelope_contract(
    client: TestClient,
    seeded_change_requests_context: SeededChangeRequestsContext,
) -> None:
    seeded = seeded_change_requests_context
    idempotency_key = f"post-envelope-{uuid4().hex[:12]}"
    try:
        res = client.post(
            "/governance/change-requests",
            json={
                "chart_id": seeded.chart_1_id,
                "proposed_by": "tester",
                "change_payload": "{}",
                "expected_version": 1,
                "idempotency_key": idempotency_key,
            },
        )

        assert res.status_code == 200
        body = res.json()
        assert body["ok"] is True
        assert "data" in body
    finally:
        _delete_change_request_by_idempotency(idempotency_key)


def test_post_change_requests_returns_422_for_missing_required_field(
    client: TestClient,
    seeded_change_requests_context: SeededChangeRequestsContext,
) -> None:
    seeded = seeded_change_requests_context

    res = client.post(
        "/governance/change-requests",
        json={
            "chart_id": seeded.chart_1_id,
            "proposed_by": "tester",
            "change_payload": "{}",
            "expected_version": 1,
        },
    )

    assert res.status_code == 422
    assert_validation_error_envelope(
        res.json(),
        expected_loc_fragment="idempotency_key",
    )


def test_post_change_requests_returns_422_for_non_integer_chart_id(client: TestClient) -> None:
    res = client.post(
        "/governance/change-requests",
        json={
            "chart_id": "abc",
            "proposed_by": "tester",
            "change_payload": "{}",
            "expected_version": 1,
            "idempotency_key": f"post-invalid-chart-{uuid4().hex[:8]}",
        },
    )

    assert res.status_code == 422
    assert_validation_error_envelope(
        res.json(),
        expected_loc_fragment="chart_id",
    )


def test_post_change_requests_returns_409_for_duplicate_idempotency_key(
    client: TestClient,
    seeded_change_requests_context: SeededChangeRequestsContext,
) -> None:
    seeded = seeded_change_requests_context
    idempotency_key = f"post-dup-{uuid4().hex[:12]}"
    try:
        first = client.post(
            "/governance/change-requests",
            json={
                "chart_id": seeded.chart_1_id,
                "proposed_by": "tester",
                "change_payload": "{}",
                "expected_version": 1,
                "idempotency_key": idempotency_key,
            },
        )
        assert first.status_code == 200

        second = client.post(
            "/governance/change-requests",
            json={
                "chart_id": seeded.chart_1_id,
                "proposed_by": "tester",
                "change_payload": "{}",
                "expected_version": 1,
                "idempotency_key": idempotency_key,
            },
        )

        assert second.status_code == 409
        body = second.json()
        assert body["ok"] is False
        assert body["error"]["code"] == "DUPLICATE_IDEMPOTENCY_KEY"
        assert body["error"]["details"]["idempotency_key"] == idempotency_key
    finally:
        _delete_change_request_by_idempotency(idempotency_key)


def test_post_change_requests_accepts_missing_chart_id_for_apply_phase_validation(
    client: TestClient,
) -> None:
    idempotency_key = f"post-missing-chart-{uuid4().hex[:12]}"
    try:
        res = client.post(
            "/governance/change-requests",
            json={
                "chart_id": 9223372036854775807,
                "proposed_by": "tester",
                "change_payload": "{}",
                "expected_version": 1,
                "idempotency_key": idempotency_key,
            },
        )

        assert res.status_code == 422
        assert_validation_error_envelope(
            res.json(),
            expected_loc_fragment="chart_id",
            expected_message_fragment="existing chart",
        )
    finally:
        _delete_change_request_by_idempotency(idempotency_key)


def test_post_change_requests_writes_single_audit_event(
    client: TestClient,
    seeded_change_requests_context: SeededChangeRequestsContext,
) -> None:
    seeded = seeded_change_requests_context
    idempotency_key = f"post-audit-count-{uuid4().hex[:12]}"
    try:
        before_count = _count_audit_events_by_correlation_id(idempotency_key)
        assert before_count == 0

        res = client.post(
            "/governance/change-requests",
            json={
                "chart_id": seeded.chart_1_id,
                "proposed_by": "tester",
                "change_payload": "{}",
                "expected_version": 1,
                "idempotency_key": idempotency_key,
            },
        )
        assert res.status_code == 200

        after_count = _count_audit_events_by_correlation_id(idempotency_key)
        assert after_count == 1
    finally:
        _delete_change_request_by_idempotency(idempotency_key)


def test_post_change_requests_audit_event_type_is_change_requested(
    client: TestClient,
    seeded_change_requests_context: SeededChangeRequestsContext,
) -> None:
    seeded = seeded_change_requests_context
    idempotency_key = f"post-audit-type-{uuid4().hex[:12]}"
    try:
        res = client.post(
            "/governance/change-requests",
            json={
                "chart_id": seeded.chart_1_id,
                "proposed_by": "tester",
                "change_payload": "{}",
                "expected_version": 1,
                "idempotency_key": idempotency_key,
            },
        )
        assert res.status_code == 200
        request_id = int(res.json()["data"]["request_id"])

        latest_event = _find_latest_audit_event_by_correlation_id(idempotency_key)
        assert latest_event is not None
        event_type, target_type, target_id = latest_event
        assert event_type == "change_requested"
        assert target_type == "change_request"
        assert target_id == request_id
    finally:
        _delete_change_request_by_idempotency(idempotency_key)
