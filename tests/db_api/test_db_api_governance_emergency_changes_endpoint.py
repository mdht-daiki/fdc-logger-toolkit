from __future__ import annotations

from collections.abc import Iterator
from datetime import UTC, datetime
from uuid import uuid4

import pytest
from fastapi.testclient import TestClient

from portfolio_fdc.db_api.db import MAIN_DB, _connect, _init_schema
from tests.utils.test_utils import assert_validation_error_envelope


def _insert_chart_set(now: str, suffix: str) -> int:
    con = _connect(MAIN_DB)
    try:
        con.execute(
            "INSERT INTO ChartSet(name, note, created_at, created_by) VALUES (?, ?, ?, ?)",
            (f"emergency_set_{suffix}", "test", now, "test"),
        )
        chart_set_id = int(con.execute("SELECT last_insert_rowid()").fetchone()[0])
        con.commit()
        return chart_set_id
    finally:
        con.close()


def _insert_chart(chart_set_id: int, suffix: str) -> int:
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
                f"TOOL_EMERGENCY_{suffix}",
                "CH_EMERGENCY",
                f"RECIPE_EMERGENCY_{suffix}",
                "dc_bias",
                1,
                "mean",
                1.0,
                2.0,
                0.8,
                2.2,
                "2026-05-10T00:00:00.000Z",
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


def _find_chart_row(chart_id: int) -> tuple[float, float, float, float, int, str, str, str]:
    con = _connect(MAIN_DB)
    try:
        row = con.execute(
            """
            SELECT warn_low, warn_high, crit_low, crit_high, version,
                   updated_by, update_reason, update_source
            FROM ChartsV2
            WHERE id = ?
            """,
            (chart_id,),
        ).fetchone()
        if row is None:
            raise RuntimeError("chart not found")
        return (
            float(row[0]),
            float(row[1]),
            float(row[2]),
            float(row[3]),
            int(row[4]),
            str(row[5]),
            str(row[6]),
            str(row[7]),
        )
    finally:
        con.close()


def _find_latest_history_row(chart_id: int) -> tuple[str, str, str]:
    con = _connect(MAIN_DB)
    try:
        row = con.execute(
            """
            SELECT change_source, change_reason, changed_by
            FROM ChartsHistory
            WHERE chart_id = ?
            ORDER BY id DESC
            LIMIT 1
            """,
            (chart_id,),
        ).fetchone()
        if row is None:
            raise RuntimeError("history not found")
        return (str(row[0]), str(row[1]), str(row[2]))
    finally:
        con.close()


def _find_emergency_change(emergency_change_id: int) -> tuple[int, str | None]:
    con = _connect(MAIN_DB)
    try:
        row = con.execute(
            """
            SELECT resulting_version, related_issue_or_pr
            FROM GovernanceEmergencyChanges
            WHERE id = ?
            """,
            (emergency_change_id,),
        ).fetchone()
        if row is None:
            raise RuntimeError("emergency change not found")
        return (int(row[0]), None if row[1] is None else str(row[1]))
    finally:
        con.close()


def _find_ratification_row(ec_id: int) -> tuple[str, str | None, str | None]:
    con = _connect(MAIN_DB)
    try:
        row = con.execute(
            """
            SELECT ratified_by_role, ratification_comment, related_pr
            FROM GovernanceRatifications
            WHERE ec_id = ?
            """,
            (ec_id,),
        ).fetchone()
        if row is None:
            raise RuntimeError("ratification not found")
        return (
            str(row[0]),
            None if row[1] is None else str(row[1]),
            None if row[2] is None else str(row[2]),
        )
    finally:
        con.close()


def _find_emergency_audit_event_type(ec_id: int) -> str:
    con = _connect(MAIN_DB)
    try:
        row = con.execute(
            """
            SELECT event_type
            FROM GovernanceAuditEvents
            WHERE target_type = 'emergency_change' AND target_id = ?
            ORDER BY id DESC
            LIMIT 1
            """,
            (ec_id,),
        ).fetchone()
        if row is None:
            raise RuntimeError("audit event not found")
        return str(row[0])
    finally:
        con.close()


def _find_notification_outbox_status(ec_id: int) -> tuple[str, int]:
    con = _connect(MAIN_DB)
    try:
        row = con.execute(
            """
            SELECT outbox.status, outbox.retry_count
            FROM GovernanceNotificationOutbox AS outbox
            JOIN GovernanceAuditEvents AS ae
              ON ae.id = outbox.event_id
            WHERE ae.target_type = 'emergency_change'
              AND ae.target_id = ?
            ORDER BY outbox.id DESC
            LIMIT 1
            """,
            (ec_id,),
        ).fetchone()
        if row is None:
            raise RuntimeError("notification outbox not found")
        return (str(row[0]), int(row[1]))
    finally:
        con.close()


def _cleanup_emergency_records(chart_set_id: int, chart_id: int, ec_id: int | None) -> None:
    con = _connect(MAIN_DB)
    try:
        if ec_id is not None:
            con.execute("DELETE FROM GovernanceRatifications WHERE ec_id = ?", (ec_id,))
            con.execute(
                """
                DELETE FROM GovernanceNotificationOutbox
                WHERE event_id IN (
                    SELECT id FROM GovernanceAuditEvents
                    WHERE target_type = 'emergency_change' AND target_id = ?
                )
                """,
                (ec_id,),
            )
            con.execute(
                "DELETE FROM GovernanceAuditEvents "
                "WHERE target_type = 'emergency_change' AND target_id = ?",
                (ec_id,),
            )
            con.execute("DELETE FROM GovernanceEmergencyChanges WHERE id = ?", (ec_id,))
        con.execute("DELETE FROM ChartsHistory WHERE chart_id = ?", (chart_id,))
        con.execute("DELETE FROM ChartsV2 WHERE id = ?", (chart_id,))
        con.execute("DELETE FROM ChartSet WHERE chart_set_id = ?", (chart_set_id,))
        con.commit()
    finally:
        con.close()


@pytest.fixture
def seeded_emergency_chart() -> Iterator[tuple[int, int]]:
    chart_set_id: int | None = None
    chart_id: int | None = None

    try:
        _init_schema(MAIN_DB)
        suffix = uuid4().hex[:10]
        now = datetime.now(UTC).isoformat()
        chart_set_id = _insert_chart_set(now, suffix)
        chart_id = _insert_chart(chart_set_id, suffix)
        yield chart_set_id, chart_id
    finally:
        if chart_set_id is not None and chart_id is not None:
            con = _connect(MAIN_DB)
            try:
                emergency_row = con.execute(
                    "SELECT id FROM GovernanceEmergencyChanges "
                    "WHERE chart_id = ? ORDER BY id DESC LIMIT 1",
                    (chart_id,),
                ).fetchone()
                ec_id = None if emergency_row is None else int(emergency_row[0])
            finally:
                con.close()
            _cleanup_emergency_records(chart_set_id, chart_id, ec_id)


def test_post_emergency_changes_success_updates_chart_and_writes_audit(
    client: TestClient,
    seeded_emergency_chart: tuple[int, int],
) -> None:
    _, chart_id = seeded_emergency_chart
    res = client.post(
        "/governance/emergency-changes",
        json={
            "chart_id": chart_id,
            "changed_by": "ops-user",
            "changed_by_role": "operator",
            "reason": "incident mitigation",
            "change_payload": '{"warn_high": 1.9, "crit_high": 2.0}',
        },
    )

    assert res.status_code == 200
    body = res.json()
    assert body["ok"] is True
    assert isinstance(body["data"]["request_id"], int)
    assert body["data"]["status"] == "applied"
    assert body["data"]["resulting_version"] == 2
    assert body["data"]["noop"] is False

    ec_id = int(body["data"]["request_id"])
    (
        warn_low,
        warn_high,
        crit_low,
        crit_high,
        version,
        updated_by,
        update_reason,
        update_source,
    ) = _find_chart_row(chart_id)
    assert warn_low == 1.0
    assert warn_high == 1.9
    assert crit_low == 0.8
    assert crit_high == 2.0
    assert version == 2
    assert updated_by == "ops-user"
    assert update_reason == "incident mitigation"
    assert update_source == "emergency_manual"

    change_version, related_issue_or_pr = _find_emergency_change(ec_id)
    assert change_version == 2
    assert related_issue_or_pr is None
    assert _find_latest_history_row(chart_id) == (
        "emergency_manual",
        "incident mitigation",
        "ops-user",
    )
    assert _find_emergency_audit_event_type(ec_id) == "emergency_changed"
    assert _find_notification_outbox_status(ec_id) == ("pending", 0)


def test_post_emergency_changes_returns_404_when_chart_not_found(client: TestClient) -> None:
    res = client.post(
        "/governance/emergency-changes",
        json={
            "chart_id": 9223372036854775807,
            "changed_by": "ops-user",
            "changed_by_role": "operator",
            "reason": "incident mitigation",
            "change_payload": '{"warn_high": 1.9}',
        },
    )

    assert res.status_code == 404
    body = res.json()
    assert body["ok"] is False
    assert body["error"]["code"] == "CHART_NOT_FOUND"


def test_post_emergency_changes_returns_422_for_invalid_change_payload(
    client: TestClient,
    seeded_emergency_chart: tuple[int, int],
) -> None:
    _, chart_id = seeded_emergency_chart
    res = client.post(
        "/governance/emergency-changes",
        json={
            "chart_id": chart_id,
            "changed_by": "ops-user",
            "changed_by_role": "operator",
            "reason": "incident mitigation",
            "change_payload": "not-json",
        },
    )

    assert res.status_code == 422
    assert_validation_error_envelope(
        res.json(),
        expected_loc_fragment="change_payload",
        expected_message_fragment="valid JSON",
    )


def test_post_emergency_changes_returns_422_for_empty_change_payload(
    client: TestClient,
    seeded_emergency_chart: tuple[int, int],
) -> None:
    _, chart_id = seeded_emergency_chart
    res = client.post(
        "/governance/emergency-changes",
        json={
            "chart_id": chart_id,
            "changed_by": "ops-user",
            "changed_by_role": "operator",
            "reason": "incident mitigation",
            "change_payload": "{}",
        },
    )

    assert res.status_code == 422
    assert_validation_error_envelope(
        res.json(),
        expected_loc_fragment="change_payload",
        expected_message_fragment="must contain at least one of",
    )


def test_post_emergency_changes_returns_422_for_typo_only_change_payload(
    client: TestClient,
    seeded_emergency_chart: tuple[int, int],
) -> None:
    _, chart_id = seeded_emergency_chart
    res = client.post(
        "/governance/emergency-changes",
        json={
            "chart_id": chart_id,
            "changed_by": "ops-user",
            "changed_by_role": "operator",
            "reason": "incident mitigation",
            "change_payload": '{"typo_key": 1.5, "invalid_field": 2.0}',
        },
    )

    assert res.status_code == 422
    assert_validation_error_envelope(
        res.json(),
        expected_loc_fragment="change_payload",
        expected_message_fragment="must contain at least one of",
    )


def test_post_emergency_changes_ratify_success_updates_related_issue_or_pr(
    client: TestClient,
    seeded_emergency_chart: tuple[int, int],
) -> None:
    _, chart_id = seeded_emergency_chart
    create_res = client.post(
        "/governance/emergency-changes",
        json={
            "chart_id": chart_id,
            "changed_by": "ops-user",
            "changed_by_role": "operator",
            "reason": "incident mitigation",
            "change_payload": '{"warn_high": 1.9, "crit_high": 2.0}',
        },
    )
    ec_id = int(create_res.json()["data"]["request_id"])

    res = client.post(
        f"/governance/emergency-changes/{ec_id}/ratify",
        json={
            "ratified_by_role": "manager",
            "ratification_comment": "reviewed after the incident",
            "related_pr": "PR-123",
        },
    )

    assert res.status_code == 200
    body = res.json()
    assert body["ok"] is True
    assert body["data"]["request_id"] == ec_id
    assert body["data"]["status"] == "ratified"

    ratified_by_role, ratification_comment, related_pr = _find_ratification_row(ec_id)
    assert ratified_by_role == "manager"
    assert ratification_comment == "reviewed after the incident"
    assert related_pr == "PR-123"

    change_version, related_issue_or_pr = _find_emergency_change(ec_id)
    assert change_version == 2
    assert related_issue_or_pr == "PR-123"
    assert _find_emergency_audit_event_type(ec_id) == "emergency_ratified"


def test_post_emergency_changes_ratify_returns_404_when_not_found(client: TestClient) -> None:
    res = client.post(
        "/governance/emergency-changes/9223372036854775807/ratify",
        json={
            "ratified_by_role": "manager",
        },
    )

    assert res.status_code == 404
    body = res.json()
    assert body["ok"] is False
    assert body["error"]["code"] == "NOT_FOUND"


def test_post_emergency_changes_ratify_returns_409_when_already_ratified(
    client: TestClient,
    seeded_emergency_chart: tuple[int, int],
) -> None:
    _, chart_id = seeded_emergency_chart
    create_res = client.post(
        "/governance/emergency-changes",
        json={
            "chart_id": chart_id,
            "changed_by": "ops-user",
            "changed_by_role": "operator",
            "reason": "incident mitigation",
            "change_payload": '{"warn_high": 1.9, "crit_high": 2.0}',
        },
    )
    ec_id = int(create_res.json()["data"]["request_id"])

    first_res = client.post(
        f"/governance/emergency-changes/{ec_id}/ratify",
        json={"ratified_by_role": "manager"},
    )
    assert first_res.status_code == 200

    second_res = client.post(
        f"/governance/emergency-changes/{ec_id}/ratify",
        json={"ratified_by_role": "manager"},
    )

    assert second_res.status_code == 409
    body = second_res.json()
    assert body["ok"] is False
    assert body["error"]["code"] == "ALREADY_RATIFIED"
