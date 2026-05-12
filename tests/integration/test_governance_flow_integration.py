"""
Governance E2E フロー統合テスト

Child 6 (#145) の統合テスト。
通常フロー（create -> approve -> apply）と
緊急フロー（emergency -> ratify）の E2E 検証、および基本的な非回帰確認。
"""

from __future__ import annotations

from datetime import UTC, datetime
from uuid import uuid4

import pytest
from fastapi.testclient import TestClient

from portfolio_fdc.db_api.db import MAIN_DB, _connect

pytestmark = pytest.mark.integration


def _insert_chart_set(suffix: str = "test") -> int:
    """ChartSet レコードを挿入し chart_set_id を返す。"""
    con = _connect(MAIN_DB)
    try:
        now = datetime.now(UTC).isoformat()
        unique_suffix = uuid4().hex[:8]
        set_name = f"test_set_{suffix}_{unique_suffix}"
        con.execute(
            "INSERT INTO ChartSet(name, note, created_at, created_by) VALUES (?, ?, ?, ?)",
            (set_name, "integration test", now, "test_user"),
        )
        chart_set_id = int(con.execute("SELECT last_insert_rowid()").fetchone()[0])
        con.commit()
        return chart_set_id
    finally:
        con.close()


def _insert_chart(
    chart_set_id: int,
    tool_id: str = "TOOL_A",
    chamber_id: str = "CH1",
    parameter: str = "temp",
    value: float = 25.0,
) -> int:
    """単純な ChartV2 レコードを挿入し chart_id を返す。"""
    con = _connect(MAIN_DB)
    try:
        now = datetime.now(UTC).isoformat()
        con.execute(
            """
            INSERT INTO ChartsV2(
                chart_set_id, tool_id, chamber_id, recipe_id, parameter,
                step_no, feature_type, warn_low, warn_high, crit_low, crit_high,
                updated_at, updated_by, update_reason, update_source
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                chart_set_id,
                tool_id,
                chamber_id,
                "RECIPE_A",
                parameter,
                1,
                "rolling_mean",
                value - 5,
                value + 5,
                value - 10,
                value + 10,
                now,
                "test_user",
                "governance_test",
                "test",
            ),
        )
        chart_id = int(con.execute("SELECT last_insert_rowid()").fetchone()[0])
        con.commit()
        return chart_id
    finally:
        con.close()


def test_governance_normal_flow_e2e(client: TestClient) -> None:
    """
    通常フロー E2E: create -> approve -> apply

    change-request を作成し、承認・適用フローが正常に機能することを確認。
    """
    # Setup
    chart_set_id = _insert_chart_set("normal_flow")
    chart_id = _insert_chart(chart_set_id)

    # 1. change-request 作成
    idempotency_key = f"test_normal_{uuid4().hex[:12]}"
    payload = {
        "chart_id": chart_id,
        "proposed_by": "engineer_01",
        "change_payload": '{"warn_low": 20.0, "warn_high": 30.0}',
        "expected_version": 1,
        "idempotency_key": idempotency_key,
    }
    response = client.post("/governance/change-requests", json=payload)
    assert response.status_code == 200
    request_id = response.json()["data"]["request_id"]

    # 2. change-request 承認
    approve_payload = {
        "approved_by": "manager_01",
        "approved_by_role": "manager",
        "comment": "OK",
    }
    response = client.post(
        f"/governance/change-requests/{request_id}/approve",
        json=approve_payload,
    )
    assert response.status_code == 200

    # 3. change-request 適用
    apply_payload = {"applied_by": "ops_01", "applied_by_role": "operator"}
    response = client.post(
        f"/governance/change-requests/{request_id}/apply",
        json=apply_payload,
    )
    assert response.status_code == 200
    assert response.json()["data"]["status"] == "applied"

    # 4. ChartV2 への反映確認
    con = _connect(MAIN_DB)
    try:
        row = con.execute(
            "SELECT warn_low, warn_high, version FROM ChartsV2 WHERE id = ?",
            (chart_id,),
        ).fetchone()
        assert row is not None
        assert abs(row[0] - 20.0) < 0.01
        assert abs(row[1] - 30.0) < 0.01
        assert row[2] >= 1
    finally:
        con.close()


def test_governance_emergency_flow_e2e(client: TestClient) -> None:
    """
    緊急フロー E2E: emergency-change -> ratify

    緊急変更を即座に適用し、事後追認フローが正常に機能することを確認。
    """
    # Setup
    chart_set_id = _insert_chart_set("emergency_flow")
    chart_id = _insert_chart(chart_set_id)

    # 1. 緊急変更を即時適用
    payload = {
        "chart_id": chart_id,
        "changed_by": "duty_engineer",
        "changed_by_role": "engineer",
        "reason": "Critical sensor anomaly",
        "change_payload": '{"warn_low": 15.0, "crit_low": 10.0}',
    }
    response = client.post("/governance/emergency-changes", json=payload)
    assert response.status_code == 200
    emergency_id = response.json()["data"]["request_id"]

    # 2. ChartV2 への反映が即座に行われたことを確認
    con = _connect(MAIN_DB)
    try:
        row = con.execute(
            "SELECT warn_low, crit_low FROM ChartsV2 WHERE id = ?",
            (chart_id,),
        ).fetchone()
        assert row is not None
        assert abs(row[0] - 15.0) < 0.01
        assert abs(row[1] - 10.0) < 0.01
    finally:
        con.close()

    # 3. 事後追認（ratify）
    ratify_payload = {
        "ratified_by": "manager_on_call",
        "ratified_by_role": "manager",
        "related_pr": "https://github.com/mdht-daiki/fdc-logger-toolkit/pull/999",
    }
    response = client.post(
        f"/governance/emergency-changes/{emergency_id}/ratify",
        json=ratify_payload,
    )
    assert response.status_code == 200


def test_governance_duplicate_ratify_blocked(client: TestClient) -> None:
    """
    緊急フローで重複追認が拒否されることを確認。
    """
    # Setup
    chart_set_id = _insert_chart_set("dup_ratify")
    chart_id = _insert_chart(chart_set_id)

    # 1. 緊急変更を適用
    payload = {
        "chart_id": chart_id,
        "changed_by": "duty_engineer",
        "changed_by_role": "engineer",
        "reason": "Emergency adjustment",
        "change_payload": '{"warn_low": 18.0}',
    }
    response = client.post("/governance/emergency-changes", json=payload)
    assert response.status_code == 200
    emergency_id = response.json()["data"]["request_id"]

    # 2. 初回追認
    ratify_payload = {
        "ratified_by": "manager_01",
        "ratified_by_role": "manager",
        "related_pr": "https://github.com/mdht-daiki/fdc-logger-toolkit/pull/1000",
    }
    response = client.post(
        f"/governance/emergency-changes/{emergency_id}/ratify",
        json=ratify_payload,
    )
    assert response.status_code == 200

    # 3. 重複追認を試みる -> 409
    response = client.post(
        f"/governance/emergency-changes/{emergency_id}/ratify",
        json=ratify_payload,
    )
    assert response.status_code == 409


def test_governance_read_endpoints_non_regression(client: TestClient) -> None:
    """
    既存 read endpoint の非回帰確認。
    write 操作後に read endpoint が正常に動作することを確認。
    """
    # Setup
    chart_set_id = _insert_chart_set("read_regression")
    chart_id = _insert_chart(chart_set_id)

    # 1. change-request 作成・適用
    idempotency_key = f"test_read_{uuid4().hex[:12]}"
    payload = {
        "chart_id": chart_id,
        "proposed_by": "engineer_01",
        "change_payload": '{"warn_low": 22.0}',
        "expected_version": 1,
        "idempotency_key": idempotency_key,
    }
    response = client.post("/governance/change-requests", json=payload)
    assert response.status_code == 200
    request_id = response.json()["data"]["request_id"]

    approve_payload = {"approved_by": "manager_01", "approved_by_role": "manager", "comment": "OK"}
    response = client.post(
        f"/governance/change-requests/{request_id}/approve",
        json=approve_payload,
    )
    assert response.status_code == 200
    assert response.json()["ok"] is True
    assert response.json()["data"]["request_id"] == request_id

    apply_payload = {"applied_by": "ops_01", "applied_by_role": "operator"}
    response = client.post(
        f"/governance/change-requests/{request_id}/apply",
        json=apply_payload,
    )
    assert response.status_code == 200
    assert response.json()["ok"] is True
    assert response.json()["data"]["request_id"] == request_id
    assert response.json()["data"]["status"] == "applied"

    # 2. GET /governance/change-requests が正常に動作
    response = client.get("/governance/change-requests")
    assert response.status_code == 200
    assert response.json()["ok"] is True
    requests_list = response.json().get("data", [])
    assert len(requests_list) >= 1

    # 3. GET /governance/audit-events が正常に動作
    now = datetime.now(UTC)
    from_ts = (
        now.replace(hour=0, minute=0, second=0, microsecond=0).isoformat().replace("+00:00", "Z")
    )
    to_ts = (
        now.replace(hour=23, minute=59, second=59, microsecond=999999)
        .isoformat()
        .replace("+00:00", "Z")
    )
    response = client.get(f"/governance/audit-events?from_ts={from_ts}&to_ts={to_ts}&limit=100")
    assert response.status_code == 200
    assert response.json()["ok"] is True
    audit_list = response.json().get("data", [])
    correlated_events = [
        event
        for event in audit_list
        if event.get("target_type") == "change_request" and event.get("target_id") == request_id
    ]
    assert correlated_events
