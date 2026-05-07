"""governance_repository と AuditEventWriter のユニットテスト。"""

from __future__ import annotations

import sqlite3
from collections.abc import Generator
from pathlib import Path

import pytest

from portfolio_fdc.db_api.audit_event_writer import AuditEventWriter
from portfolio_fdc.db_api.db import _init_schema
from portfolio_fdc.db_api.governance_repository import (
    GovernanceApplyResultsRepository,
    GovernanceApprovalsRepository,
    GovernanceChangeRequestRepository,
    GovernanceEmergencyChangesRepository,
    GovernanceNotFoundError,
    GovernanceRatificationsRepository,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def db_path(tmp_path: Path) -> Path:
    path = tmp_path / "test_governance.db"
    _init_schema(path)
    return path


@pytest.fixture()
def con(db_path: Path) -> Generator[sqlite3.Connection, None, None]:
    c = sqlite3.connect(db_path.as_posix())
    c.execute("PRAGMA foreign_keys=ON;")
    # テスト用の ChartsV2 ダミー行を挿入しておく
    c.execute(
        """
        INSERT INTO ChartsV2
            (chart_set_id, tool_id, chamber_id, recipe_id, parameter,
             step_no, feature_type, updated_at)
        VALUES (1, 'T01', 'C01', 'R01', 'param1', 1, 'mean', '2026-01-01T00:00:00.000Z')
        """
    )
    c.commit()
    yield c
    c.close()


def _chart_id(con: sqlite3.Connection) -> int:
    return con.execute("SELECT id FROM ChartsV2 LIMIT 1").fetchone()[0]


# ---------------------------------------------------------------------------
# schema 冪等性テスト
# ---------------------------------------------------------------------------


def test_init_schema_is_idempotent(tmp_path: Path) -> None:
    """_init_schema を2回呼んでもエラーなし。"""
    path = tmp_path / "idempotent.db"
    _init_schema(path)
    _init_schema(path)  # 2回目もエラーなし


def test_init_schema_does_not_alter_existing_columns(db_path: Path) -> None:
    """governance テーブル追加が ChartsV2 の既存列を壊さないこと。"""
    con = sqlite3.connect(db_path.as_posix())
    cols = {row[1] for row in con.execute("PRAGMA table_info(ChartsV2)").fetchall()}
    con.close()
    # 既存列が揃っていること
    for col in (
        "id",
        "chart_set_id",
        "tool_id",
        "chamber_id",
        "recipe_id",
        "parameter",
        "step_no",
        "feature_type",
        "updated_at",
    ):
        assert col in cols, f"既存列 {col!r} が消えている"
    # 新規追加列が存在すること
    assert "version" in cols
    assert "chart_name" in cols


# ---------------------------------------------------------------------------
# GovernanceChangeRequestRepository
# ---------------------------------------------------------------------------


class TestGovernanceChangeRequestRepository:
    repo = GovernanceChangeRequestRepository()

    def _create_request(
        self,
        con: sqlite3.Connection,
        *,
        chart_id: int,
        proposed_by: str,
        proposed_at: str,
        idempotency_key: str,
    ) -> int:
        rid = self.repo.create(
            con,
            chart_id=chart_id,
            proposed_by=proposed_by,
            proposed_at=proposed_at,
            change_payload="{}",
            expected_version=1,
            idempotency_key=idempotency_key,
        )
        con.commit()
        return rid

    def test_create_and_find_by_id(self, con: sqlite3.Connection) -> None:
        cid = _chart_id(con)
        new_id = self.repo.create(
            con,
            chart_id=cid,
            proposed_by="alice",
            proposed_at="2026-05-01T00:00:00.000Z",
            change_payload='{"ucl": 1.5}',
            expected_version=1,
            idempotency_key="key-001",
        )
        con.commit()

        row = self.repo.find_by_id(con, new_id)
        assert row.id == new_id
        assert row.chart_id == cid
        assert row.status == "pending"
        assert row.proposed_by == "alice"
        assert row.idempotency_key == "key-001"

    def test_update_status(self, con: sqlite3.Connection) -> None:
        cid = _chart_id(con)
        new_id = self.repo.create(
            con,
            chart_id=cid,
            proposed_by="bob",
            proposed_at="2026-05-01T00:00:00.000Z",
            change_payload="{}",
            expected_version=1,
            idempotency_key="key-002",
        )
        con.commit()

        self.repo.update_status(con, new_id, "approved")
        con.commit()

        row = self.repo.find_by_id(con, new_id)
        assert row.status == "approved"

    def test_find_by_id_not_found(self, con: sqlite3.Connection) -> None:
        with pytest.raises(GovernanceNotFoundError):
            self.repo.find_by_id(con, 99999)

    def test_update_status_not_found(self, con: sqlite3.Connection) -> None:
        with pytest.raises(GovernanceNotFoundError):
            self.repo.update_status(con, 99999, "approved")

    def test_duplicate_idempotency_key_raises(self, con: sqlite3.Connection) -> None:
        cid = _chart_id(con)
        self.repo.create(
            con,
            chart_id=cid,
            proposed_by="alice",
            proposed_at="2026-05-01T00:00:00.000Z",
            change_payload="{}",
            expected_version=1,
            idempotency_key="dup-key",
        )
        con.commit()

        with pytest.raises(sqlite3.IntegrityError):
            self.repo.create(
                con,
                chart_id=cid,
                proposed_by="alice",
                proposed_at="2026-05-01T00:01:00.000Z",
                change_payload="{}",
                expected_version=1,
                idempotency_key="dup-key",
            )

    def test_list_returns_empty_when_no_records(self, con: sqlite3.Connection) -> None:
        rows = self.repo.list(con)
        assert rows == []

    def test_list_filters_by_status_and_chart_id(self, con: sqlite3.Connection) -> None:
        chart_1 = _chart_id(con)
        chart_2_row_id = con.execute(
            """
            INSERT INTO ChartsV2
                (chart_set_id, tool_id, chamber_id, recipe_id, parameter,
                 step_no, feature_type, updated_at)
            VALUES (1, 'T01', 'C01', 'R01', 'param2', 2, 'mean', '2026-01-01T00:00:00.000Z')
            """
        ).lastrowid
        if chart_2_row_id is None:
            raise RuntimeError("Failed to insert chart row for test")
        chart_2 = int(chart_2_row_id)
        con.commit()

        req_1 = self._create_request(
            con,
            chart_id=chart_1,
            proposed_by="alice",
            proposed_at="2026-05-01T00:00:00.000Z",
            idempotency_key="list-key-1",
        )
        req_2 = self._create_request(
            con,
            chart_id=chart_1,
            proposed_by="bob",
            proposed_at="2026-05-01T01:00:00.000Z",
            idempotency_key="list-key-2",
        )
        req_3 = self._create_request(
            con,
            chart_id=chart_2,
            proposed_by="carol",
            proposed_at="2026-05-01T02:00:00.000Z",
            idempotency_key="list-key-3",
        )
        self.repo.update_status(con, req_2, "approved")
        con.commit()

        rows = self.repo.list(con, status="pending", chart_id=chart_1)

        assert [r.id for r in rows] == [req_1]
        assert all(r.status == "pending" for r in rows)
        assert all(r.chart_id == chart_1 for r in rows)
        assert req_3 not in [r.id for r in rows]

    def test_list_filters_by_from_to(self, con: sqlite3.Connection) -> None:
        chart_id = _chart_id(con)
        req_1 = self._create_request(
            con,
            chart_id=chart_id,
            proposed_by="alice",
            proposed_at="2026-05-01T00:00:00.000Z",
            idempotency_key="range-key-1",
        )
        req_2 = self._create_request(
            con,
            chart_id=chart_id,
            proposed_by="bob",
            proposed_at="2026-05-02T00:00:00.000Z",
            idempotency_key="range-key-2",
        )
        req_3 = self._create_request(
            con,
            chart_id=chart_id,
            proposed_by="carol",
            proposed_at="2026-05-03T00:00:00.000Z",
            idempotency_key="range-key-3",
        )

        rows = self.repo.list(
            con,
            from_ts="2026-05-02T00:00:00.000Z",
            to_ts="2026-05-03T00:00:00.000Z",
        )

        assert [r.id for r in rows] == [req_3, req_2]
        assert req_1 not in [r.id for r in rows]

    def test_list_applies_limit_and_offset(self, con: sqlite3.Connection) -> None:
        chart_id = _chart_id(con)
        self._create_request(
            con,
            chart_id=chart_id,
            proposed_by="alice",
            proposed_at="2026-05-01T00:00:00.000Z",
            idempotency_key="page-key-1",
        )
        req_2 = self._create_request(
            con,
            chart_id=chart_id,
            proposed_by="bob",
            proposed_at="2026-05-02T00:00:00.000Z",
            idempotency_key="page-key-2",
        )
        self._create_request(
            con,
            chart_id=chart_id,
            proposed_by="carol",
            proposed_at="2026-05-03T00:00:00.000Z",
            idempotency_key="page-key-3",
        )

        rows = self.repo.list(con, limit=1, offset=1)

        assert len(rows) == 1
        assert rows[0].id == req_2


# ---------------------------------------------------------------------------
# GovernanceApprovalsRepository
# ---------------------------------------------------------------------------


class TestGovernanceApprovalsRepository:
    req_repo = GovernanceChangeRequestRepository()
    repo = GovernanceApprovalsRepository()

    def _make_request(self, con: sqlite3.Connection, key: str = "key-a") -> int:
        cid = _chart_id(con)
        rid = self.req_repo.create(
            con,
            chart_id=cid,
            proposed_by="alice",
            proposed_at="2026-05-01T00:00:00.000Z",
            change_payload="{}",
            expected_version=1,
            idempotency_key=key,
        )
        con.commit()
        return rid

    def test_create_and_find_by_request_id(self, con: sqlite3.Connection) -> None:
        rid = self._make_request(con)
        aid = self.repo.create(
            con,
            request_id=rid,
            approved_by="manager",
            approved_by_role="approver",
            approved_at="2026-05-02T00:00:00.000Z",
            comment="LGTM",
        )
        con.commit()

        row = self.repo.find_by_request_id(con, rid)
        assert row.id == aid
        assert row.request_id == rid
        assert row.approved_by == "manager"
        assert row.approved_by_role == "approver"
        assert row.comment == "LGTM"

    def test_find_not_found(self, con: sqlite3.Connection) -> None:
        with pytest.raises(GovernanceNotFoundError):
            self.repo.find_by_request_id(con, 99999)

    def test_duplicate_approval_raises(self, con: sqlite3.Connection) -> None:
        rid = self._make_request(con, key="key-dup-a")
        self.repo.create(
            con,
            request_id=rid,
            approved_by="mgr1",
            approved_by_role="approver",
            approved_at="2026-05-02T00:00:00.000Z",
        )
        con.commit()

        with pytest.raises(sqlite3.IntegrityError):
            self.repo.create(
                con,
                request_id=rid,
                approved_by="mgr2",
                approved_by_role="approver",
                approved_at="2026-05-02T01:00:00.000Z",
            )


# ---------------------------------------------------------------------------
# GovernanceApplyResultsRepository
# ---------------------------------------------------------------------------


class TestGovernanceApplyResultsRepository:
    req_repo = GovernanceChangeRequestRepository()
    repo = GovernanceApplyResultsRepository()

    def _make_request(self, con: sqlite3.Connection, key: str = "key-b") -> int:
        cid = _chart_id(con)
        rid = self.req_repo.create(
            con,
            chart_id=cid,
            proposed_by="alice",
            proposed_at="2026-05-01T00:00:00.000Z",
            change_payload="{}",
            expected_version=1,
            idempotency_key=key,
        )
        con.commit()
        return rid

    def test_create_success_and_find(self, con: sqlite3.Connection) -> None:
        rid = self._make_request(con)
        result_id = self.repo.create(
            con,
            request_id=rid,
            applied_at="2026-05-03T00:00:00.000Z",
            success=1,
            resulting_version=2,
        )
        con.commit()

        row = self.repo.find_by_request_id(con, rid)
        assert row.id == result_id
        assert row.success == 1
        assert row.resulting_version == 2
        assert row.error_code is None

    def test_create_failure_and_find(self, con: sqlite3.Connection) -> None:
        rid = self._make_request(con, key="key-b2")
        self.repo.create(
            con,
            request_id=rid,
            applied_at="2026-05-03T00:00:00.000Z",
            success=0,
            error_code="STALE_VERSION",
            error_message="version mismatch",
        )
        con.commit()

        row = self.repo.find_by_request_id(con, rid)
        assert row.success == 0
        assert row.resulting_version is None
        assert row.error_code == "STALE_VERSION"

    def test_find_not_found(self, con: sqlite3.Connection) -> None:
        with pytest.raises(GovernanceNotFoundError):
            self.repo.find_by_request_id(con, 99999)

    def test_duplicate_result_raises(self, con: sqlite3.Connection) -> None:
        rid = self._make_request(con, key="key-dup-b")
        self.repo.create(
            con,
            request_id=rid,
            applied_at="2026-05-03T00:00:00.000Z",
            success=1,
        )
        con.commit()

        with pytest.raises(sqlite3.IntegrityError):
            self.repo.create(
                con,
                request_id=rid,
                applied_at="2026-05-03T01:00:00.000Z",
                success=0,
            )


# ---------------------------------------------------------------------------
# GovernanceEmergencyChangesRepository
# ---------------------------------------------------------------------------


class TestGovernanceEmergencyChangesRepository:
    repo = GovernanceEmergencyChangesRepository()

    def test_create_and_find_by_id(self, con: sqlite3.Connection) -> None:
        cid = _chart_id(con)
        ec_id = self.repo.create(
            con,
            chart_id=cid,
            changed_by="ops-user",
            changed_by_role="operator",
            changed_at="2026-05-04T00:00:00.000Z",
            reason="production incident",
            before_json='{"ucl": 1.0}',
            after_json='{"ucl": 1.5}',
            resulting_version=2,
        )
        con.commit()

        row = self.repo.find_by_id(con, ec_id)
        assert row.id == ec_id
        assert row.chart_id == cid
        assert row.changed_by == "ops-user"
        assert row.reason == "production incident"
        assert row.related_issue_or_pr is None

    def test_find_not_found(self, con: sqlite3.Connection) -> None:
        with pytest.raises(GovernanceNotFoundError):
            self.repo.find_by_id(con, 99999)


# ---------------------------------------------------------------------------
# GovernanceRatificationsRepository
# ---------------------------------------------------------------------------


class TestGovernanceRatificationsRepository:
    ec_repo = GovernanceEmergencyChangesRepository()
    repo = GovernanceRatificationsRepository()

    def _make_ec(self, con: sqlite3.Connection) -> int:
        cid = _chart_id(con)
        ec_id = self.ec_repo.create(
            con,
            chart_id=cid,
            changed_by="ops",
            changed_by_role="operator",
            changed_at="2026-05-04T00:00:00.000Z",
            reason="incident",
            before_json="{}",
            after_json="{}",
            resulting_version=2,
        )
        con.commit()
        return ec_id

    def test_create_and_find_by_ec_id(self, con: sqlite3.Connection) -> None:
        ec_id = self._make_ec(con)
        rat_id = self.repo.create(
            con,
            ec_id=ec_id,
            ratified_by_role="manager",
            ratified_at="2026-05-05T00:00:00.000Z",
            ratification_comment="approved retrospectively",
        )
        con.commit()

        row = self.repo.find_by_ec_id(con, ec_id)
        assert row.id == rat_id
        assert row.ec_id == ec_id
        assert row.ratified_by_role == "manager"
        assert row.ratification_comment == "approved retrospectively"

    def test_find_not_found(self, con: sqlite3.Connection) -> None:
        with pytest.raises(GovernanceNotFoundError):
            self.repo.find_by_ec_id(con, 99999)

    def test_duplicate_ratification_raises(self, con: sqlite3.Connection) -> None:
        ec_id = self._make_ec(con)
        self.repo.create(
            con,
            ec_id=ec_id,
            ratified_by_role="manager",
            ratified_at="2026-05-05T00:00:00.000Z",
        )
        con.commit()

        with pytest.raises(sqlite3.IntegrityError):
            self.repo.create(
                con,
                ec_id=ec_id,
                ratified_by_role="manager2",
                ratified_at="2026-05-05T01:00:00.000Z",
            )


# ---------------------------------------------------------------------------
# AuditEventWriter
# ---------------------------------------------------------------------------


class TestAuditEventWriter:
    writer = AuditEventWriter()

    def test_write_records_all_required_columns(self, con: sqlite3.Connection) -> None:
        """write() 後、全必須列が記録されること。"""
        event_id = self.writer.write(
            con,
            event_type="change_requested",
            actor="alice",
            actor_role="requester",
            target_type="change_request",
            target_id=1,
            occurred_at="2026-05-01T00:00:00.000Z",
        )
        con.commit()

        row = con.execute(
            """
            SELECT event_type, actor, actor_role, target_type, target_id, occurred_at
            FROM GovernanceAuditEvents WHERE id = ?
            """,
            (event_id,),
        ).fetchone()
        assert row is not None
        assert row[0] == "change_requested"
        assert row[1] == "alice"
        assert row[2] == "requester"
        assert row[3] == "change_request"
        assert row[4] == 1
        assert row[5] == "2026-05-01T00:00:00.000Z"

    def test_write_before_json_none_stores_null(self, con: sqlite3.Connection) -> None:
        """before_json=None で呼び出した場合、DB の before_json は NULL。"""
        event_id = self.writer.write(
            con,
            event_type="change_request_approved",
            actor="mgr",
            actor_role="approver",
            target_type="change_request",
            target_id=1,
            occurred_at="2026-05-02T00:00:00.000Z",
            before_json=None,
        )
        con.commit()

        row = con.execute(
            "SELECT before_json, after_json FROM GovernanceAuditEvents WHERE id = ?",
            (event_id,),
        ).fetchone()
        assert row[0] is None
        assert row[1] is None

    def test_write_with_before_after_json(self, con: sqlite3.Connection) -> None:
        """before_json / after_json が正しく保存されること。"""
        event_id = self.writer.write(
            con,
            event_type="change_request_applied",
            actor="system",
            actor_role="service",
            target_type="chart",
            target_id=1,
            occurred_at="2026-05-03T00:00:00.000Z",
            before_json='{"ucl": 1.0}',
            after_json='{"ucl": 1.5}',
        )
        con.commit()

        row = con.execute(
            "SELECT before_json, after_json FROM GovernanceAuditEvents WHERE id = ?",
            (event_id,),
        ).fetchone()
        assert row[0] == '{"ucl": 1.0}'
        assert row[1] == '{"ucl": 1.5}'

    def test_write_does_not_normalize_occurred_at(self, con: sqlite3.Connection) -> None:
        """writer は occurred_at を変換せず、渡した値をそのまま保存する。"""
        raw_ts = "2026-05-01T12:34:56.789Z"
        event_id = self.writer.write(
            con,
            event_type="emergency_changed",
            actor="ops",
            actor_role="operator",
            target_type="chart",
            target_id=1,
            occurred_at=raw_ts,
        )
        con.commit()

        stored = con.execute(
            "SELECT occurred_at FROM GovernanceAuditEvents WHERE id = ?",
            (event_id,),
        ).fetchone()[0]
        assert stored == raw_ts
