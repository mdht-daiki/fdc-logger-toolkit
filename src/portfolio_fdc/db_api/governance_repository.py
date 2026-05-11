"""Governance テーブル群の repository 層。"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass


class GovernanceNotFoundError(LookupError):
    """要求されたレコードが存在しない場合に送出する。"""


# ---------------------------------------------------------------------------
# GovernanceChangeRequests
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ChangeRequestRow:
    """GovernanceChangeRequests の 1 行を表す DTO。"""

    id: int
    chart_id: int
    status: str
    proposed_by: str
    proposed_at: str
    change_payload: str
    expected_version: int
    idempotency_key: str


class GovernanceChangeRequestRepository:
    """GovernanceChangeRequests テーブルへの CRUD 操作を提供する。"""

    def create(
        self,
        con: sqlite3.Connection,
        chart_id: int,
        proposed_by: str,
        proposed_at: str,
        change_payload: str,
        expected_version: int,
        idempotency_key: str,
    ) -> int:
        """申請を INSERT し、生成された id を返す。"""
        cur = con.execute(
            """
            INSERT INTO GovernanceChangeRequests
                (chart_id, proposed_by, proposed_at, change_payload,
                 expected_version, idempotency_key)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (chart_id, proposed_by, proposed_at, change_payload, expected_version, idempotency_key),
        )
        return cur.lastrowid  # type: ignore[return-value]

    def find_by_id(self, con: sqlite3.Connection, record_id: int) -> ChangeRequestRow:
        """指定 id の申請を返す。存在しない場合は GovernanceNotFoundError を送出する。"""
        row = con.execute(
            """
            SELECT id, chart_id, status, proposed_by, proposed_at,
                   change_payload, expected_version, idempotency_key
            FROM GovernanceChangeRequests
            WHERE id = ?
            """,
            (record_id,),
        ).fetchone()
        if row is None:
            raise GovernanceNotFoundError(f"GovernanceChangeRequests id={record_id} not found")
        return ChangeRequestRow(*row)

    def update_status(
        self,
        con: sqlite3.Connection,
        record_id: int,
        new_status: str,
    ) -> None:
        """status 列のみを更新する。対象行が存在しない場合は GovernanceNotFoundError を送出する。"""
        cur = con.execute(
            "UPDATE GovernanceChangeRequests SET status = ? WHERE id = ?",
            (new_status, record_id),
        )
        if cur.rowcount == 0:
            raise GovernanceNotFoundError(f"GovernanceChangeRequests id={record_id} not found")

    def list(
        self,
        con: sqlite3.Connection,
        *,
        status: str | None = None,
        chart_id: int | None = None,
        from_ts: str | None = None,
        to_ts: str | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> list[ChangeRequestRow]:
        """条件に一致する申請一覧を返す。0 件時は空リストを返す。"""
        sql = """
            SELECT id, chart_id, status, proposed_by, proposed_at,
                   change_payload, expected_version, idempotency_key
            FROM GovernanceChangeRequests
        """
        where_clauses: list[str] = []
        params: list[object] = []

        if status is not None:
            where_clauses.append("status = ?")
            params.append(status)
        if chart_id is not None:
            where_clauses.append("chart_id = ?")
            params.append(chart_id)
        if from_ts is not None:
            where_clauses.append("datetime(proposed_at) >= datetime(?)")
            params.append(from_ts)
        if to_ts is not None:
            where_clauses.append("datetime(proposed_at) <= datetime(?)")
            params.append(to_ts)

        if where_clauses:
            sql += " WHERE " + " AND ".join(where_clauses)

        sql += " ORDER BY datetime(proposed_at) DESC, id DESC LIMIT ? OFFSET ?"
        params.extend((limit, offset))

        rows = con.execute(sql, params).fetchall()
        return [ChangeRequestRow(*row) for row in rows]


# ---------------------------------------------------------------------------
# GovernanceApprovals
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ApprovalRow:
    """GovernanceApprovals の 1 行を表す DTO。"""

    id: int
    request_id: int
    approved_by: str
    approved_by_role: str
    approved_at: str
    comment: str | None


class GovernanceApprovalsRepository:
    """GovernanceApprovals テーブルへの操作を提供する。"""

    def create(
        self,
        con: sqlite3.Connection,
        request_id: int,
        approved_by: str,
        approved_by_role: str,
        approved_at: str,
        comment: str | None = None,
    ) -> int:
        """承認レコードを INSERT し、生成された id を返す。"""
        cur = con.execute(
            """
            INSERT INTO GovernanceApprovals
                (request_id, approved_by, approved_by_role, approved_at, comment)
            VALUES (?, ?, ?, ?, ?)
            """,
            (request_id, approved_by, approved_by_role, approved_at, comment),
        )
        return cur.lastrowid  # type: ignore[return-value]

    def find_by_request_id(self, con: sqlite3.Connection, request_id: int) -> ApprovalRow:
        """request_id で承認レコードを返す。存在しない場合は GovernanceNotFoundError を送出する。"""
        row = con.execute(
            """
            SELECT id, request_id, approved_by, approved_by_role, approved_at, comment
            FROM GovernanceApprovals
            WHERE request_id = ?
            """,
            (request_id,),
        ).fetchone()
        if row is None:
            raise GovernanceNotFoundError(f"GovernanceApprovals request_id={request_id} not found")
        return ApprovalRow(*row)


# ---------------------------------------------------------------------------
# GovernanceApplyResults
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ApplyResultRow:
    """GovernanceApplyResults の 1 行を表す DTO。"""

    id: int
    request_id: int
    applied_at: str
    success: int
    resulting_version: int | None
    error_code: str | None
    error_message: str | None


class GovernanceApplyResultsRepository:
    """GovernanceApplyResults テーブルへの操作を提供する。"""

    def create(
        self,
        con: sqlite3.Connection,
        request_id: int,
        applied_at: str,
        success: int,
        resulting_version: int | None = None,
        error_code: str | None = None,
        error_message: str | None = None,
    ) -> int:
        """適用結果を INSERT し、生成された id を返す。"""
        cur = con.execute(
            """
            INSERT INTO GovernanceApplyResults
                (request_id, applied_at, success, resulting_version,
                 error_code, error_message)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (request_id, applied_at, success, resulting_version, error_code, error_message),
        )
        return cur.lastrowid  # type: ignore[return-value]

    def find_by_request_id(self, con: sqlite3.Connection, request_id: int) -> ApplyResultRow:
        """request_id で適用結果を返す。存在しない場合は GovernanceNotFoundError を送出する。"""
        row = con.execute(
            """
            SELECT id, request_id, applied_at, success, resulting_version,
                   error_code, error_message
            FROM GovernanceApplyResults
            WHERE request_id = ?
            """,
            (request_id,),
        ).fetchone()
        if row is None:
            raise GovernanceNotFoundError(
                f"GovernanceApplyResults request_id={request_id} not found"
            )
        return ApplyResultRow(*row)


# ---------------------------------------------------------------------------
# GovernanceEmergencyChanges
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class EmergencyChangeRow:
    """GovernanceEmergencyChanges の 1 行を表す DTO。"""

    id: int
    chart_id: int
    changed_by: str
    changed_by_role: str
    changed_at: str
    reason: str
    before_json: str
    after_json: str
    resulting_version: int
    related_issue_or_pr: str | None


class GovernanceEmergencyChangesRepository:
    """GovernanceEmergencyChanges テーブルへの操作を提供する。"""

    def create(
        self,
        con: sqlite3.Connection,
        chart_id: int,
        changed_by: str,
        changed_by_role: str,
        changed_at: str,
        reason: str | None,
        before_json: str,
        after_json: str,
        resulting_version: int,
        related_issue_or_pr: str | None = None,
    ) -> int:
        """緊急変更レコードを INSERT し、生成された id を返す。"""
        stored_reason = reason if reason is not None else ""
        cur = con.execute(
            """
            INSERT INTO GovernanceEmergencyChanges
                (chart_id, changed_by, changed_by_role, changed_at, reason,
                 before_json, after_json, resulting_version, related_issue_or_pr)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                chart_id,
                changed_by,
                changed_by_role,
                changed_at,
                stored_reason,
                before_json,
                after_json,
                resulting_version,
                related_issue_or_pr,
            ),
        )
        return cur.lastrowid  # type: ignore[return-value]

    def find_by_id(self, con: sqlite3.Connection, record_id: int) -> EmergencyChangeRow:
        """指定 id の緊急変更レコードを返す。

        存在しない場合は GovernanceNotFoundError を送出する。
        """
        row = con.execute(
            """
            SELECT id, chart_id, changed_by, changed_by_role, changed_at, reason,
                   before_json, after_json, resulting_version, related_issue_or_pr
            FROM GovernanceEmergencyChanges
            WHERE id = ?
            """,
            (record_id,),
        ).fetchone()
        if row is None:
            raise GovernanceNotFoundError(f"GovernanceEmergencyChanges id={record_id} not found")
        return EmergencyChangeRow(*row)


# ---------------------------------------------------------------------------
# GovernanceRatifications
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RatificationRow:
    """GovernanceRatifications の 1 行を表す DTO。"""

    id: int
    ec_id: int
    ratified_by_role: str
    ratified_at: str
    ratification_comment: str | None
    related_pr: str | None


class GovernanceRatificationsRepository:
    """GovernanceRatifications テーブルへの操作を提供する。"""

    def create(
        self,
        con: sqlite3.Connection,
        ec_id: int,
        ratified_by_role: str,
        ratified_at: str,
        ratification_comment: str | None = None,
        related_pr: str | None = None,
    ) -> int:
        """追認レコードを INSERT し、生成された id を返す。"""
        cur = con.execute(
            """
            INSERT INTO GovernanceRatifications
                (ec_id, ratified_by_role, ratified_at, ratification_comment, related_pr)
            VALUES (?, ?, ?, ?, ?)
            """,
            (ec_id, ratified_by_role, ratified_at, ratification_comment, related_pr),
        )
        return cur.lastrowid  # type: ignore[return-value]

    def find_by_ec_id(self, con: sqlite3.Connection, ec_id: int) -> RatificationRow:
        """ec_id で追認レコードを返す。存在しない場合は GovernanceNotFoundError を送出する。"""
        row = con.execute(
            """
            SELECT id, ec_id, ratified_by_role, ratified_at,
                   ratification_comment, related_pr
            FROM GovernanceRatifications
            WHERE ec_id = ?
            """,
            (ec_id,),
        ).fetchone()
        if row is None:
            raise GovernanceNotFoundError(f"GovernanceRatifications ec_id={ec_id} not found")
        return RatificationRow(*row)
