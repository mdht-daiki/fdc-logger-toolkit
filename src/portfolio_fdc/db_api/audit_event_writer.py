"""GovernanceAuditEvents への書き込みを担う AuditEventWriter。"""

from __future__ import annotations

import sqlite3


class AuditEventWriter:
    """GovernanceAuditEvents テーブルへの監査イベント書き込みを担う。

    occurred_at の正規化は呼び出し側（service 層）の責務。
    writer 内では正規化を行わない。
    """

    def write(
        self,
        con: sqlite3.Connection,
        event_type: str,
        actor: str,
        actor_role: str,
        target_type: str,
        target_id: int,
        occurred_at: str,
        before_json: str | None = None,
        after_json: str | None = None,
        correlation_id: str | None = None,
    ) -> int:
        """監査イベントを INSERT し、生成された id を返す。"""
        cur = con.execute(
            """
            INSERT INTO GovernanceAuditEvents
                (event_type, actor, actor_role, target_type, target_id,
                 occurred_at, before_json, after_json, correlation_id)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                event_type,
                actor,
                actor_role,
                target_type,
                target_id,
                occurred_at,
                before_json,
                after_json,
                correlation_id,
            ),
        )
        return cur.lastrowid  # type: ignore[return-value]
