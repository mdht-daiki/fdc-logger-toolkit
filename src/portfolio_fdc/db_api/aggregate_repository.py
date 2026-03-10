"""aggregate 用の永続化処理をまとめたリポジトリ層。"""

from __future__ import annotations

from .db import MAIN_DB, _connect
from .schemas import ParameterIn, ProcessInfoIn, StepWindowIn


def write_process(p: ProcessInfoIn) -> None:
    """`ProcessInfo` を upsert で 1 件保存する。"""
    con = _connect(MAIN_DB)
    try:
        con.execute(
            """
            INSERT INTO ProcessInfo
            (process_id, tool_id, chamber_id, recipe_id, start_ts, end_ts, raw_csv_path)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(process_id) DO UPDATE SET
                tool_id=excluded.tool_id,
                chamber_id=excluded.chamber_id,
                recipe_id=excluded.recipe_id,
                start_ts=excluded.start_ts,
                end_ts=excluded.end_ts,
                raw_csv_path=excluded.raw_csv_path;
            """,
            (
                p.process_id,
                p.tool_id,
                p.chamber_id,
                p.recipe_id,
                p.start_ts,
                p.end_ts,
                p.raw_csv_path,
            ),
        )
        con.commit()
    finally:
        con.close()


def write_step_windows_bulk(items: list[StepWindowIn]) -> int:
    """`StepWindows` を一括挿入し、投入件数を返す。"""
    if not items:
        return 0
    con = _connect(MAIN_DB)
    try:
        con.executemany(
            """
            INSERT INTO StepWindows
            (process_id, step_no, start_ts, end_ts, source_channel)
            VALUES (?, ?, ?, ?, ?);
            """,
            [(w.process_id, w.step_no, w.start_ts, w.end_ts, w.source_channel) for w in items],
        )
        con.commit()
        return len(items)
    finally:
        con.close()


def write_parameters_bulk(params: list[ParameterIn]) -> int:
    """`Parameters` を一括挿入し、投入件数を返す。"""
    if not params:
        return 0
    con = _connect(MAIN_DB)
    try:
        con.executemany(
            """
            INSERT INTO Parameters
            (process_id, parameter, step_no, feature_type, feature_value)
            VALUES (?, ?, ?, ?, ?);
            """,
            [
                (p.process_id, p.parameter, p.step_no, p.feature_type, p.feature_value)
                for p in params
            ],
        )
        con.commit()
        return len(params)
    finally:
        con.close()


def delete_process(process_id: str) -> int:
    """関連テーブルを含めて `process_id` を削除し、削除件数を返す。"""
    con = _connect(MAIN_DB)
    try:
        con.execute("BEGIN")
        con.execute("DELETE FROM StepWindows WHERE process_id = ?", (process_id,))
        con.execute("DELETE FROM Parameters WHERE process_id = ?", (process_id,))
        deleted = con.execute(
            "DELETE FROM ProcessInfo WHERE process_id = ?", (process_id,)
        ).rowcount
        con.commit()
        return int(deleted)
    except Exception:
        con.rollback()
        raise
    finally:
        con.close()
