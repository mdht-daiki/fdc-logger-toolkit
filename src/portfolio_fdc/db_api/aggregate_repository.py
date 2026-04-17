"""aggregate 用の永続化処理をまとめたリポジトリ層。"""

from __future__ import annotations

import sqlite3

from .db import MAIN_DB, _connect
from .schemas import AggregateWriteIn, ParameterIn, ProcessInfoIn, StepWindowIn

_UPSERT_PROCESS_SQL = """
    INSERT INTO ProcessInfo
    (
        process_id,
        tool_id,
        chamber_id,
        recipe_id,
        start_ts,
        end_ts,
        raw_csv_path,
        lot_id,
        wafer_id
    )
    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
    ON CONFLICT(process_id) DO UPDATE SET
        tool_id=excluded.tool_id,
        chamber_id=excluded.chamber_id,
        recipe_id=excluded.recipe_id,
        start_ts=excluded.start_ts,
        end_ts=excluded.end_ts,
        raw_csv_path=excluded.raw_csv_path,
        lot_id=COALESCE(excluded.lot_id, ProcessInfo.lot_id),
        wafer_id=COALESCE(excluded.wafer_id, ProcessInfo.wafer_id);
"""

_INSERT_STEP_WINDOWS_SQL = """
    INSERT INTO StepWindows
    (process_id, step_no, start_ts, end_ts, source_channel)
    VALUES (?, ?, ?, ?, ?);
"""

_INSERT_PARAMETERS_SQL = """
    INSERT INTO Parameters
    (process_id, parameter, step_no, feature_type, feature_value)
    VALUES (?, ?, ?, ?, ?);
"""


def _process_row(
    p: ProcessInfoIn,
) -> tuple[str, str, str, str, str, str, str, str | None, str | None]:
    """`ProcessInfoIn` を ProcessInfo upsert 用のタプル順へ変換する。"""
    return (
        p.process_id,
        p.tool_id,
        p.chamber_id,
        p.recipe_id,
        p.start_ts.isoformat(),
        p.end_ts.isoformat(),
        p.raw_csv_path,
        p.lot_id,
        p.wafer_id,
    )


def _step_window_rows(items: list[StepWindowIn]) -> list[tuple[str, int, str, str, str]]:
    """`StepWindowIn` 一覧を StepWindows insert 用のタプル一覧へ変換する。"""
    return [
        (w.process_id, w.step_no, w.start_ts.isoformat(), w.end_ts.isoformat(), w.source_channel)
        for w in items
    ]


def _parameter_rows(params: list[ParameterIn]) -> list[tuple[str, str, int, str, float]]:
    """`ParameterIn` 一覧を Parameters insert 用のタプル一覧へ変換する。"""
    return [(p.process_id, p.parameter, p.step_no, p.feature_type, p.feature_value) for p in params]


def _write_process_with_conn(con: sqlite3.Connection, p: ProcessInfoIn) -> None:
    """既存コネクションで ProcessInfo を upsert する。"""
    con.execute(_UPSERT_PROCESS_SQL, _process_row(p))


def _insert_step_windows_with_conn(con: sqlite3.Connection, items: list[StepWindowIn]) -> None:
    """既存コネクションで StepWindows を一括挿入する。"""
    if not items:
        return
    con.executemany(_INSERT_STEP_WINDOWS_SQL, _step_window_rows(items))


def _insert_parameters_with_conn(con: sqlite3.Connection, params: list[ParameterIn]) -> None:
    """既存コネクションで Parameters を一括挿入する。"""
    if not params:
        return
    con.executemany(_INSERT_PARAMETERS_SQL, _parameter_rows(params))


def _purge_related_tables_for_process(con: sqlite3.Connection, process_id: str) -> None:
    """既存コネクションで process_id 配下の関連テーブルを削除する。"""
    con.execute("DELETE FROM StepWindows WHERE process_id = ?", (process_id,))
    con.execute("DELETE FROM Parameters WHERE process_id = ?", (process_id,))


def write_process(p: ProcessInfoIn) -> None:
    """`ProcessInfo` を upsert で 1 件保存する。"""
    con = _connect(MAIN_DB)
    try:
        _write_process_with_conn(con, p)
        con.commit()
    finally:
        con.close()


def write_step_windows_bulk(items: list[StepWindowIn]) -> int:
    """`StepWindows` を一括挿入し、投入件数を返す。"""
    if not items:
        return 0
    con = _connect(MAIN_DB)
    try:
        _insert_step_windows_with_conn(con, items)
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
        _insert_parameters_with_conn(con, params)
        con.commit()
        return len(params)
    finally:
        con.close()


def delete_process(process_id: str) -> int:
    """関連テーブルを含めて `process_id` を削除し、削除件数を返す。"""
    con = _connect(MAIN_DB)
    try:
        con.execute("BEGIN")
        _purge_related_tables_for_process(con, process_id)
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


def write_aggregate_atomic(payload: AggregateWriteIn) -> dict[str, int | bool]:
    """ProcessInfo/StepWindows/Parameters を 1 トランザクションで保存する。"""
    con = _connect(MAIN_DB)
    try:
        process_id = payload.process.process_id
        con.execute("BEGIN")
        _write_process_with_conn(con, payload.process)

        _purge_related_tables_for_process(con, process_id)

        _insert_step_windows_with_conn(con, payload.step_windows)
        _insert_parameters_with_conn(con, payload.parameters)
        con.commit()
        return {
            "ok": True,
            "step_windows": len(payload.step_windows),
            "parameters": len(payload.parameters),
        }
    except Exception:
        con.rollback()
        raise
    finally:
        con.close()
