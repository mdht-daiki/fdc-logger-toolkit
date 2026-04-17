"""SQLite 接続設定と初期スキーマ作成を提供するユーティリティ。"""

from __future__ import annotations

import os
import sqlite3
from datetime import datetime
from pathlib import Path

DB_DIR_ENV_VAR = "PORTFOLIO_DB_DIR"
DEFAULT_DB_DIR = Path(__file__).resolve().parents[3] / "data" / "db"


def _resolve_db_dir() -> Path:
    """環境変数があれば優先し、未設定時は既定パスへフォールバックする。"""
    raw = os.getenv(DB_DIR_ENV_VAR)
    if not raw:
        return DEFAULT_DB_DIR
    return Path(raw).expanduser().resolve()


DB_DIR = _resolve_db_dir()
MAIN_DB = DB_DIR / "main.db"
TEMP_DB = DB_DIR / "temp.db"


def _connect(db_path: Path) -> sqlite3.Connection:
    """指定 DB への接続を開き、必要な PRAGMA を適用して返す。"""
    con = sqlite3.connect(db_path.as_posix(), check_same_thread=False)
    con.execute("PRAGMA journal_mode=DELETE;")
    con.execute("PRAGMA synchronous=NORMAL;")
    con.execute("PRAGMA foreign_keys=ON;")
    return con


def _init_schema(db_path: Path) -> None:
    """DB ファイル作成と必須テーブル/インデックスの初期化を行う。"""
    db_path.parent.mkdir(parents=True, exist_ok=True)
    con = _connect(db_path)
    try:
        con.execute(
            """
            CREATE TABLE IF NOT EXISTS ProcessInfo (
                process_id TEXT PRIMARY KEY,
                tool_id TEXT NOT NULL,
                chamber_id TEXT NOT NULL,
                recipe_id TEXT NOT NULL,
                start_ts TEXT NOT NULL,
                end_ts TEXT NOT NULL,
                raw_csv_path TEXT NOT NULL,
                lot_id TEXT,
                wafer_id TEXT
            );
            """
        )
        try:
            con.execute("ALTER TABLE ProcessInfo ADD COLUMN lot_id TEXT")
        except sqlite3.OperationalError as e:
            if "duplicate column name" not in str(e):
                raise
        try:
            con.execute("ALTER TABLE ProcessInfo ADD COLUMN wafer_id TEXT")
        except sqlite3.OperationalError as e:
            if "duplicate column name" not in str(e):
                raise
        con.execute(
            """
            CREATE TABLE IF NOT EXISTS StepWindows (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                process_id TEXT NOT NULL,
                step_no INTEGER NOT NULL,
                start_ts TEXT NOT NULL,
                end_ts TEXT NOT NULL,
                source_channel TEXT NOT NULL,
                FOREIGN KEY (process_id) REFERENCES ProcessInfo (process_id)
            );
            """
        )
        con.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_step_windows_process
            ON StepWindows (process_id);
            """
        )
        con.execute(
            """
            CREATE TABLE IF NOT EXISTS Parameters (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                process_id TEXT NOT NULL,
                parameter TEXT NOT NULL,
                step_no INTEGER NOT NULL,
                feature_type TEXT NOT NULL,
                feature_value REAL NOT NULL,
                FOREIGN KEY (process_id) REFERENCES ProcessInfo (process_id)
            );
            """
        )
        con.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_params_process
            ON Parameters (process_id);
            """
        )
        con.execute(
            """
            CREATE TABLE IF NOT EXISTS Charts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                tool_id TEXT NOT NULL,
                chamber_id TEXT NOT NULL,
                recipe_id TEXT NOT NULL,
                parameter TEXT NOT NULL,
                step_no INTEGER NOT NULL,
                feature_type TEXT NOT NULL,
                warn_low REAL,
                warn_high REAL,
                crit_low REAL,
                crit_high REAL,
                UNIQUE(tool_id, chamber_id, recipe_id, parameter, step_no, feature_type)
            );
            """
        )
        con.execute(
            """
            CREATE TABLE IF NOT EXISTS JudgementResults (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                process_id TEXT NOT NULL,
                tool_id TEXT NOT NULL,
                chamber_id TEXT NOT NULL,
                recipe_id TEXT NOT NULL,
                status TEXT NOT NULL,
                judged_at TEXT NOT NULL,
                message_json TEXT NOT NULL,
                FOREIGN KEY(process_id) REFERENCES ProcessInfo(process_id)
            );
            """
        )
        con.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_judge_process
            ON JudgementResults(process_id);
            """
        )
        con.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_judge_time
            ON JudgementResults(judged_at);
            """
        )
        con.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_judgementresults_recipe_judged_at_id
            ON JudgementResults(recipe_id, judged_at DESC, id DESC);
            """
        )
        con.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_processinfo_lot_process
            ON ProcessInfo(lot_id, process_id);
            """
        )
        con.execute(
            """
            CREATE TABLE IF NOT EXISTS ChartSet (
                chart_set_id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                note TEXT,
                created_at TEXT NOT NULL,
                created_by TEXT
            );
            """
        )
        con.execute(
            """
            CREATE TABLE IF NOT EXISTS ActiveChartSet (
                id INTEGER PRIMARY KEY CHECK (id = 1),
                chart_set_id INTEGER NOT NULL,
                updated_at TEXT NOT NULL,
                updated_by TEXT,
                FOREIGN KEY(chart_set_id) REFERENCES ChartSet(chart_set_id)
            );
            """
        )
        con.execute(
            """
            CREATE TABLE IF NOT EXISTS ChartsV2 (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                chart_set_id INTEGER NOT NULL,
                tool_id TEXT NOT NULL,
                chamber_id TEXT NOT NULL,
                recipe_id TEXT NOT NULL,
                parameter TEXT NOT NULL,
                step_no INTEGER NOT NULL,
                feature_type TEXT NOT NULL,
                warn_low REAL,
                warn_high REAL,
                crit_low REAL,
                crit_high REAL,
                updated_at TEXT NOT NULL,
                updated_by TEXT,
                update_reason TEXT,
                update_source TEXT,
                UNIQUE(
                    chart_set_id,
                    tool_id,
                    chamber_id,
                    recipe_id,
                    parameter,
                    step_no,
                    feature_type
                ),
                FOREIGN KEY(chart_set_id) REFERENCES ChartSet(chart_set_id)
            );
            """
        )
        con.execute(
            """
            CREATE TABLE IF NOT EXISTS ChartsHistory (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                chart_set_id INTEGER NOT NULL,
                tool_id TEXT NOT NULL,
                chamber_id TEXT NOT NULL,
                recipe_id TEXT NOT NULL,
                parameter TEXT NOT NULL,
                step_no INTEGER NOT NULL,
                feature_type TEXT NOT NULL,
                old_warn_low REAL,
                old_warn_high REAL,
                old_crit_low REAL,
                old_crit_high REAL,
                new_warn_low REAL,
                new_warn_high REAL,
                new_crit_low REAL,
                new_crit_high REAL,
                changed_at TEXT NOT NULL,
                changed_by TEXT,
                change_reason TEXT,
                change_source TEXT,
                chart_id INTEGER,
                FOREIGN KEY(chart_set_id) REFERENCES ChartSet(chart_set_id)
            );
            """
        )
        # Migration: add chart_id column for existing databases that predate this column.
        # chart_id stores the ChartsV2 primary key at the time of the change, preserving
        # the stable identifier in audit records even after the chart row is later deleted.
        try:
            con.execute("ALTER TABLE ChartsHistory ADD COLUMN chart_id INTEGER")
            con.execute(
                """
                UPDATE ChartsHistory
                SET chart_id = (
                    SELECT c.id
                    FROM ChartsV2 c
                    WHERE c.chart_set_id = ChartsHistory.chart_set_id
                      AND c.tool_id = ChartsHistory.tool_id
                      AND c.chamber_id = ChartsHistory.chamber_id
                      AND c.recipe_id = ChartsHistory.recipe_id
                      AND c.parameter = ChartsHistory.parameter
                      AND c.step_no = ChartsHistory.step_no
                      AND c.feature_type = ChartsHistory.feature_type
                )
                WHERE chart_id IS NULL
                """
            )
        except sqlite3.OperationalError:
            pass  # column already exists – migration already applied
        con.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_charts_v2_lookup
            ON ChartsV2(chart_set_id, tool_id, chamber_id, recipe_id);
            """
        )
        con.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_charts_hist_time
            ON ChartsHistory(changed_at);
            """
        )
        con.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_charts_hist_key
            ON ChartsHistory(
                chart_set_id,
                tool_id,
                chamber_id,
                recipe_id,
                parameter,
                step_no,
                feature_type
            );
            """
        )

        row = con.execute(
            "SELECT chart_set_id FROM ChartSet ORDER BY chart_set_id ASC LIMIT 1"
        ).fetchone()
        if row is None:
            now = datetime.now().isoformat()
            con.execute(
                "INSERT INTO ChartSet(name, note, created_at, created_by) VALUES (?, ?, ?, ?)",
                ("default", "auto-created", now, "system"),
            )
            cs_id = con.execute(
                "SELECT chart_set_id FROM ChartSet ORDER BY chart_set_id DESC LIMIT 1"
            ).fetchone()[0]
            con.execute(
                (
                    "INSERT OR REPLACE INTO ActiveChartSet("
                    "id, chart_set_id, updated_at, updated_by"
                    ") VALUES (1, ?, ?, ?)"
                ),
                (cs_id, now, "system"),
            )
        else:
            cs_id = row[0]
            now = datetime.now().isoformat()
            con.execute(
                (
                    "INSERT OR IGNORE INTO ActiveChartSet("
                    "id, chart_set_id, updated_at, updated_by"
                    ") VALUES (1, ?, ?, ?)"
                ),
                (cs_id, now, "system"),
            )
        con.commit()
    finally:
        con.close()
