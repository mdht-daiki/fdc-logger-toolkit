from __future__ import annotations

import sqlite3
from collections.abc import Callable, Iterator

import pytest
from fastapi.testclient import TestClient

from portfolio_fdc.db_api import app as db_app
from portfolio_fdc.db_api.db import MAIN_DB

_MIN_SQLITE_VERSION = (3, 38, 0)
_current_sqlite_version = tuple(int(x) for x in sqlite3.sqlite_version.split("."))
if _current_sqlite_version < _MIN_SQLITE_VERSION:
    raise RuntimeError(
        "SQLite >= 3.38.0 required for timezone-aware datetime() comparisons, "
        f"got {sqlite3.sqlite_version}"
    )


@pytest.fixture
def db_api_client() -> Iterator[TestClient]:
    """Lifespan を含めて DB API アプリへアクセスする TestClient を提供する。"""
    with TestClient(db_app.app) as test_client:
        yield test_client


@pytest.fixture
def client(db_api_client: TestClient) -> TestClient:
    """既存テストとの互換のため client 名でも同じ TestClient を提供する。"""
    return db_api_client


@pytest.fixture
def count_rows() -> Callable[[str], tuple[int, int, int]]:
    """指定 process_id の ProcessInfo/StepWindows/Parameters 件数を返す関数。"""

    def _count_rows(process_id: str) -> tuple[int, int, int]:
        con = sqlite3.connect(MAIN_DB.as_posix())
        try:
            p = con.execute(
                "SELECT COUNT(*) FROM processInfo WHERE process_id = ?",
                (process_id,),
            ).fetchone()[0]
            s = con.execute(
                "SELECT COUNT(*) FROM StepWindows WHERE process_id = ?",
                (process_id,),
            ).fetchone()[0]
            f = con.execute(
                "SELECT COUNT(*) FROM Parameters WHERE process_id = ?",
                (process_id,),
            ).fetchone()[0]
            return int(p), int(s), int(f)
        finally:
            con.close()

    return _count_rows
