from __future__ import annotations

import sqlite3
from datetime import datetime
from pathlib import Path
from urllib.parse import urlsplit

import pytest
from fastapi.testclient import TestClient

from portfolio_fdc.dashboard.app import app as dashboard_app
from portfolio_fdc.db_api.db import MAIN_DB
from portfolio_fdc.main import aggregate, run_once, scrape
from portfolio_fdc.tools import generate_logger_csv

pytestmark = pytest.mark.integration


class _BridgeResponse:
    def __init__(self, status_code: int, payload: dict):
        self.status_code = status_code
        self._payload = payload

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP error status={self.status_code}")

    def json(self) -> dict:
        return self._payload


def test_readme_demo_flow_smoke(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    db_api_client: TestClient,
) -> None:
    raw_csv = tmp_path / "raw" / "logger_raw_demo.csv"
    scrape_out = tmp_path / "scrape" / "scrape_TOOL_A.csv"
    detail_dir = tmp_path / "detail"
    monkeypatch.setattr(scrape, "STATE_DIR", tmp_path / "state")

    generate_logger_csv.write_logger_csv(
        path=raw_csv,
        start_ts=datetime.fromisoformat("2026-02-19T00:00:00"),
        seconds=180,
        scenario="mix",
        seed=41,
        append=False,
        tool_id="TOOL_A",
    )

    def fake_post(url: str, json, timeout: int):
        path = urlsplit(url).path
        response = db_api_client.post(path, json=json)
        return _BridgeResponse(status_code=response.status_code, payload=response.json())

    def fake_delete(url: str, json=None, timeout: int = 30):
        path = urlsplit(url).path
        if json is None:
            response = db_api_client.request("DELETE", path)
        else:
            response = db_api_client.request("DELETE", path, json=json)
        return _BridgeResponse(status_code=response.status_code, payload=response.json())

    monkeypatch.setattr(aggregate.requests, "post", fake_post)
    monkeypatch.setattr(aggregate.requests, "delete", fake_delete)

    summary = run_once.run_once(
        tool_id="TOOL_A",
        raw_csv_path=raw_csv,
        db_api="http://testserver",
        now=datetime.fromisoformat("2026-02-19T00:02:59"),
        config_path=Path("src/portfolio_fdc/configs/aggregate_tools.yaml"),
        scrape_output_csv=scrape_out,
        detail_out_dir=detail_dir,
    )

    assert summary.scraped_rows > 0
    assert scrape_out.exists()

    con = sqlite3.connect(MAIN_DB.as_posix())
    try:
        rows = con.execute(
            """
            SELECT process_id
            FROM processInfo
            WHERE raw_csv_path LIKE ?
            """,
            (f"{detail_dir.as_posix()}%",),
        ).fetchall()
    finally:
        con.close()

    process_ids = [str(row[0]) for row in rows]
    assert process_ids

    for process_id in process_ids:
        aggregate.delete_process("http://testserver", process_id)

    dashboard_client = dashboard_app.server.test_client()
    response = dashboard_client.get("/_dash-layout")
    assert response.status_code == 200
