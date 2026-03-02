from __future__ import annotations

import sqlite3
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from urllib.parse import urlsplit
from uuid import uuid4

import pandas as pd
import pytest
import yaml
from fastapi.testclient import TestClient

from portfolio_fdc.db_api import app as db_app
from portfolio_fdc.db_api.db import MAIN_DB
from portfolio_fdc.main import aggregate

pytestmark = pytest.mark.integration


@dataclass
class _RequestsBridgeResponse:
    status_code: int
    _payload: dict

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP error status={self.status_code}")

    def json(self) -> dict:
        return self._payload


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


def test_aggregate_http_flow_to_db_api(monkeypatch) -> None:
    client = TestClient(db_app.app)

    def fake_post(url: str, json, timeout: int):
        path = urlsplit(url).path
        response = client.post(path, json=json)
        return _RequestsBridgeResponse(status_code=response.status_code, _payload=response.json())

    def fake_delete(url: str, json, timeout: int):
        path = urlsplit(url).path
        response = client.request("DELETE", path, json=json)
        return _RequestsBridgeResponse(status_code=response.status_code, _payload=response.json())

    monkeypatch.setattr(aggregate.requests, "post", fake_post)
    monkeypatch.setattr(aggregate.requests, "delete", fake_delete)

    process_id = f"agg_{uuid4().hex}"
    start_ts = pd.Timestamp(datetime.now().isoformat())
    end_ts = start_ts + pd.Timedelta(seconds=5)

    features = [
        {
            "parameter": "dc_bias",
            "step_no": 1,
            "feature_type": "mean",
            "feature_value": 1.0,
        },
        {
            "parameter": "cl2_flow",
            "step_no": 1,
            "feature_type": "max",
            "feature_value": 12.0,
        },
    ]
    step_windows = [(1, start_ts, end_ts)]

    try:
        aggregate.post_one_process(
            db_api="http://testserver",
            tool_id="TOOL_A",
            chamber_id="CH1",
            recipe_id="UNKNOWN",
            process_id=process_id,
            start_ts=start_ts,
            end_ts=end_ts,
            raw_csv_path=f"data/detail/detail_TOOL_A_CH1_{process_id}.csv",
        )
        aggregate.post_step_windows(
            db_api="http://testserver",
            process_id=process_id,
            step_windows=step_windows,
            source_channel="dc_bias",
        )
        aggregate.post_features(
            db_api="http://testserver",
            process_id=process_id,
            feats=features,
        )

        process_count, step_count, feature_count = _count_rows(process_id)
        assert process_count == 1
        assert step_count == 1
        assert feature_count == 2

    finally:
        aggregate.delete_process("http://testserver", process_id)

    process_count, step_count, feature_count = _count_rows(process_id)
    assert process_count == 0
    assert step_count == 0
    assert feature_count == 0


def test_aggregate_main_non_dry_run_posts_to_db_api(tmp_path: Path, monkeypatch) -> None:
    client = TestClient(db_app.app)

    def fake_post(url: str, json, timeout: int):
        path = urlsplit(url).path
        response = client.post(path, json=json)
        return _RequestsBridgeResponse(status_code=response.status_code, _payload=response.json())

    def fake_delete(url: str, json, timeout: int):
        path = urlsplit(url).path
        response = client.request("DELETE", path, json=json)
        return _RequestsBridgeResponse(status_code=response.status_code, _payload=response.json())

    monkeypatch.setattr(aggregate.requests, "post", fake_post)
    monkeypatch.setattr(aggregate.requests, "delete", fake_delete)

    tool_id = f"TOOL_{uuid4().hex[:8]}"
    chamber_id = f"CH_{uuid4().hex[:6]}"

    input_csv = tmp_path / "scrape_out.csv"
    config_yaml = tmp_path / "aggregate_tools.yaml"
    detail_dir = tmp_path / "detail"

    ts = pd.date_range("2026-02-19T00:00:00", periods=10, freq="s")
    df = pd.DataFrame(
        {
            "timestamp": ts,
            "tool_id": [tool_id] * len(ts),
            "chamber_id": [chamber_id] * len(ts),
            "dc_bias": [0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0],
            "cl2_flow": [0.0, 10.0, 10.5, 11.0, 11.5, 12.0, 12.5, 0.0, 0.0, 0.0],
            "apc_pressure": [100.0, 101.0, 102.0, 103.0, 104.0, 105.0, 106.0, 107.0, 108.0, 109.0],
        }
    )
    df.to_csv(input_csv, index=False)
    config_yaml.write_text(yaml.safe_dump({"tools": {}}), encoding="utf-8")

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "aggregate.py",
            "--input",
            str(input_csv),
            "--config",
            str(config_yaml),
            "--detail-out",
            str(detail_dir),
            "--db-api",
            "http://testserver",
        ],
    )

    aggregate.main()

    con = sqlite3.connect(MAIN_DB.as_posix())
    try:
        row = con.execute(
            """
            SELECT process_id, raw_csv_path
            FROM processInfo
            WHERE tool_id = ? AND chamber_id = ?
            ORDER BY start_ts DESC
            LIMIT 1
            """,
            (tool_id, chamber_id),
        ).fetchone()
        assert row is not None
        process_id = str(row[0])
        raw_csv_path = str(row[1])

        step_count = con.execute(
            "SELECT COUNT(*) FROM StepWindows WHERE process_id = ?",
            (process_id,),
        ).fetchone()[0]
        feature_count = con.execute(
            "SELECT COUNT(*) FROM Parameters WHERE process_id = ?",
            (process_id,),
        ).fetchone()[0]
    finally:
        con.close()

    assert step_count >= 1
    assert feature_count > 0
    assert Path(raw_csv_path).exists()

    aggregate.delete_process("http://testserver", process_id)
    process_count, step_count, feature_count = _count_rows(process_id)
    assert process_count == 0
    assert step_count == 0
    assert feature_count == 0
