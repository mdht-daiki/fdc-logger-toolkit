from __future__ import annotations

import sqlite3
from datetime import datetime
from uuid import uuid4

import pytest
from fastapi.testclient import TestClient

from portfolio_fdc.db_api import app as db_app
from portfolio_fdc.db_api.db import MAIN_DB

pytestmark = pytest.mark.integration


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


def test_db_api_minimum_flow_for_aggregate_contract() -> None:
    client = TestClient(db_app.app)
    process_id = f"it_{uuid4().hex}"

    process_payload = {
        "process_id": process_id,
        "tool_id": "TOOL_A",
        "chamber_id": "CH1",
        "recipe_id": "UNKNOWN",
        "start_ts": datetime.now().isoformat(),
        "end_ts": datetime.now().isoformat(),
        "raw_csv_path": f"data/detail/detail_TOOL_A_CH1_{process_id}.csv",
    }

    try:
        created = client.post("/processes", json=process_payload)
        assert created.status_code == 200
        assert created.json()["ok"] is True

        steps_payload = [
            {
                "process_id": process_id,
                "step_no": 1,
                "start_ts": process_payload["start_ts"],
                "end_ts": process_payload["end_ts"],
                "source_channel": "dc_bias",
            }
        ]
        step_res = client.post("/step_windows/bulk", json=steps_payload)
        assert step_res.status_code == 200
        assert step_res.json() == {"ok": True, "inserted": 1}

        features_payload = [
            {
                "process_id": process_id,
                "parameter": "dc_bias",
                "step_no": 1,
                "feature_type": "mean",
                "feature_value": 1.23,
            }
        ]
        feature_res = client.post("/parameters/bulk", json=features_payload)
        assert feature_res.status_code == 200
        assert feature_res.json() == {"ok": True, "inserted": 1}

        process_count, step_count, feature_count = _count_rows(process_id)
        assert process_count == 1
        assert step_count == 1
        assert feature_count == 1

    finally:
        deleted = client.request("DELETE", f"/processes/{process_id}")
        assert deleted.status_code == 200
        assert deleted.json()["ok"] is True

    process_count, step_count, feature_count = _count_rows(process_id)
    assert process_count == 0
    assert step_count == 0
    assert feature_count == 0


def test_db_api_bulk_empty_and_delete_missing() -> None:
    client = TestClient(db_app.app)
    missing_process_id = f"missing_{uuid4().hex}"

    step_res = client.post("/step_windows/bulk", json=[])
    assert step_res.status_code == 200
    assert step_res.json() == {"ok": True, "inserted": 0}

    feature_res = client.post("/parameters/bulk", json=[])
    assert feature_res.status_code == 200
    assert feature_res.json() == {"ok": True, "inserted": 0}

    deleted = client.request("DELETE", f"/processes/{missing_process_id}")
    assert deleted.status_code == 200
    assert deleted.json() == {"ok": True, "deleted": 0}


def test_db_api_delete_process_new_and_legacy_endpoint_consistency() -> None:
    """新旧 DELETE エンドポイントが同等の削除結果を返すことを確認する。"""
    client = TestClient(db_app.app)
    process_id_a = f"del_a_{uuid4().hex}"
    process_id_b = f"del_b_{uuid4().hex}"

    for process_id in (process_id_a, process_id_b):
        payload = {
            "process_id": process_id,
            "tool_id": "TOOL_A",
            "chamber_id": "CH1",
            "recipe_id": "UNKNOWN",
            "start_ts": datetime.now().isoformat(),
            "end_ts": datetime.now().isoformat(),
            "raw_csv_path": f"data/detail/detail_TOOL_A_CH1_{process_id}.csv",
        }
        created = client.post("/processes", json=payload)
        assert created.status_code == 200
        assert created.json()["ok"] is True

    deleted_new = client.request("DELETE", f"/processes/{process_id_a}")
    assert deleted_new.status_code == 200
    assert deleted_new.json() == {"ok": True, "deleted": 1}

    deleted_legacy = client.request("DELETE", "/processes", json={"process_id": process_id_b})
    assert deleted_legacy.status_code == 200
    assert deleted_legacy.json() == {"ok": True, "deleted": 1}
    assert deleted_legacy.headers.get("Deprecation") == "true"

    deleted_missing_new = client.request("DELETE", f"/processes/{process_id_a}")
    assert deleted_missing_new.status_code == 200
    assert deleted_missing_new.json() == {"ok": True, "deleted": 0}

    deleted_missing_legacy = client.request(
        "DELETE", "/processes", json={"process_id": process_id_b}
    )
    assert deleted_missing_legacy.status_code == 200
    assert deleted_missing_legacy.json() == {"ok": True, "deleted": 0}


def test_db_api_process_upsert_on_same_process_id() -> None:
    client = TestClient(db_app.app)
    process_id = f"upsert_{uuid4().hex}"

    first = {
        "process_id": process_id,
        "tool_id": "TOOL_A",
        "chamber_id": "CH1",
        "recipe_id": "RCP_OLD",
        "start_ts": datetime.now().isoformat(),
        "end_ts": datetime.now().isoformat(),
        "raw_csv_path": f"data/detail/detail_old_{process_id}.csv",
    }
    second = {
        "process_id": process_id,
        "tool_id": "TOOL_A",
        "chamber_id": "CH1",
        "recipe_id": "RCP_NEW",
        "start_ts": datetime.now().isoformat(),
        "end_ts": datetime.now().isoformat(),
        "raw_csv_path": f"data/detail/detail_new_{process_id}.csv",
    }

    try:
        r1 = client.post("/processes", json=first)
        assert r1.status_code == 200
        assert r1.json()["ok"] is True

        r2 = client.post("/processes", json=second)
        assert r2.status_code == 200
        assert r2.json()["ok"] is True

        con = sqlite3.connect(MAIN_DB.as_posix())
        try:
            cnt = con.execute(
                "SELECT COUNT(*) FROM processInfo WHERE process_id = ?",
                (process_id,),
            ).fetchone()[0]
            row = con.execute(
                "SELECT recipe_id, raw_csv_path FROM processInfo WHERE process_id = ?",
                (process_id,),
            ).fetchone()
        finally:
            con.close()

        assert int(cnt) == 1
        assert row is not None
        assert row[0] == "RCP_NEW"
        assert row[1] == second["raw_csv_path"]
    finally:
        resp = client.request("DELETE", f"/processes/{process_id}")
        assert resp.status_code == 200, (
            f"cleanup failed: status={resp.status_code}, body={resp.text}"
        )


def test_db_api_aggregate_write_accepts_empty_lists() -> None:
    """`/aggregate/write` が空の step_windows/parameters を受理できることを確認する。"""
    client = TestClient(db_app.app)
    process_id = f"agg_empty_{uuid4().hex}"

    payload = {
        "process": {
            "process_id": process_id,
            "tool_id": "TOOL_A",
            "chamber_id": "CH1",
            "recipe_id": "UNKNOWN",
            "start_ts": datetime.now().isoformat(),
            "end_ts": datetime.now().isoformat(),
            "raw_csv_path": f"data/detail/detail_TOOL_A_CH1_{process_id}.csv",
        },
        "step_windows": [],
        "parameters": [],
    }

    try:
        res = client.post("/aggregate/write", json=payload)
        assert res.status_code == 200
        assert res.json() == {"ok": True, "step_windows": 0, "parameters": 0}

        process_count, step_count, feature_count = _count_rows(process_id)
        assert process_count == 1
        assert step_count == 0
        assert feature_count == 0
    finally:
        resp = client.request("DELETE", f"/processes/{process_id}")
        assert resp.status_code == 200, (
            f"cleanup failed: status={resp.status_code}, body={resp.text}"
        )


def test_db_api_aggregate_write_rejects_mismatched_process_id() -> None:
    """`/aggregate/write` が process_id 不一致を 422 で拒否することを確認する。"""
    client = TestClient(db_app.app)
    process_id = f"agg_bad_{uuid4().hex}"

    payload = {
        "process": {
            "process_id": process_id,
            "tool_id": "TOOL_A",
            "chamber_id": "CH1",
            "recipe_id": "UNKNOWN",
            "start_ts": datetime.now().isoformat(),
            "end_ts": datetime.now().isoformat(),
            "raw_csv_path": f"data/detail/detail_TOOL_A_CH1_{process_id}.csv",
        },
        "step_windows": [
            {
                "process_id": f"other_{process_id}",
                "step_no": 1,
                "start_ts": datetime.now().isoformat(),
                "end_ts": datetime.now().isoformat(),
                "source_channel": "dc_bias",
            }
        ],
        "parameters": [],
    }

    res = client.post("/aggregate/write", json=payload)

    assert res.status_code == 422
    details = res.json()["detail"]
    assert any(
        "process_id must match process.process_id" in item.get("msg", "") for item in details
    )


def test_db_api_aggregate_write_returns_500_on_runner_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`/aggregate/write` 実行時例外が HTTP 500 と detail に変換されることを確認する。"""
    client = TestClient(db_app.app)
    process_id = f"agg_err_{uuid4().hex}"

    def fail_atomic(*args, **kwargs):
        _ = args, kwargs
        raise RuntimeError("forced aggregate write failure")

    monkeypatch.setattr(db_app, "write_aggregate_atomic", fail_atomic)

    payload = {
        "process": {
            "process_id": process_id,
            "tool_id": "TOOL_A",
            "chamber_id": "CH1",
            "recipe_id": "UNKNOWN",
            "start_ts": datetime.now().isoformat(),
            "end_ts": datetime.now().isoformat(),
            "raw_csv_path": f"data/detail/detail_TOOL_A_CH1_{process_id}.csv",
        },
        "step_windows": [],
        "parameters": [],
    }

    res = client.post("/aggregate/write", json=payload)

    assert res.status_code == 500
    assert "forced aggregate write failure" in res.json()["detail"]
