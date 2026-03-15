from __future__ import annotations

import sqlite3
from collections.abc import Iterator, Mapping
from datetime import datetime, timedelta
from email.utils import parsedate_to_datetime
from urllib.parse import quote
from uuid import uuid4

import pytest
from fastapi.testclient import TestClient

from portfolio_fdc.db_api import app as db_app
from portfolio_fdc.db_api.db import MAIN_DB

pytestmark = pytest.mark.integration


@pytest.fixture
def client() -> Iterator[TestClient]:
    """Lifespan を含めて DB API アプリへアクセスする TestClient を提供する。"""
    with TestClient(db_app.app) as test_client:
        yield test_client


def _count_rows(process_id: str) -> tuple[int, int, int]:
    """指定 process_id の ProcessInfo/StepWindows/Parameters 件数を返す。"""
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


def build_process_payload(process_id: str) -> dict[str, str]:
    """ProcessInfo 作成 API 用の標準 payload を返す。"""
    return {
        "process_id": process_id,
        "tool_id": "TOOL_A",
        "chamber_id": "CH1",
        "recipe_id": "UNKNOWN",
        "start_ts": datetime.now().isoformat(),
        "end_ts": datetime.now().isoformat(),
        "raw_csv_path": f"data/detail/detail_TOOL_A_CH1_{process_id}.csv",
    }


def assert_legacy_migration_headers(
    response_headers: Mapping[str, str],
    process_id: str | None,
    expected_sunset: str,
) -> None:
    """legacy DELETE 応答の移行ヘッダ契約を検証する。"""
    assert response_headers.get("Deprecation") == "true"
    sunset_str = response_headers.get("Sunset")
    assert sunset_str == expected_sunset
    parsed = parsedate_to_datetime(sunset_str)
    assert parsed.tzinfo is not None
    assert parsed.utcoffset() == timedelta(0)

    if process_id is None:
        expected_link = '</processes>; rel="successor-version"'
    else:
        expected_link = f'</processes/{quote(process_id, safe="")}>; rel="successor-version"'
    assert response_headers.get("Link") == expected_link


def test_db_api_minimum_flow_for_aggregate_contract(client: TestClient) -> None:
    """process/step/parameter の最小フローが保存・削除まで成立することを確認する。"""
    process_id = f"it_{uuid4().hex}"

    process_payload = build_process_payload(process_id)

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


def test_db_api_bulk_empty_and_delete_missing(client: TestClient) -> None:
    """空 bulk と存在しない process_id の削除が成功応答になることを確認する。"""
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


def test_db_api_delete_process_new_and_legacy_endpoint_consistency(client: TestClient) -> None:
    """新旧 DELETE エンドポイントが同等の削除結果を返すことを確認する。"""
    process_id_a = f"del_a_{uuid4().hex}"
    process_id_b = f"del_b_{uuid4().hex}"
    try:
        for process_id in (process_id_a, process_id_b):
            payload = build_process_payload(process_id)
            created = client.post("/processes", json=payload)
            assert created.status_code == 200
            assert created.json()["ok"] is True

        deleted_new = client.request("DELETE", f"/processes/{quote(process_id_a, safe='')}")
        assert deleted_new.status_code == 200
        assert deleted_new.json() == {"ok": True, "deleted": 1}

        deleted_legacy = client.request("DELETE", "/processes", json={"process_id": process_id_b})
        assert deleted_legacy.status_code == 200
        assert deleted_legacy.json() == {"ok": True, "deleted": 1}
        assert_legacy_migration_headers(
            deleted_legacy.headers,
            process_id_b,
            db_app.LEGACY_DELETE_PROCESSES_SUNSET,
        )

        deleted_missing_new = client.request("DELETE", f"/processes/{quote(process_id_a, safe='')}")
        assert deleted_missing_new.status_code == 200
        assert deleted_missing_new.json() == {"ok": True, "deleted": 0}

        deleted_missing_legacy = client.request(
            "DELETE", "/processes", json={"process_id": process_id_b}
        )
        assert deleted_missing_legacy.status_code == 200
        assert deleted_missing_legacy.json() == {"ok": True, "deleted": 0}
        assert_legacy_migration_headers(
            deleted_missing_legacy.headers,
            process_id_b,
            db_app.LEGACY_DELETE_PROCESSES_SUNSET,
        )
    finally:
        cleanup_new = client.request("DELETE", f"/processes/{quote(process_id_a, safe='')}")
        assert cleanup_new.status_code == 200, (
            f"cleanup_new failed: status={cleanup_new.status_code}, body={cleanup_new.text}"
        )
        cleanup_legacy = client.request("DELETE", "/processes", json={"process_id": process_id_b})
        assert cleanup_legacy.status_code == 200, (
            f"cleanup_legacy failed: status={cleanup_legacy.status_code}, "
            f"body={cleanup_legacy.text}"
        )


def test_db_api_legacy_delete_validation_error_still_has_migration_headers(
    client: TestClient,
) -> None:
    """`DELETE /processes` の 422 応答にも移行ヘッダが付与されることを確認する。"""
    res = client.request("DELETE", "/processes", json={})

    assert res.status_code == 422
    assert_legacy_migration_headers(res.headers, None, db_app.LEGACY_DELETE_PROCESSES_SUNSET)


def test_db_api_process_upsert_on_same_process_id(client: TestClient) -> None:
    """同一 process_id の再登録で upsert されることを確認する。"""
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


def test_db_api_aggregate_write_accepts_empty_lists(client: TestClient) -> None:
    """`/aggregate/write` が空の step_windows/parameters を受理できることを確認する。"""
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


def test_db_api_aggregate_write_rejects_mismatched_process_id(client: TestClient) -> None:
    """`/aggregate/write` が process_id 不一致を 422 で拒否することを確認する。"""
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
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`/aggregate/write` 実行時例外が HTTP 500 と detail に変換されることを確認する。"""
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


def test_db_api_legacy_delete_preserves_migration_headers_on_error(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`/processes` の例外応答でも移行ヘッダが維持されることを確認する。"""

    def fail_delete(*args, **kwargs):
        """テスト用: legacy delete を強制失敗させる。"""
        _ = args, kwargs
        raise RuntimeError("forced delete failure")

    monkeypatch.setattr(db_app, "delete_process", fail_delete)

    process_id = f"legacy_err_{uuid4().hex}"
    res = client.request("DELETE", "/processes", json={"process_id": process_id})

    assert res.status_code == 500
    assert "forced delete failure" in res.json()["detail"]
    assert_legacy_migration_headers(
        res.headers,
        process_id,
        db_app.LEGACY_DELETE_PROCESSES_SUNSET,
    )


def test_db_api_delete_by_path_accepts_process_id_with_slash(client: TestClient) -> None:
    """`/processes/{process_id:path}` が `/` を含む process_id を削除できることを確認する。"""
    process_id = f"tool/A/{uuid4().hex}"
    payload = build_process_payload(process_id)

    try:
        created = client.post("/processes", json=payload)
        assert created.status_code == 200
        assert created.json() == {"ok": True}

        deleted = client.request("DELETE", f"/processes/{quote(process_id, safe='')}")
        assert deleted.status_code == 200
        assert deleted.json() == {"ok": True, "deleted": 1}
    finally:
        # Ensure cleanup even when assertions fail midway.
        cleanup = client.request("DELETE", f"/processes/{quote(process_id, safe='')}")
        assert cleanup.status_code == 200, (
            f"cleanup failed: status={cleanup.status_code}, body={cleanup.text}"
        )
