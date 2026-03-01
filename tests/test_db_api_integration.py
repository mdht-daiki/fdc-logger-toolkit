from __future__ import annotations

from datetime import datetime
from uuid import uuid4

import pytest
from fastapi.testclient import TestClient

from portfolio_fdc.db_api import app as db_app

pytestmark = pytest.mark.integration


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

    finally:
        deleted = client.request("DELETE", "/processes", json={"process_id": process_id})
        assert deleted.status_code == 200
        assert deleted.json()["ok"] is True
