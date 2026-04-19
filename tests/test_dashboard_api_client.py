from __future__ import annotations

from typing import Any

import pytest

from portfolio_fdc.dashboard.api_client import (
    APIError,
    get_chart_points,
    get_charts,
    get_process_waveform_preview,
)


class _FakeResponse:
    def __init__(self, status_code: int, payload: dict[str, Any]):
        self.status_code = status_code
        self._payload = payload

    def json(self) -> dict[str, Any]:
        return self._payload


def test_get_charts_returns_data_on_ok_true(monkeypatch: pytest.MonkeyPatch) -> None:
    def _fake_get(*_args: Any, **_kwargs: Any) -> _FakeResponse:
        return _FakeResponse(200, {"ok": True, "data": [{"chart_id": "CHART_1"}]})

    monkeypatch.setattr("portfolio_fdc.dashboard.api_client.requests.get", _fake_get)

    rows = get_charts("http://localhost:8000")

    assert rows == [{"chart_id": "CHART_1"}]


def test_get_charts_raises_api_error_for_error_envelope(monkeypatch: pytest.MonkeyPatch) -> None:
    def _fake_get(*_args: Any, **_kwargs: Any) -> _FakeResponse:
        return _FakeResponse(
            404,
            {
                "ok": False,
                "error": {
                    "code": "NOT_FOUND",
                    "message": "judge result not found",
                    "details": {"result_id": "JR_999"},
                },
            },
        )

    monkeypatch.setattr("portfolio_fdc.dashboard.api_client.requests.get", _fake_get)

    with pytest.raises(APIError) as exc_info:
        get_charts("http://localhost:8000")

    assert exc_info.value.code == "NOT_FOUND"
    assert exc_info.value.message == "judge result not found"
    assert exc_info.value.status_code == 404


def test_get_charts_raises_api_error_for_detail_payload(monkeypatch: pytest.MonkeyPatch) -> None:
    def _fake_get(*_args: Any, **_kwargs: Any) -> _FakeResponse:
        return _FakeResponse(400, {"detail": "Invalid chart_id"})

    monkeypatch.setattr("portfolio_fdc.dashboard.api_client.requests.get", _fake_get)

    with pytest.raises(APIError) as exc_info:
        get_charts("http://localhost:8000")

    assert exc_info.value.message == "Invalid chart_id"
    assert exc_info.value.status_code == 400


def test_get_chart_points_returns_data(monkeypatch: pytest.MonkeyPatch) -> None:
    def _fake_get(*_args: Any, **_kwargs: Any) -> _FakeResponse:
        return _FakeResponse(
            200,
            {
                "ok": True,
                "data": [
                    {
                        "process_id": "P1",
                        "feature_value": 2.3,
                        "process_start_ts": "2026-04-19T00:00:00.000Z",
                    }
                ],
            },
        )

    monkeypatch.setattr("portfolio_fdc.dashboard.api_client.requests.get", _fake_get)

    rows = get_chart_points("http://localhost:8000", chart_id="CHART_1", params={"limit": 30})

    assert rows[0]["process_id"] == "P1"
    assert rows[0]["feature_value"] == 2.3


def test_get_process_waveform_preview_returns_data(monkeypatch: pytest.MonkeyPatch) -> None:
    def _fake_get(*_args: Any, **_kwargs: Any) -> _FakeResponse:
        return _FakeResponse(
            200,
            {
                "ok": True,
                "data": {
                    "process_id": "P1",
                    "source_path": "data/raw.csv",
                    "points": [{"x": "t1", "y": 1.2}],
                },
            },
        )

    monkeypatch.setattr("portfolio_fdc.dashboard.api_client.requests.get", _fake_get)
    data = get_process_waveform_preview(
        "http://localhost:8000", process_id="P1", params={"limit": 100}
    )

    assert data["process_id"] == "P1"
    assert data["points"][0]["y"] == 1.2
