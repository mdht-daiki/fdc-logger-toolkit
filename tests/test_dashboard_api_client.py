from __future__ import annotations

from typing import Any

import pytest

from portfolio_fdc.dashboard.api_client import (
    APIError,
    get_active_charts,
    get_chart_points,
    get_charts,
    get_charts_history,
    get_judge_result,
    get_judge_results,
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


@pytest.mark.parametrize(
    ("call", "expected_fragment"),
    [
        (lambda: get_charts("http://localhost:8000"), "/charts"),
        (lambda: get_charts_history("http://localhost:8000"), "/charts/history"),
        (
            lambda: get_chart_points("http://localhost:8000", chart_id="CHART_1"),
            "/charts/CHART_1/points",
        ),
        (lambda: get_judge_results("http://localhost:8000"), "/judge/results"),
    ],
)
def test_list_getters_raise_api_error_for_invalid_shape(
    monkeypatch: pytest.MonkeyPatch,
    call: Any,
    expected_fragment: str,
) -> None:
    def _fake_get(*_args: Any, **_kwargs: Any) -> _FakeResponse:
        return _FakeResponse(200, {"ok": True, "data": {"unexpected": True}})

    monkeypatch.setattr("portfolio_fdc.dashboard.api_client.requests.get", _fake_get)

    with pytest.raises(APIError) as exc_info:
        call()

    assert expected_fragment in exc_info.value.message
    assert "expected list" in exc_info.value.message
    assert "unexpected" in exc_info.value.message


@pytest.mark.parametrize(
    ("call", "expected_fragment"),
    [
        (lambda: get_active_charts("http://localhost:8000"), "/charts/active"),
        (
            lambda: get_process_waveform_preview("http://localhost:8000", process_id="P1"),
            "/processes/P1/waveform-preview",
        ),
        (
            lambda: get_judge_result("http://localhost:8000", result_id="JR_1"),
            "/judge/results/JR_1",
        ),
    ],
)
def test_dict_getters_raise_api_error_for_invalid_shape(
    monkeypatch: pytest.MonkeyPatch,
    call: Any,
    expected_fragment: str,
) -> None:
    def _fake_get(*_args: Any, **_kwargs: Any) -> _FakeResponse:
        return _FakeResponse(200, {"ok": True, "data": ["unexpected"]})

    monkeypatch.setattr("portfolio_fdc.dashboard.api_client.requests.get", _fake_get)

    with pytest.raises(APIError) as exc_info:
        call()

    assert expected_fragment in exc_info.value.message
    assert "expected dict" in exc_info.value.message
    assert "unexpected" in exc_info.value.message
