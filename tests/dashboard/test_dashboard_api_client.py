from __future__ import annotations

from typing import Any

import pytest
import requests

from portfolio_fdc.dashboard.api_client import (
    APIError,
    _request_envelope,
    get_active_charts,
    get_chart_points,
    get_charts,
    get_charts_history,
    get_judge_result,
    get_judge_results,
    get_process_waveform_preview,
    parse_api_error,
    parse_utc_millis,
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


def test_path_segment_encodes_special_chars(monkeypatch):
    # chart_idに/や%など特殊文字を含めた場合のエンドポイントパスを検証

    import urllib.parse

    called: dict[str, Any] = {}

    def _fake_get(url: str, *args, **kwargs) -> _FakeResponse:
        called["url"] = url
        # レスポンスは最低限でOK
        return _FakeResponse(200, {"ok": True, "data": []})

    monkeypatch.setattr("portfolio_fdc.dashboard.api_client.requests.get", _fake_get)

    # / → %2F, % → %25 など
    chart_id = "CHART/1%"
    get_chart_points("http://localhost:8000", chart_id=chart_id)
    # 呼び出しURLのパス部分を厳密に検証
    parsed = urllib.parse.urlparse(called["url"])
    assert parsed.path == "/charts/CHART%2F1%25/points"


# --- #153: parse_utc_millis ---


def test_parse_utc_millis_returns_dash_for_none() -> None:
    assert parse_utc_millis(None) == "-"


def test_parse_utc_millis_returns_dash_for_empty_string() -> None:
    assert parse_utc_millis("") == "-"


def test_parse_utc_millis_converts_z_suffix_to_utc() -> None:
    result = parse_utc_millis("2026-04-19T00:00:00.123Z")
    assert result == "2026-04-19 00:00:00.123 UTC"


def test_parse_utc_millis_converts_string_without_millis() -> None:
    result = parse_utc_millis("2026-04-19T00:00:00Z")
    assert result == "2026-04-19 00:00:00.000 UTC"


def test_parse_utc_millis_returns_original_for_invalid_string() -> None:
    result = parse_utc_millis("not-a-timestamp")
    assert result == "not-a-timestamp"


def test_parse_utc_millis_converts_non_utc_timezone_to_utc() -> None:
    # +09:00 → UTC (9時間引く)
    result = parse_utc_millis("2026-04-19T09:00:00.000+09:00")
    assert result == "2026-04-19 00:00:00.000 UTC"


# --- #153: parse_api_error ---


def test_parse_api_error_with_error_envelope_returns_code_and_message() -> None:
    payload = {"error": {"code": "NOT_FOUND", "message": "resource not found"}}
    err = parse_api_error(payload, 404)
    assert err.code == "NOT_FOUND"
    assert err.message == "resource not found"
    assert err.status_code == 404


def test_parse_api_error_with_detail_key_returns_detail_message() -> None:
    payload = {"detail": "Invalid input"}
    err = parse_api_error(payload, 400)
    assert err.message == "Invalid input"
    assert err.status_code == 400


def test_parse_api_error_with_no_error_or_detail_returns_generic() -> None:
    payload: dict[str, Any] = {"ok": False}
    err = parse_api_error(payload, 500)
    assert err.message == "API error (status=500)"
    assert err.status_code == 500


def test_parse_api_error_with_non_dict_payload_returns_generic() -> None:
    err = parse_api_error(None, 503)  # type: ignore[arg-type]
    assert err.message == "API error (status=503)"
    assert err.status_code == 503


# --- #153: _request_envelope ネットワークエラー ---


def test_request_envelope_converts_request_exception_to_api_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def _raise_network_error(*_args: Any, **_kwargs: Any) -> None:
        raise requests.RequestException("connection refused")

    monkeypatch.setattr("portfolio_fdc.dashboard.api_client.requests.get", _raise_network_error)

    with pytest.raises(APIError) as exc_info:
        _request_envelope("http://localhost:8000", "/charts")

    assert "Network error" in exc_info.value.message
    assert "connection refused" in exc_info.value.message
