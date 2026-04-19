from __future__ import annotations

from typing import Any

import pytest
from dash import html

from portfolio_fdc.dashboard.api_client import APIError
from portfolio_fdc.dashboard.app import load_data, refresh_chart_name_options, validate_base_url


def test_refresh_chart_name_options_keeps_dropdown_unselected_without_chart_id(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def _fake_get_charts(*_args: Any, **_kwargs: Any) -> list[dict[str, Any]]:
        return [
            {"chart_id": "CHART_1", "chart_name": "Chart One"},
            {"chart_id": "CHART_2", "chart_name": "Chart Two"},
        ]

    monkeypatch.setattr("portfolio_fdc.dashboard.app.get_charts", _fake_get_charts)

    options, value = refresh_chart_name_options(1, "http://localhost:8000", "", "")

    assert [option["value"] for option in options] == ["CHART_1", "CHART_2"]
    assert value is None


def test_refresh_chart_name_options_preserves_explicit_chart_id(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def _fake_get_charts(*_args: Any, **_kwargs: Any) -> list[dict[str, Any]]:
        return [
            {"chart_id": "CHART_1", "chart_name": "Chart One"},
            {"chart_id": "CHART_2", "chart_name": "Chart Two"},
        ]

    monkeypatch.setattr("portfolio_fdc.dashboard.app.get_charts", _fake_get_charts)

    options, value = refresh_chart_name_options(
        1,
        "http://localhost:8000",
        "",
        "CHART_2",
    )

    assert [option["value"] for option in options] == ["CHART_1", "CHART_2"]
    assert value == "CHART_2"


def test_refresh_chart_name_options_does_not_fetch_before_load_click(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    called = False

    def _fake_get_charts(*_args: Any, **_kwargs: Any) -> list[dict[str, Any]]:
        nonlocal called
        called = True
        return [{"chart_id": "CHART_1", "chart_name": "Chart One"}]

    monkeypatch.setattr("portfolio_fdc.dashboard.app.get_charts", _fake_get_charts)

    options, value = refresh_chart_name_options(0, "http://localhost:8000", "", "")

    assert options == []
    assert value is None
    assert called is False


def test_load_data_shows_prompt_before_first_load_click() -> None:
    content, error = load_data("active", 0, "http://localhost:8000", "", "", "")

    assert isinstance(content, html.Div)
    assert content.children == "Press Load to fetch data"
    assert error == ""


def test_load_data_renders_active_tab_after_load_click(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def _fake_render_active_tab(*_args: Any, **_kwargs: Any) -> html.Div:
        return html.Div("ACTIVE_RENDERED")

    monkeypatch.setattr(
        "portfolio_fdc.dashboard.app._render_active_tab",
        _fake_render_active_tab,
    )

    content, error = load_data("active", 1, "http://localhost:8000", "", "", "")

    assert isinstance(content, html.Div)
    assert content.children == "ACTIVE_RENDERED"
    assert error == ""


def test_validate_base_url_accepts_localhost() -> None:
    assert validate_base_url("http://localhost:8000") == "http://localhost:8000"


def test_load_data_rejects_invalid_base_url() -> None:
    content, error = load_data("active", 1, "file:///etc/passwd", "", "", "")

    assert isinstance(content, html.Div)
    assert error == "Invalid db_api base URL [INVALID_BASE_URL]"


def test_refresh_chart_name_options_rejects_invalid_base_url(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    called = False

    def _fake_get_charts(*_args: Any, **_kwargs: Any) -> list[dict[str, Any]]:
        nonlocal called
        called = True
        return [{"chart_id": "CHART_1", "chart_name": "Chart One"}]

    monkeypatch.setattr("portfolio_fdc.dashboard.app.get_charts", _fake_get_charts)

    options, value = refresh_chart_name_options(1, "ftp://localhost:8000", "", "")

    assert options == []
    assert value is None
    assert called is False


def test_validate_base_url_rejects_credentialed_url() -> None:
    with pytest.raises(APIError):
        validate_base_url("http://user:pass@localhost:8000")


def test_validate_base_url_rejects_zero_bind_host_by_default() -> None:
    with pytest.raises(APIError):
        validate_base_url("http://0.0.0.0:8000")
