from __future__ import annotations

import socket
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
    assert validate_base_url("http://localhost:8000")[0] == "http://localhost:8000"


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


def test_validate_base_url_rejects_zero_bind_host_when_allowed_hosts_empty(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("PORTFOLIO_DB_API_ALLOWED_HOSTS", "")

    with pytest.raises(APIError):
        validate_base_url("http://0.0.0.0:8000")


def test_validate_base_url_accepts_ipv6_loopback() -> None:
    assert validate_base_url("http://[::1]:8000")[0] == "http://[::1]:8000"


def test_validate_base_url_rejects_invalid_and_zero_ports() -> None:
    with pytest.raises(APIError):
        validate_base_url("http://localhost:70000")

    with pytest.raises(APIError):
        validate_base_url("http://localhost:0")


# --- 以下 #159 テストギャップ対応 ---


def test_validate_base_url_rejects_path_query_fragment() -> None:
    # パス付き
    with pytest.raises(APIError):
        validate_base_url("http://localhost:8000/api")
    # クエリ付き
    with pytest.raises(APIError):
        validate_base_url("http://localhost:8000?foo=bar")
    # フラグメント付き
    with pytest.raises(APIError):
        validate_base_url("http://localhost:8000#frag")


def test_validate_base_url_allowed_hosts_env(monkeypatch):
    # 許可リスト追加ホストの動作確認（mockでrestricted IPを返す）
    test_host = "example.com"
    url = f"http://{test_host}:80"

    # restricted IP (127.0.0.1) を返すようにmock
    def fake_getaddrinfo(host, port, *args, **kwargs):
        return [
            (socket.AF_INET, socket.SOCK_STREAM, 6, "", ("127.0.0.1", port)),
        ]

    monkeypatch.setattr(socket, "getaddrinfo", fake_getaddrinfo)
    # 許可リスト未設定ならreject（_is_restricted_ip分岐）
    monkeypatch.delenv("PORTFOLIO_DB_API_ALLOWED_HOSTS", raising=False)
    with pytest.raises(APIError):
        validate_base_url(url)
    # 許可リストに含めれば通る
    monkeypatch.setenv("PORTFOLIO_DB_API_ALLOWED_HOSTS", test_host)
    result = validate_base_url(url)
    assert result[1] == test_host


def test_validate_base_url_returns_correct_hostname() -> None:
    # 戻り値[1]が正規化済みhostname
    assert validate_base_url("http://LOCALHOST:8000")[1] == "localhost"
    assert validate_base_url("http://127.0.0.1:8000")[1] == "127.0.0.1"


def test_validate_base_url_ip_url_conversion(monkeypatch):
    # 非localhost外部URLのip_url変換（mockで固定IP返却）
    monkeypatch.delenv("PORTFOLIO_DB_API_ALLOWED_HOSTS", raising=False)
    test_host = "github.com"
    url = f"http://{test_host}:80"

    # 固定IP（8.8.8.8: Google Public DNS）を返すようにmock
    def fake_getaddrinfo(host, port, *args, **kwargs):
        return [
            (socket.AF_INET, socket.SOCK_STREAM, 6, "", ("8.8.8.8", port)),
        ]

    monkeypatch.setattr(socket, "getaddrinfo", fake_getaddrinfo)
    ip_url, hostname = validate_base_url(url)
    assert ip_url == "http://8.8.8.8:80"
    assert hostname == test_host
