from __future__ import annotations

import socket
from pathlib import Path
from typing import Any

import pytest
from dash import html

from portfolio_fdc.dashboard.api_client import APIError
from portfolio_fdc.dashboard.app import (
    app,
    load_data,
    move_to_active_by_chart_name,
    refresh_change_request_listing,
    refresh_chart_name_options,
    refresh_failed_notifications,
    render_active_drilldown,
    select_chart_from_table,
    submit_change_request_apply,
    submit_change_request_approve,
    submit_change_request_create,
    submit_notification_retry,
    sync_active_selected_base_url,
    sync_filters_from_url,
    validate_base_url,
)


def _find_div_by_class_token(root: html.Div, class_token: str) -> html.Div:
    children = root.children
    if children is None:
        normalized_children: list[Any] = []
    elif isinstance(children, (list, tuple)):
        normalized_children = list(children)
    else:
        normalized_children = [children]

    for child in normalized_children:
        if not isinstance(child, html.Div):
            continue
        class_name = getattr(child, "className", "") or ""
        if class_token in class_name.split():
            return child
    raise AssertionError(f"Could not find html.Div with class token: {class_token}")


def test_dashboard_filter_controls_are_wrapping_for_narrow_viewports() -> None:
    assert isinstance(app.layout, html.Div)
    controls_row = _find_div_by_class_token(app.layout, "dashboard-filter-controls")
    assert isinstance(controls_row, html.Div)
    assert controls_row.style["display"] == "flex"
    assert controls_row.style["flexWrap"] == "wrap"
    assert controls_row.style["width"] == "100%"
    assert "dashboard-filter-controls" in controls_row.className.split()
    control_groups = controls_row.children
    for group in control_groups:
        assert "dashboard-filter-group" in group.className
        if "dashboard-filter-load" not in group.className:
            assert group.style["minWidth"] == "0"

    load_group = _find_div_by_class_token(controls_row, "dashboard-filter-load")
    assert load_group.style["minWidth"] == "96px"
    assert load_group.style["flex"] == "1 1 120px"


def test_dashboard_layout_exposes_responsive_css_hooks() -> None:
    assert isinstance(app.layout, html.Div)
    tabs_wrapper = _find_div_by_class_token(app.layout, "dashboard-tabs-wrap")
    assert isinstance(tabs_wrapper, html.Div)
    assert "dashboard-tabs-wrap" in tabs_wrapper.className.split()

    controls_row = _find_div_by_class_token(app.layout, "dashboard-filter-controls")
    load_group = _find_div_by_class_token(controls_row, "dashboard-filter-load")
    assert "dashboard-filter-load" in load_group.className

    assets_dir = (
        getattr(app, "assets_folder", None)
        or getattr(app, "assets_path", None)
        or getattr(app.config, "assets_folder", None)
    )
    assert assets_dir
    assets_css_path = Path(assets_dir).resolve() / "dashboard.css"
    assert assets_css_path.exists()

    css_path = (
        Path(__file__).resolve().parents[2]
        / "src"
        / "portfolio_fdc"
        / "dashboard"
        / "assets"
        / "dashboard.css"
    )
    assert css_path.exists()


def _extract_brace_block(text: str, after: str) -> str:
    """Return the content of the first { ... } block following `after` in `text`."""
    idx = text.find(after)
    if idx == -1:
        return ""
    brace_open = text.find("{", idx)
    if brace_open == -1:
        return ""
    depth = 0
    for i in range(brace_open, len(text)):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                return text[brace_open + 1 : i]
    return ""


def test_dashboard_css_contains_required_media_queries() -> None:
    css_path = (
        Path(__file__).resolve().parents[2]
        / "src"
        / "portfolio_fdc"
        / "dashboard"
        / "assets"
        / "dashboard.css"
    )
    css_text = css_path.read_text(encoding="utf-8")

    assert "@media" in css_text
    assert "720px" in css_text
    assert "480px" in css_text

    # overflow-x must appear inside the .dashboard-tabs-wrap selector block
    tabs_wrap_block = _extract_brace_block(css_text, ".dashboard-tabs-wrap")
    assert "overflow-x" in tabs_wrap_block

    # .dashboard-filter-group must contain width: 100% inside the 480px media block
    media_480_block = _extract_brace_block(css_text, "480px")
    filter_group_block = _extract_brace_block(media_480_block, ".dashboard-filter-group")
    assert "width: 100%" in filter_group_block


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
    content, error = load_data("active", 0, "http://localhost:8000", "", "", "", None)

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

    content, error = load_data("active", 1, "http://localhost:8000", "", "", "", None)

    assert isinstance(content, html.Div)
    assert content.children == "ACTIVE_RENDERED"
    assert error == ""


def test_load_data_renders_emergency_tab_after_load_click(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def _fake_render_emergency_tab(*_args: Any, **_kwargs: Any) -> html.Div:
        return html.Div("EMERGENCY_RENDERED")

    monkeypatch.setattr(
        "portfolio_fdc.dashboard.app._render_emergency_tab",
        _fake_render_emergency_tab,
    )

    content, error = load_data("emergency", 1, "http://localhost:8000", "", "", "", None)

    assert isinstance(content, html.Div)
    assert content.children == "EMERGENCY_RENDERED"
    assert error == ""


def test_load_data_renders_change_requests_tab_after_load_click(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def _fake_render_change_requests_tab(*_args: Any, **_kwargs: Any) -> html.Div:
        return html.Div("CHANGE_REQUESTS_RENDERED")

    monkeypatch.setattr(
        "portfolio_fdc.dashboard.app._render_change_requests_tab",
        _fake_render_change_requests_tab,
    )

    content, error = load_data("change_requests", 1, "http://localhost:8000", "", "", "", None)

    assert isinstance(content, html.Div)
    assert content.children == "CHANGE_REQUESTS_RENDERED"
    assert error == ""


def test_load_data_renders_notification_retry_tab_after_load_click(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def _fake_render_notification_retry_tab(*_args: Any, **_kwargs: Any) -> html.Div:
        return html.Div("NOTIFICATION_RETRY_RENDERED")

    monkeypatch.setattr(
        "portfolio_fdc.dashboard.app._render_notification_retry_tab",
        _fake_render_notification_retry_tab,
    )

    content, error = load_data("notification_retry", 1, "http://localhost:8000", "", "", "", None)

    assert isinstance(content, html.Div)
    assert content.children == "NOTIFICATION_RETRY_RENDERED"
    assert error == ""


def test_validate_base_url_accepts_localhost() -> None:
    assert validate_base_url("http://localhost:8000")[0] == "http://localhost:8000"


def test_load_data_rejects_invalid_base_url() -> None:
    content, error = load_data("active", 1, "file:///etc/passwd", "", "", "", None)

    assert isinstance(content, html.Div)
    assert error == "Invalid db_api base URL [INVALID_BASE_URL]"


def test_load_data_uses_chart_name_selection_as_fallback_for_chart_id(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def _fake_render_active_tab(base_url: str, recipe_id: str, chart_id: str) -> html.Div:
        return html.Div(f"ACTIVE:{chart_id}")

    monkeypatch.setattr(
        "portfolio_fdc.dashboard.app._render_active_tab",
        _fake_render_active_tab,
    )

    content, error = load_data(
        "active",
        1,
        "http://localhost:8000",
        "",
        "",
        "",
        "CHART_2",
    )

    assert isinstance(content, html.Div)
    assert content.children == "ACTIVE:CHART_2"
    assert error == ""


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


def test_submit_change_request_create_formats_success(monkeypatch: pytest.MonkeyPatch) -> None:
    def _fake_create_change_request(base_url: str, payload: dict[str, Any]) -> dict[str, Any]:
        assert base_url == "http://localhost:8000"
        assert payload["chart_id"] == 12
        assert payload["proposed_by"] == "tester"
        assert payload["expected_version"] == 1
        assert payload["idempotency_key"] == "idem-1"
        return {"request_id": 99, "status": "pending"}

    monkeypatch.setattr(
        "portfolio_fdc.dashboard.app.create_change_request", _fake_create_change_request
    )

    result = submit_change_request_create(
        1,
        "http://localhost:8000",
        "12",
        "tester",
        '{"warn_low": 20.0}',
        "1",
        "idem-1",
    )

    assert "Change request create success" in result
    assert "request_id=99" in result
    assert "status=pending" in result


def test_submit_change_request_create_surfaces_conflict_current_info(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "portfolio_fdc.dashboard.app.create_change_request",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            APIError(
                message="idempotency_key already exists",
                code="DUPLICATE_IDEMPOTENCY_KEY",
                status_code=409,
                details={"current": {"version": 3, "updated_at": "2026-05-29T00:00:00.000Z"}},
            )
        ),
    )

    result = submit_change_request_create(
        1,
        "http://localhost:8000",
        "12",
        "tester",
        '{"warn_low": 20.0}',
        "1",
        "idem-1",
    )

    assert "Change request create failed" in result
    assert "[DUPLICATE_IDEMPOTENCY_KEY]" in result
    assert "status=409" in result
    assert "current=(version=3, updated_at=2026-05-29T00:00:00.000Z)" in result


def test_submit_change_request_approve_formats_success(monkeypatch: pytest.MonkeyPatch) -> None:
    def _fake_approve_change_request(
        base_url: str, request_id: int, payload: dict[str, Any]
    ) -> dict[str, Any]:
        assert base_url == "http://localhost:8000"
        assert request_id == 88
        assert payload["approved_by"] == "ops"
        assert payload["approved_by_role"] == "manager"
        return {"request_id": 88, "status": "approved"}

    monkeypatch.setattr(
        "portfolio_fdc.dashboard.app.approve_change_request", _fake_approve_change_request
    )

    result = submit_change_request_approve(
        1,
        "http://localhost:8000",
        "88",
        "ops",
        "manager",
        "looks good",
    )

    assert "Change request approve success" in result
    assert "request_id=88" in result
    assert "status=approved" in result


def test_submit_change_request_approve_surfaces_server_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "portfolio_fdc.dashboard.app.approve_change_request",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(APIError(message="boom", status_code=500)),
    )

    result = submit_change_request_approve(
        1,
        "http://localhost:8000",
        "88",
        "ops",
        "manager",
        "looks good",
    )

    assert "Change request approve failed" in result
    assert "status=500" in result


def test_submit_change_request_apply_formats_success(monkeypatch: pytest.MonkeyPatch) -> None:
    def _fake_apply_change_request(
        base_url: str, request_id: int, payload: dict[str, Any]
    ) -> dict[str, Any]:
        assert base_url == "http://localhost:8000"
        assert request_id == 77
        assert payload["applied_by"] == "ops"
        assert payload["applied_by_role"] == "operator"
        return {"request_id": 77, "status": "applied", "resulting_version": 3, "noop": False}

    monkeypatch.setattr(
        "portfolio_fdc.dashboard.app.apply_change_request", _fake_apply_change_request
    )

    result = submit_change_request_apply(
        1,
        "http://localhost:8000",
        "77",
        "ops",
        "operator",
        "apply reason",
    )

    assert "Change request apply success" in result
    assert "request_id=77" in result
    assert "status=applied" in result
    assert "resulting_version=3" in result


def test_refresh_change_request_listing_surfaces_validation_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "portfolio_fdc.dashboard.app.get_change_requests",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            APIError(message="invalid filter", code="VALIDATION_ERROR", status_code=422)
        ),
    )

    result_text, list_block, detail_block = refresh_change_request_listing(
        1,
        "http://localhost:8000",
        "pending",
        "21",
        "",
        "",
        "10",
        "0",
        "11",
    )

    assert "Change request list failed" in result_text
    assert "status=422" in result_text
    assert isinstance(list_block, list)
    assert isinstance(detail_block, list)


def test_refresh_change_request_listing_renders_rows_and_detail(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def _fake_get_change_requests(
        base_url: str, params: dict[str, Any] | None = None
    ) -> list[dict[str, Any]]:
        assert base_url == "http://localhost:8000"
        assert params is not None
        assert params["limit"] == 10
        assert params["offset"] == 0
        return [
            {
                "id": 11,
                "chart_id": 21,
                "status": "pending",
                "proposed_by": "tester",
                "proposed_at": "2026-05-29T00:00:00.000Z",
                "change_payload": '{"warn_low": 20.0}',
                "expected_version": 1,
                "idempotency_key": "idem-11",
            }
        ]

    monkeypatch.setattr(
        "portfolio_fdc.dashboard.app.get_change_requests", _fake_get_change_requests
    )

    result_text, list_block, detail_block = refresh_change_request_listing(
        1,
        "http://localhost:8000",
        "pending",
        "21",
        "",
        "",
        "10",
        "0",
        "11",
    )

    assert result_text == "Loaded 1 change request(s)"
    assert isinstance(list_block, list)
    assert isinstance(detail_block, list)
    assert any(
        getattr(child, "children", None) == "Change Requests: 1 rows" for child in list_block
    )
    assert any(
        "request_id=11" in getattr(child, "children", "")
        for child in detail_block
        if hasattr(child, "children")
    )


def test_refresh_failed_notifications_renders_rows(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def _fake_get_failed_notifications(
        base_url: str, params: dict[str, Any] | None = None
    ) -> list[dict[str, Any]]:
        assert base_url == "http://localhost:8000"
        assert params is not None
        assert params["limit"] == 10
        assert params["offset"] == 0
        return [
            {
                "event_id": 101,
                "status": "failed",
                "retry_count": 2,
                "next_retry_at": "2026-05-31T00:10:00.000Z",
                "last_attempt_at": "2026-05-31T00:05:00.000Z",
                "last_error": "smtp timeout",
            }
        ]

    monkeypatch.setattr(
        "portfolio_fdc.dashboard.app.get_failed_notifications", _fake_get_failed_notifications
    )

    result_text, rows = refresh_failed_notifications(
        1,
        "http://localhost:8000",
        "",
        "10",
        "0",
    )

    assert result_text == "Loaded 1 failed notification(s)"
    assert len(rows) == 1
    assert rows[0]["event_id"] == 101
    assert rows[0]["status"] == "failed"


def test_submit_notification_retry_formats_success(monkeypatch: pytest.MonkeyPatch) -> None:
    def _fake_retry_notification(base_url: str, event_id: int) -> dict[str, Any]:
        assert base_url == "http://localhost:8000"
        assert event_id == 101
        return {
            "event_id": 101,
            "status": "pending",
            "retry_count": 2,
            "next_retry_at": "2026-05-31T00:10:00.000Z",
        }

    monkeypatch.setattr("portfolio_fdc.dashboard.app.retry_notification", _fake_retry_notification)

    result = submit_notification_retry(1, "http://localhost:8000", "101")

    assert "Notification retry success" in result
    assert "event_id=101" in result
    assert "status=pending" in result
    assert "retry_count=2" in result


def test_submit_notification_retry_surfaces_api_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "portfolio_fdc.dashboard.app.retry_notification",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            APIError(message="only failed notification can be retried", status_code=400)
        ),
    )

    result = submit_notification_retry(1, "http://localhost:8000", "101")

    assert "Notification retry failed" in result
    assert "status=400" in result


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


# --- #153 / #161: sync_filters_from_url ---


def test_sync_filters_from_url_accepts_valid_tabs() -> None:
    for tab in (
        "charts",
        "active",
        "history",
        "judge",
        "change_requests",
        "emergency",
        "notification_retry",
    ):
        result_tab, _, _, _ = sync_filters_from_url(f"?tab={tab}")
        assert result_tab == tab


def test_sync_filters_from_url_falls_back_to_charts_for_invalid_tab() -> None:
    result_tab, _, _, _ = sync_filters_from_url("?tab=unknown")
    assert result_tab == "charts"


def test_sync_filters_from_url_parses_multiple_query_params() -> None:
    tab, recipe_id, chart_id, result_id = sync_filters_from_url(
        "?tab=judge&recipe_id=RCP_1&chart_id=CHART_2&result_id=JR_3"
    )
    assert tab == "judge"
    assert recipe_id == "RCP_1"
    assert chart_id == "CHART_2"
    assert result_id == "JR_3"


def test_sync_filters_from_url_handles_empty_search() -> None:
    tab, recipe_id, chart_id, result_id = sync_filters_from_url("")
    assert tab == "charts"
    assert recipe_id == ""
    assert chart_id == ""
    assert result_id == ""


# --- #153 / #161: move_to_active_by_chart_name ---


def test_move_to_active_by_chart_name_returns_no_update_for_none() -> None:
    from dash import no_update

    tab, chart_id, search = move_to_active_by_chart_name(None, "", "")
    assert tab is no_update
    assert chart_id is no_update
    assert search is no_update


def test_move_to_active_by_chart_name_switches_tab_and_updates_url() -> None:
    tab, chart_id, search = move_to_active_by_chart_name("CHART_1", "RCP_1", "")
    assert tab == "active"
    assert chart_id == "CHART_1"
    assert "tab=active" in search
    assert "recipe_id=RCP_1" in search
    assert "chart_id=CHART_1" in search


# --- #153 / #161: select_chart_from_table ---


def test_select_chart_from_table_returns_no_update_for_none_cell() -> None:
    from dash import no_update

    result = select_chart_from_table(None)
    assert result is no_update


def test_select_chart_from_table_returns_no_update_when_data_key_missing() -> None:
    from dash import no_update

    result = select_chart_from_table({"rowIndex": 0})
    assert result is no_update


def test_select_chart_from_table_returns_chart_id_on_valid_cell() -> None:
    result = select_chart_from_table({"rowIndex": 1, "data": {"chart_id": "CHART_2"}})
    assert result == "CHART_2"


# --- #153 / #161: sync_active_selected_base_url ---


def test_sync_active_selected_base_url_returns_given_url() -> None:
    result = sync_active_selected_base_url("http://localhost:8000")
    assert result == "http://localhost:8000"


# --- #153 / #161: render_active_drilldown ---


def test_render_active_drilldown_returns_empty_figure_for_none_click_data() -> None:
    figure = render_active_drilldown(None, "http://localhost:8000")
    assert figure["layout"]["title"] == "Raw Waveform Drilldown"


def test_render_active_drilldown_returns_empty_figure_for_empty_points() -> None:
    figure = render_active_drilldown({"points": []}, "http://localhost:8000")
    assert figure["layout"]["title"] == "Raw Waveform Drilldown"


def test_render_active_drilldown_returns_empty_figure_when_customdata_not_str() -> None:
    figure = render_active_drilldown({"points": [{"customdata": 123}]}, "http://localhost:8000")
    assert figure["layout"]["title"] == "Raw Waveform Drilldown"


def test_render_active_drilldown_returns_waveform_figure_on_success(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "portfolio_fdc.dashboard.app.validate_base_url",
        lambda url: (url, "localhost"),
    )

    def _fake_preview(base_url: str, process_id: str, params: Any = None) -> dict[str, Any]:
        return {"process_id": process_id, "points": [{"x": "t1", "y": 1.0}]}

    monkeypatch.setattr(
        "portfolio_fdc.dashboard.app.get_process_waveform_preview",
        _fake_preview,
    )

    click_data = {"points": [{"customdata": "P1"}]}
    figure = render_active_drilldown(click_data, "http://localhost:8000")
    assert figure["layout"]["title"] == "Raw Waveform Preview (P1)"


def test_render_active_drilldown_returns_empty_figure_on_api_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "portfolio_fdc.dashboard.app.validate_base_url",
        lambda url: (url, "localhost"),
    )

    def _raise_api_error(*_args: Any, **_kwargs: Any) -> None:
        raise APIError(message="not found")

    monkeypatch.setattr(
        "portfolio_fdc.dashboard.app.get_process_waveform_preview",
        _raise_api_error,
    )

    click_data = {"points": [{"customdata": "P1"}]}
    figure = render_active_drilldown(click_data, "http://localhost:8000")
    assert figure["layout"]["title"] == "Raw Waveform Drilldown"
    annotations = figure["layout"].get("annotations", [])
    error_text = " ".join(a.get("text", "") for a in annotations)
    assert "not found" in error_text
