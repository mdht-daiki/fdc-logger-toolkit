import logging
from unittest.mock import MagicMock

import pytest

from src.portfolio_fdc.dashboard.services.active_drilldown import ActiveDrilldownService
from src.portfolio_fdc.dashboard.services.chart_name_options import ChartNameOptionService
from src.portfolio_fdc.dashboard.services.navigation import NavigationService
from src.portfolio_fdc.dashboard.services.tab_load import TabLoadService
from src.portfolio_fdc.dashboard.services.url_filters import UrlFilterService


@pytest.fixture
def logger():
    return logging.getLogger("test")


@pytest.fixture
def deps():
    return MagicMock()


def test_active_drilldown_service_render_active_drilldown(logger, deps):
    service = ActiveDrilldownService(logger, deps)
    # click_dataがNoneの場合
    result = service.render_active_drilldown(None, "base_url")
    assert "Click a point" in result["layout"]["annotations"][0]["text"]


def test_active_drilldown_service_points_empty(logger, deps):
    service = ActiveDrilldownService(logger, deps)
    # pointsが空リスト
    result = service.render_active_drilldown({"points": []}, "base_url")
    assert "feature point" in result["layout"]["annotations"][0]["text"]


def test_active_drilldown_service_point_not_dict(logger, deps):
    service = ActiveDrilldownService(logger, deps)
    # points[0]がdictでない
    result = service.render_active_drilldown({"points": [123]}, "base_url")
    assert "Only feature points" in result["layout"]["annotations"][0]["text"]


def test_active_drilldown_service_process_id_invalid(logger, deps):
    service = ActiveDrilldownService(logger, deps)
    # process_idがstrでない
    result = service.render_active_drilldown({"points": [{"customdata": 123}]}, "base_url")
    assert "Only feature points" in result["layout"]["annotations"][0]["text"]


def test_active_drilldown_service_apierror(logger, deps):
    service = ActiveDrilldownService(logger, deps)
    deps.validate_base_url.return_value = "safe_url"
    deps.get_process_waveform_preview.side_effect = __import__(
        "src.portfolio_fdc.dashboard.api_client"
    ).portfolio_fdc.dashboard.api_client.APIError("msg")
    # APIError発生時
    click_data = {"points": [{"customdata": "pid"}]}
    result = service.render_active_drilldown(click_data, "base_url")
    assert "Failed to load waveform" in result["layout"]["annotations"][0]["text"]


def test_active_drilldown_service_unexpected_exception(logger, deps, caplog):
    service = ActiveDrilldownService(logger, deps)
    deps.validate_base_url.return_value = "safe_url"
    deps.get_process_waveform_preview.side_effect = Exception("unexpected")
    click_data = {"points": [{"customdata": "pid"}]}
    with caplog.at_level("ERROR"):
        result = service.render_active_drilldown(click_data, "base_url")
    assert "unexpected error" in result["layout"]["annotations"][0]["text"]
    assert any(
        "Unexpected error while rendering active drilldown" in r for r in caplog.text.splitlines()
    )


def test_chart_name_option_service_refresh_chart_name_options(logger, deps):
    service = ChartNameOptionService(logger, deps)
    deps.validate_base_url.return_value = "safe_url"
    deps.get_charts.return_value = [{"chart_id": "c1", "chart_name": "name1"}]
    options, _ = service.refresh_chart_name_options(1, "base_url", "r1", "c1")
    assert isinstance(options, list)


def test_chart_name_option_service_get_charts_unexpected_exception(logger, deps, caplog):
    service = ChartNameOptionService(logger, deps)
    deps.validate_base_url.return_value = "safe_url"
    deps.get_charts.side_effect = Exception("unexpected")
    with caplog.at_level("ERROR"):
        options, selected = service.refresh_chart_name_options(1, "base_url", "r1", "c1")
    assert options == []
    assert selected is None
    assert any(
        "Unexpected error while refreshing chart options" in r for r in caplog.text.splitlines()
    )


def test_navigation_service_move_to_active_by_chart_name():
    service = NavigationService()
    tab, chart_id, search = service.move_to_active_by_chart_name("c1", "r1", "")
    assert tab == "active"
    assert chart_id == "c1"
    assert search.startswith("?")


def test_navigation_service_selected_chart_id_none():
    service = NavigationService()
    # Noneの場合
    tab, chart_id, search = service.move_to_active_by_chart_name(None, "r1", "")
    from dash import no_update

    assert tab is no_update
    assert chart_id is no_update
    assert search is no_update


def test_navigation_service_selected_chart_id_empty():
    service = NavigationService()
    # 空文字の場合
    tab, chart_id, search = service.move_to_active_by_chart_name("", "r1", "")
    from dash import no_update

    assert tab is no_update
    assert chart_id is no_update
    assert search is no_update


def test_navigation_service_current_search_equal():
    service = NavigationService()
    # current_searchが一致する場合
    params = {"tab": "active", "chart_id": "c1", "recipe_id": "r1"}
    from urllib.parse import urlencode

    current_search = f"?{urlencode(params)}"
    tab, chart_id, search = service.move_to_active_by_chart_name("c1", "r1", current_search)
    from dash import no_update

    assert tab is no_update
    assert chart_id is no_update
    assert search is no_update


def test_tab_load_service_load_data(logger, deps):
    service = TabLoadService(logger, deps)
    deps.validate_base_url.return_value = "safe_url"
    deps.render_charts_tab.return_value = ("charts", "")
    result, _ = service.load_data("charts", 1, "base_url", "r1", "c1", "res1")
    assert result == ("charts", "") or hasattr(result, "children")


def test_url_filter_service_sync_filters_from_url():
    service = UrlFilterService()
    tab, recipe_id, chart_id, result_id = service.sync_filters_from_url(
        "?tab=active&recipe_id=r1&chart_id=c1&result_id=res1"
    )
    assert tab == "active"
    assert recipe_id == "r1"
    assert chart_id == "c1"
    assert result_id == "res1"


def test_url_filter_service_invalid_tab():
    service = UrlFilterService()
    # 不正なタブ値は"charts"にフォールバック
    tab, *_ = service.sync_filters_from_url("?tab=invalid&recipe_id=r&chart_id=c&result_id=res")
    assert tab == "charts"


def test_url_filter_service_empty_string():
    service = UrlFilterService()
    # 空文字列は全て空返却（tabは"charts"）
    tab, recipe_id, chart_id, result_id = service.sync_filters_from_url("")
    assert tab == "charts"
    assert recipe_id == ""
    assert chart_id == ""
    assert result_id == ""


def test_url_filter_service_url_encoded():
    service = UrlFilterService()
    # URLエンコードされたクエリ
    tab, recipe_id, chart_id, result_id = service.sync_filters_from_url(
        "?tab=history&recipe_id=r%201&chart_id=c%2F1&result_id=res%3D1"
    )
    assert tab == "history"
    assert recipe_id == "r 1"
    assert chart_id == "c/1"
    assert result_id == "res=1"


def test_tab_load_service_tab_branches(logger, deps):
    service = TabLoadService(logger, deps)
    deps.validate_base_url.return_value = "safe_url"
    deps.render_charts_tab.return_value = ("charts", "")
    deps.render_active_tab.return_value = ("active", "")
    deps.render_history_tab.return_value = ("history", "")
    deps.render_judge_tab.return_value = ("judge", "")
    # charts
    result, _ = service.load_data("charts", 1, "base_url", "r1", "c1", "res1")
    assert result == ("charts", "") or hasattr(result, "children")
    # active
    result, _ = service.load_data("active", 1, "base_url", "r1", "c1", "res1")
    assert result == ("active", "") or hasattr(result, "children")
    # history
    result, _ = service.load_data("history", 1, "base_url", "r1", "c1", "res1")
    assert result == ("history", "") or hasattr(result, "children")
    # judge
    result, _ = service.load_data("judge", 1, "base_url", "r1", "c1", "res1")
    assert result == ("judge", "") or hasattr(result, "children")


def test_tab_load_service_apierror_code_and_no_code(logger, deps):
    service = TabLoadService(logger, deps)
    deps.validate_base_url.return_value = "safe_url"
    # codeあり
    api_error = __import__(
        "src.portfolio_fdc.dashboard.api_client"
    ).portfolio_fdc.dashboard.api_client.APIError("msg", code="E001")
    deps.render_charts_tab.side_effect = api_error
    result, msg = service.load_data("charts", 1, "base_url", "r1", "c1", "res1")
    assert "E001" in msg
    # codeなし
    api_error2 = __import__(
        "src.portfolio_fdc.dashboard.api_client"
    ).portfolio_fdc.dashboard.api_client.APIError("msg2")
    deps.render_charts_tab.side_effect = api_error2
    result, msg = service.load_data("charts", 1, "base_url", "r1", "c1", "res1")
    assert "msg2" in msg and "[" not in msg


def test_tab_load_service_unexpected_exception(logger, deps, caplog):
    service = TabLoadService(logger, deps)
    deps.validate_base_url.return_value = "safe_url"
    deps.render_charts_tab.side_effect = Exception("unexpected")
    with caplog.at_level("ERROR"):
        result, msg = service.load_data("charts", 1, "base_url", "r1", "c1", "res1")
    assert "Unexpected error while loading dashboard data" in msg
    assert any("Unexpected error in load_data callback" in r for r in caplog.text.splitlines())
