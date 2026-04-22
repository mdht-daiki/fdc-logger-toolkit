import logging
from unittest.mock import MagicMock

import pytest

from src.portfolio_fdc.dashboard.api_client import APIError
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


def test_active_drilldown_service_validate_base_url_apierror(logger, deps):
    service = ActiveDrilldownService(logger, deps)
    deps.validate_base_url.side_effect = APIError("msg")
    click_data = {"points": [{"customdata": "pid"}]}
    result = service.render_active_drilldown(click_data, "base_url")
    text = result["layout"]["annotations"][0]["text"]
    assert "Failed to load waveform" in text
    assert "msg" in text


def test_active_drilldown_service_preview_not_dict(logger, deps):
    service = ActiveDrilldownService(logger, deps)
    deps.validate_base_url.return_value = "safe_url"
    # previewがstr
    deps.get_process_waveform_preview.return_value = "not_a_dict"
    click_data = {"points": [{"customdata": "pid"}]}
    result = service.render_active_drilldown(click_data, "base_url")
    text = result["layout"]["annotations"][0]["text"]
    assert "No waveform preview data" in text or "No waveform" in text
    # previewがlist
    deps.get_process_waveform_preview.return_value = [1, 2, 3]
    result = service.render_active_drilldown(click_data, "base_url")
    text = result["layout"]["annotations"][0]["text"]
    assert "No waveform preview data" in text or "No waveform" in text


def test_active_drilldown_service_customdata_empty_string(logger, deps):
    service = ActiveDrilldownService(logger, deps)
    click_data = {"points": [{"customdata": ""}]}
    result = service.render_active_drilldown(click_data, "base_url")
    text = result["layout"]["annotations"][0]["text"]
    assert "Only feature points" in text


def test_active_drilldown_service_apierror(logger, deps):
    service = ActiveDrilldownService(logger, deps)
    deps.validate_base_url.return_value = "safe_url"
    deps.get_process_waveform_preview.side_effect = APIError("msg")
    # APIError発生時
    click_data = {"points": [{"customdata": "pid"}]}
    result = service.render_active_drilldown(click_data, "base_url")
    # 厳密にAPIError.messageが含まれることを検証
    text = result["layout"]["annotations"][0]["text"]
    assert "Failed to load waveform" in text
    assert "msg" in text


def test_active_drilldown_service_unexpected_exception(logger, deps, caplog):
    service = ActiveDrilldownService(logger, deps)
    deps.validate_base_url.return_value = "safe_url"
    deps.get_process_waveform_preview.side_effect = Exception("unexpected")
    click_data = {"points": [{"customdata": "pid"}]}
    with caplog.at_level(logging.ERROR, logger="test"):
        result = service.render_active_drilldown(click_data, "base_url")
    assert "unexpected error" in result["layout"]["annotations"][0]["text"]
    assert any(
        r.levelname == "ERROR"
        and "Unexpected error while rendering active drilldown" in r.getMessage()
        for r in caplog.records
    )


def test_chart_name_option_service_refresh_chart_name_options(logger, deps):
    service = ChartNameOptionService(logger, deps)
    deps.validate_base_url.return_value = "safe_url"
    deps.get_charts.return_value = [{"chart_id": "c1", "chart_name": "name1"}]
    options, selected = service.refresh_chart_name_options(1, "base_url", "r1", "c1")
    assert isinstance(options, list)
    assert len(options) == 1
    assert options[0]["value"] == "c1"
    assert "name1" in options[0]["label"]
    assert selected == "c1"


def test_chart_name_option_service_get_charts_unexpected_exception(logger, deps, caplog):
    service = ChartNameOptionService(logger, deps)
    deps.validate_base_url.return_value = "safe_url"
    deps.get_charts.side_effect = Exception("unexpected")
    with caplog.at_level(logging.ERROR, logger="test"):
        options, selected = service.refresh_chart_name_options(1, "base_url", "r1", "c1")
    assert options == []
    assert selected is None
    assert any(
        r.levelname == "ERROR"
        and "Unexpected error while refreshing chart options" in r.getMessage()
        for r in caplog.records
    )


def test_navigation_service_move_to_active_by_chart_name():
    def test_navigation_service_move_to_active_by_chart_name_recipe_id_omitted():
        service = NavigationService()
        tab, chart_id, search = service.move_to_active_by_chart_name("c1", "", "")
        import urllib.parse

        parsed = urllib.parse.parse_qs(search.lstrip("?"))
        assert "recipe_id" not in parsed
        assert parsed["tab"] == ["active"]
        assert parsed["chart_id"] == ["c1"]

    import urllib.parse

    service = NavigationService()
    tab, chart_id, search = service.move_to_active_by_chart_name("c1", "r1", "")
    assert tab == "active"
    assert chart_id == "c1"
    assert search.startswith("?")
    # searchの内容を厳密に検証
    parsed = urllib.parse.parse_qs(search.lstrip("?"))
    assert parsed["tab"] == ["active"]
    assert parsed["chart_id"] == ["c1"]
    assert parsed["recipe_id"] == ["r1"]


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
    # n_clicks=0 early return
    result, msg = service.load_data("charts", 0, "base_url", "r1", "c1", "res1")
    from dash import html

    assert isinstance(result, html.Div)
    assert "Press Load" in result.children
    assert msg == ""

    deps.validate_base_url.return_value = "safe_url"
    deps.render_charts_tab.return_value = ("charts", "")
    result, msg = service.load_data("charts", 1, "base_url", "r1", "c1", "res1")
    assert result == ("charts", "")
    assert msg == ""


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


def test_url_filter_service_sync_filters_from_url_none():
    service = UrlFilterService()
    # Noneの場合
    tab, recipe_id, chart_id, result_id = service.sync_filters_from_url(None)
    assert tab == "charts"
    assert recipe_id == ""
    assert chart_id == ""
    assert result_id == ""


def test_url_filter_service_sync_filters_from_url_partial_keys():
    def test_url_filter_service_tab_judge():
        service = UrlFilterService()
        tab, recipe_id, chart_id, result_id = service.sync_filters_from_url(
            "?tab=judge&recipe_id=r1&chart_id=c1&result_id=res1"
        )
        assert tab == "judge"
        assert recipe_id == "r1"
        assert chart_id == "c1"
        assert result_id == "res1"

    service = UrlFilterService()
    # recipe_idのみ
    tab, recipe_id, chart_id, result_id = service.sync_filters_from_url("?recipe_id=r1")
    assert tab == "charts"
    assert recipe_id == "r1"
    assert chart_id == ""
    assert result_id == ""

    # tab, chart_idのみ
    tab, recipe_id, chart_id, result_id = service.sync_filters_from_url("?tab=history&chart_id=c1")
    assert tab == "history"
    assert recipe_id == ""
    assert chart_id == "c1"
    assert result_id == ""

    # tabが不正かつ他キーなし
    tab, recipe_id, chart_id, result_id = service.sync_filters_from_url("?tab=invalid")
    assert tab == "charts"
    assert recipe_id == ""
    assert chart_id == ""
    assert result_id == ""


def test_tab_load_service_tab_branches(logger, deps):
    service = TabLoadService(logger, deps)
    deps.validate_base_url.return_value = "safe_url"
    deps.render_charts_tab.return_value = ("charts", "")
    deps.render_active_tab.return_value = ("active", "")
    deps.render_history_tab.return_value = ("history", "")
    deps.render_judge_tab.return_value = ("judge", "")
    # charts
    result, msg = service.load_data("charts", 1, "base_url", "r1", "c1", "res1")
    assert result == ("charts", "")
    assert msg == ""
    # active
    result, msg = service.load_data("active", 1, "base_url", "r1", "c1", "res1")
    assert result == ("active", "")
    assert msg == ""
    # history
    result, msg = service.load_data("history", 1, "base_url", "r1", "c1", "res1")
    assert result == ("history", "")
    assert msg == ""
    # judge
    result, msg = service.load_data("judge", 1, "base_url", "r1", "c1", "res1")
    assert result == ("judge", "")
    assert msg == ""


def test_tab_load_service_apierror_code_and_no_code(logger, deps):
    service = TabLoadService(logger, deps)
    deps.validate_base_url.return_value = "safe_url"
    # codeあり
    api_error = APIError("msg", code="E001")
    deps.render_charts_tab.side_effect = api_error
    result, msg = service.load_data("charts", 1, "base_url", "r1", "c1", "res1")
    assert msg == "msg [E001]"
    # codeなし
    api_error2 = APIError("msg2")
    deps.render_charts_tab.side_effect = api_error2
    result, msg = service.load_data("charts", 1, "base_url", "r1", "c1", "res1")
    assert msg == "msg2"


def test_tab_load_service_unexpected_exception(logger, deps, caplog):
    service = TabLoadService(logger, deps)
    deps.validate_base_url.return_value = "safe_url"
    deps.render_charts_tab.side_effect = Exception("unexpected")
    with caplog.at_level(logging.ERROR, logger="test"):
        result, msg = service.load_data("charts", 1, "base_url", "r1", "c1", "res1")
    assert "Unexpected error while loading dashboard data" in msg
    assert any(
        r.levelname == "ERROR" and "Unexpected error in load_data callback" in r.getMessage()
        for r in caplog.records
    )


def test_tab_load_service_validate_base_url_apierror(logger, deps):
    service = TabLoadService(logger, deps)
    deps.validate_base_url.side_effect = APIError("msg", code="E001")
    result, msg = service.load_data("charts", 1, "base_url", "r1", "c1", "res1")
    assert msg == "msg [E001]"
    deps.validate_base_url.side_effect = APIError("msg2")
    result, msg = service.load_data("charts", 1, "base_url", "r1", "c1", "res1")
    assert msg == "msg2"


def test_tab_load_service_unknown_tab_fallback(logger, deps):
    service = TabLoadService(logger, deps)
    deps.validate_base_url.return_value = "safe_url"
    deps.render_judge_tab.return_value = ("judge", "")
    # 未知タブはjudge_tabにフォールバック
    result, msg = service.load_data("unknown", 1, "base_url", "r1", "c1", "res1")
    assert result == ("judge", "")
    assert msg == ""


def test_chart_name_option_service_n_clicks_zero(logger, deps):
    service = ChartNameOptionService(logger, deps)
    options, selected = service.refresh_chart_name_options(0, "base_url", "r1", "c1")
    assert options == []
    assert selected is None


def test_chart_name_option_service_validate_base_url_apierror(logger, deps, caplog):
    service = ChartNameOptionService(logger, deps)
    deps.validate_base_url.side_effect = APIError("msg", code="E001")
    with caplog.at_level(logging.ERROR, logger="test"):
        options, selected = service.refresh_chart_name_options(1, "base_url", "r1", "c1")
    assert options == []
    assert selected is None


def test_chart_name_option_service_recipe_id_omitted(logger, deps):
    service = ChartNameOptionService(logger, deps)
    deps.validate_base_url.return_value = "safe_url"
    deps.get_charts.return_value = [{"chart_id": "c1", "chart_name": "name1"}]
    service.refresh_chart_name_options(1, "base_url", "", "c1")
    # recipe_idが空の場合、paramsに含まれない
    called_args = deps.get_charts.call_args[1]["params"]
    assert "recipe_id" not in called_args


def test_chart_name_option_service_selected_none_when_not_found(logger, deps):
    def test_chart_name_option_service_validate_base_url_tuple(logger, deps):
        service = ChartNameOptionService(logger, deps)
        # validate_base_urlがtupleを返す場合の分岐網羅
        deps.validate_base_url.return_value = ("safe_url", "dummy")
        deps.get_charts.return_value = [{"chart_id": "c1", "chart_name": "name1"}]
        options, selected = service.refresh_chart_name_options(1, "base_url", "r1", "c1")
        assert isinstance(options, list)
        assert len(options) == 1
        assert options[0]["value"] == "c1"
        assert "name1" in options[0]["label"]
        assert selected == "c1"

    def test_tab_load_service_validate_base_url_tuple(logger, deps):
        service = TabLoadService(logger, deps)
        deps.validate_base_url.return_value = ("safe_url", "dummy")
        deps.render_charts_tab.return_value = ("charts", "")
        deps.render_active_tab.return_value = ("active", "")
        deps.render_history_tab.return_value = ("history", "")
        deps.render_judge_tab.return_value = ("judge", "")
        # charts
        result, msg = service.load_data("charts", 1, "base_url", "r1", "c1", "res1")
        assert result == ("charts", "")
        assert msg == ""
        # active
        result, msg = service.load_data("active", 1, "base_url", "r1", "c1", "res1")
        assert result == ("active", "")
        assert msg == ""
        # history
        result, msg = service.load_data("history", 1, "base_url", "r1", "c1", "res1")
        assert result == ("history", "")
        assert msg == ""
        # judge
        result, msg = service.load_data("judge", 1, "base_url", "r1", "c1", "res1")
        assert result == ("judge", "")
        assert msg == ""

    service = ChartNameOptionService(logger, deps)
    deps.validate_base_url.return_value = "safe_url"
    deps.get_charts.return_value = [{"chart_id": "c1", "chart_name": "name1"}]
    options, selected = service.refresh_chart_name_options(1, "base_url", "r1", "not_found")
    assert options
    assert selected is None


def test_active_drilldown_service_validate_base_url_tuple(logger, deps):
    service = ActiveDrilldownService(logger, deps)
    # validate_base_urlがtupleを返す場合の分岐網羅
    deps.validate_base_url.return_value = ("safe_url", "dummy")
    deps.get_process_waveform_preview.return_value = {"points": [{"y": 1}, {"y": 2}, {"y": 3}]}
    click_data = {"points": [{"customdata": "pid"}]}
    result = service.render_active_drilldown(click_data, "base_url")
    deps.get_process_waveform_preview.assert_called_with("safe_url", "pid", params={"limit": 500})
    assert result["data"]


def test_chart_name_option_service_get_charts_apierror(logger, deps, caplog):
    service = ChartNameOptionService(logger, deps)
    deps.validate_base_url.return_value = "safe_url"
    deps.get_charts.side_effect = APIError("msg2")
    with caplog.at_level(logging.ERROR, logger="test"):
        options, selected = service.refresh_chart_name_options(1, "base_url", "r1", "c1")
    assert options == []
    assert selected is None


def test_navigation_service_select_chart_from_table_none_cases():
    service = NavigationService()
    from dash import no_update

    # active_cell=None
    assert service.select_chart_from_table(None, [{"chart_id": "c1"}]) is no_update
    # data=None
    assert service.select_chart_from_table({"row": 0}, None) is no_update


def test_navigation_service_select_chart_from_table_row_idx_cases():
    service = NavigationService()
    from dash import no_update

    data = [{"chart_id": "c1"}, {"chart_id": "c2"}]
    # row_idx out of range
    assert service.select_chart_from_table({"row": 2}, data) is no_update
    # row_idx negative
    assert service.select_chart_from_table({"row": -1}, data) is no_update
    # row_idx not int
    assert service.select_chart_from_table({"row": "0"}, data) is no_update


def test_navigation_service_select_chart_from_table_chart_id_empty():
    service = NavigationService()
    from dash import no_update

    data = [{"chart_id": ""}]
    assert service.select_chart_from_table({"row": 0}, data) is no_update


def test_navigation_service_select_chart_from_table_normal():
    service = NavigationService()
    data = [{"chart_id": "c1"}, {"chart_id": "c2"}]
    assert service.select_chart_from_table({"row": 1}, data) == "c2"


def test_navigation_service_sync_active_selected_base_url():
    service = NavigationService()
    from src.portfolio_fdc.dashboard.base_url import DEFAULT_DB_API_BASE_URL

    # 空文字列→デフォルト
    assert service.sync_active_selected_base_url("") == DEFAULT_DB_API_BASE_URL
    # 非空はそのまま
    assert service.sync_active_selected_base_url("http://example.com") == "http://example.com"
