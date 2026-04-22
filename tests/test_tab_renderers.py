from unittest.mock import MagicMock, patch

from dash import html

from src.portfolio_fdc.dashboard import tab_renderers


# --- render_charts_tab ---
def test_render_charts_tab_minimal():
    with patch("src.portfolio_fdc.dashboard.tab_renderers.get_charts") as mock_get_charts:
        mock_get_charts.return_value = [
            {
                "chart_id": "C1",
                "is_active": True,
                "recipe_id": "R1",
                "parameter": "P",
                "step_no": 1,
                "feature_type": "F",
                "warning_lcl": 1,
                "warning_ucl": 2,
                "critical_lcl": 3,
                "critical_ucl": 4,
                "updated_at": 1710000000000,
            }
        ]
        div = tab_renderers.render_charts_tab("base", "R1")
        assert isinstance(div, html.Div)
        # get_chartsの呼び出し引数を検証
        mock_get_charts.assert_called_once_with("base", params={"recipe_id": "R1"})
        # ヘッダ文言の厳密一致
        assert div.children[0].children == "Charts: 1 rows"


# --- render_active_tab ---
def test_render_active_tab_minimal():
    with (
        patch(
            "src.portfolio_fdc.dashboard.tab_renderers.get_active_charts"
        ) as mock_get_active_charts,
        patch(
            "src.portfolio_fdc.dashboard.tab_renderers.get_chart_points"
        ) as mock_get_chart_points,
        patch("src.portfolio_fdc.dashboard.tab_renderers.spc_band_with_points_figure") as mock_fig,
        patch("src.portfolio_fdc.dashboard.tab_renderers.empty_drilldown_figure") as mock_empty,
        patch(
            "src.portfolio_fdc.dashboard.tab_renderers.parse_utc_millis"
        ) as mock_parse_utc_millis,
    ):
        mock_get_active_charts.return_value = {
            "charts": [
                {
                    "chart_id": "C1",
                    "parameter": "P",
                    "step_no": 1,
                    "feature_type": "F",
                    "warning_lcl": 1,
                    "warning_ucl": 2,
                    "critical_lcl": 3,
                    "critical_ucl": 4,
                }
            ],
            "activated_at": 1710000000000,
        }
        mock_get_chart_points.return_value = [{"x": 1, "y": 2}]
        mock_fig.return_value = MagicMock()
        mock_empty.return_value = MagicMock()
        mock_parse_utc_millis.return_value = "2024-03-10 09:20:00"
        div = tab_renderers.render_active_tab("base", "R1", "C1")
        assert isinstance(div, html.Div)
        mock_get_active_charts.assert_called_once_with("base", params={"recipe_id": "R1"})
        mock_get_chart_points.assert_called_once_with("base", chart_id="C1", params={"limit": 30})
        assert div.children[0].children == "Active chart_set=- / activated_at=2024-03-10 09:20:00"


# --- render_history_tab ---
def test_render_history_tab_minimal():
    with patch(
        "src.portfolio_fdc.dashboard.tab_renderers.get_charts_history"
    ) as mock_get_charts_history:
        mock_get_charts_history.return_value = [
            {
                "history_id": "H1",
                "chart_id": "C1",
                "change_source": "src",
                "change_reason": "reason",
                "changed_by": "user",
                "changed_at": 1710000000000,
            }
        ]
        div = tab_renderers.render_history_tab("base", "C1")
        assert isinstance(div, html.Div)
        mock_get_charts_history.assert_called_once_with(
            "base", params={"limit": 100, "chart_id": "C1"}
        )
        assert div.children[0].children == "History: 1 rows"


# --- render_judge_tab ---
def test_render_judge_tab_minimal():
    with (
        patch(
            "src.portfolio_fdc.dashboard.tab_renderers._build_judge_table_rows"
        ) as mock_table_rows,
        patch(
            "src.portfolio_fdc.dashboard.tab_renderers._build_judge_drilldown_links"
        ) as mock_links,
        patch("src.portfolio_fdc.dashboard.tab_renderers._build_judge_detail_block") as mock_detail,
        patch(
            "src.portfolio_fdc.dashboard.tab_renderers.get_judge_result"
        ) as mock_get_judge_result,
    ):
        mock_table_rows.return_value = (
            [{"result_id": "R1", "level": "OK"}],
            [{"result_id": "R1", "level": "OK"}],
        )
        mock_links.return_value = [html.Li("link1")]
        mock_detail.return_value = html.Div("detail")
        mock_get_judge_result.return_value = {"result_id": "RID1", "level": "OK"}
        div = tab_renderers.render_judge_tab("base", "R1", "C1", "RID1")
        assert isinstance(div, html.Div)
        mock_table_rows.assert_called_once_with(
            "base", {"limit": 200, "recipe_id": "R1", "chart_id": "C1"}
        )
        mock_links.assert_called_once()
        mock_detail.assert_called_once()
        mock_get_judge_result.assert_called_once_with("base", result_id="RID1")
        assert div.children[0].children == "Judge Results (priority: NG > WARN > OK)"
