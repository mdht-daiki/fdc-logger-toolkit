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
        assert "Charts:" in div.children[0].children


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
        div = tab_renderers.render_active_tab("base", "R1", "C1")
        assert isinstance(div, html.Div)
        assert "Active chart_set=" in div.children[0].children


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
        assert "History:" in div.children[0].children


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
        assert "Judge Results" in div.children[0].children
