from __future__ import annotations

import logging
import os
from typing import Any

from dash import Dash, Input, Output, State, dcc, html

from .api_client import get_charts, get_process_waveform_preview
from .base_url import DEFAULT_DB_API_BASE_URL, validate_base_url
from .controller import DashboardController, DashboardDependencies
from .tab_renderers import (
    render_active_tab,
    render_charts_tab,
    render_history_tab,
    render_judge_tab,
)

logger = logging.getLogger(__name__)


def _build_controller() -> DashboardController:
    deps = DashboardDependencies(
        validate_base_url=validate_base_url,
        get_charts=get_charts,
        get_process_waveform_preview=get_process_waveform_preview,
        render_charts_tab=_render_charts_tab,
        render_active_tab=_render_active_tab,
        render_history_tab=_render_history_tab,
        render_judge_tab=_render_judge_tab,
    )
    return DashboardController(logger, deps)


app = Dash(
    __name__,
    suppress_callback_exceptions=False,
    title="FDC Dashboard Baseline",
    meta_tags=[{"name": "viewport", "content": "width=device-width, initial-scale=1"}],
    assets_folder=os.path.join(os.path.dirname(__file__), "assets"),
)

# Re-exported aliases for backward compatibility.
# IMPORTANT: Tests should monkeypatch tab_renderers module directly, not these aliases,
# to avoid silently breaking when tab_renderers is refactored.
# See: https://github.com/mdht-daiki/fdc-logger-toolkit/issues/163
_render_charts_tab = render_charts_tab
_render_active_tab = render_active_tab
_render_history_tab = render_history_tab
_render_judge_tab = render_judge_tab

app.layout = html.Div(
    [
        dcc.Location(id="url", refresh=False),
        dcc.Store(id="active-selected-base-url", data=DEFAULT_DB_API_BASE_URL),
        html.H2("FDC Dashboard Read-only Baseline"),
        html.Div(
            "Issue #146 scope: /charts /charts/active /charts/history "
            "/judge/results /judge/results/{id}"
        ),
        html.Div(
            [
                html.Div(
                    [
                        html.Label("db_api base URL"),
                        dcc.Input(
                            id="base-url",
                            type="text",
                            value=DEFAULT_DB_API_BASE_URL,
                            style={"width": "100%"},
                        ),
                    ],
                    style={
                        "display": "flex",
                        "flexDirection": "column",
                        "gap": "4px",
                        "flex": "2 1 320px",
                        "minWidth": "0",
                    },
                    className="dashboard-filter-group dashboard-filter-base-url",
                ),
                html.Div(
                    [
                        html.Label("recipe_id"),
                        dcc.Input(id="recipe-id", type="text", value="", style={"width": "100%"}),
                    ],
                    style={
                        "display": "flex",
                        "flexDirection": "column",
                        "gap": "4px",
                        "flex": "1 1 180px",
                        "minWidth": "0",
                    },
                    className="dashboard-filter-group",
                ),
                html.Div(
                    [
                        html.Label("chart_id"),
                        dcc.Input(id="chart-id", type="text", value="", style={"width": "100%"}),
                    ],
                    style={
                        "display": "flex",
                        "flexDirection": "column",
                        "gap": "4px",
                        "flex": "1 1 160px",
                        "minWidth": "0",
                    },
                    className="dashboard-filter-group",
                ),
                html.Div(
                    [
                        html.Label("chart_name"),
                        dcc.Dropdown(
                            id="chart-name",
                            options=[],
                            value=None,
                            placeholder="Select chart",
                            style={"width": "100%"},
                        ),
                    ],
                    style={
                        "display": "flex",
                        "flexDirection": "column",
                        "gap": "4px",
                        "flex": "3 1 360px",
                        "minWidth": "0",
                    },
                    className="dashboard-filter-group dashboard-filter-chart-name",
                ),
                html.Div(
                    [
                        html.Label("result_id"),
                        dcc.Input(id="result-id", type="text", value="", style={"width": "100%"}),
                    ],
                    style={
                        "display": "flex",
                        "flexDirection": "column",
                        "gap": "4px",
                        "flex": "1 1 160px",
                        "minWidth": "0",
                    },
                    className="dashboard-filter-group",
                ),
                html.Div(
                    [html.Button("Load", id="load-btn", n_clicks=0, style={"width": "100%"})],
                    style={
                        "display": "flex",
                        "alignItems": "flex-end",
                        "flex": "1 1 120px",
                        "minWidth": "96px",
                    },
                    className="dashboard-filter-group dashboard-filter-load",
                ),
            ],
            style={
                "display": "flex",
                "flexWrap": "wrap",
                "gap": "8px",
                "alignItems": "flex-end",
                "width": "100%",
                "margin": "12px 0",
            },
            className="dashboard-filter-controls",
        ),
        html.Div(
            id="error-banner",
            style={"color": "#b00020", "fontWeight": "bold", "marginBottom": "8px"},
        ),
        html.Div(
            dcc.Tabs(
                id="tabs",
                value="charts",
                children=[
                    dcc.Tab(label="Charts", value="charts"),
                    dcc.Tab(label="Active", value="active"),
                    dcc.Tab(label="History", value="history"),
                    dcc.Tab(label="Judge", value="judge"),
                ],
            ),
            className="dashboard-tabs-wrap",
        ),
        html.Div(id="tab-content"),
    ],
    style={"padding": "12px", "fontFamily": "Segoe UI, sans-serif"},
)


@app.callback(
    Output("tabs", "value"),
    Output("recipe-id", "value"),
    Output("chart-id", "value"),
    Output("result-id", "value"),
    Input("url", "search"),
    prevent_initial_call=False,
)
def sync_filters_from_url(search: str) -> tuple[str, str, str, str]:
    return _build_controller().sync_filters_from_url(search)


@app.callback(
    Output("tab-content", "children"),
    Output("error-banner", "children"),
    Input("tabs", "value"),
    Input("load-btn", "n_clicks"),
    State("base-url", "value"),
    State("recipe-id", "value"),
    State("chart-id", "value"),
    State("result-id", "value"),
    State("chart-name", "value"),
)
def load_data(
    active_tab: str,
    n_clicks: int,
    base_url: str,
    recipe_id: str,
    chart_id: str,
    result_id: str,
    selected_chart_id: str | None,
) -> tuple[Any, str]:
    return _build_controller().load_data(
        active_tab,
        n_clicks,
        base_url,
        recipe_id,
        chart_id,
        result_id,
        selected_chart_id,
    )


@app.callback(
    Output("chart-name", "options"),
    Output("chart-name", "value"),
    Input("load-btn", "n_clicks"),
    State("base-url", "value"),
    State("recipe-id", "value"),
    State("chart-id", "value"),
)
def refresh_chart_name_options(
    n_clicks: int,
    base_url: str,
    recipe_id: str,
    chart_id: str,
) -> tuple[list[dict[str, str]], str | None]:
    return _build_controller().refresh_chart_name_options(
        n_clicks,
        base_url,
        recipe_id,
        chart_id,
    )


@app.callback(
    Output("tabs", "value", allow_duplicate=True),
    Output("chart-id", "value", allow_duplicate=True),
    Output("url", "search", allow_duplicate=True),
    Input("chart-name", "value"),
    State("recipe-id", "value"),
    State("url", "search"),
    prevent_initial_call=True,
)
def move_to_active_by_chart_name(
    selected_chart_id: str | None,
    recipe_id: str,
    current_search: str,
) -> tuple[str, str, str] | tuple[Any, Any, Any]:
    return _build_controller().move_to_active_by_chart_name(
        selected_chart_id,
        recipe_id,
        current_search,
    )


@app.callback(
    Output("chart-name", "value", allow_duplicate=True),
    Input("charts-table", "cellClicked"),
    prevent_initial_call=True,
)
def select_chart_from_table(
    cell_clicked: dict[str, Any] | None,
) -> str | Any:
    return _build_controller().select_chart_from_table(cell_clicked)


@app.callback(
    Output("active-selected-base-url", "data"),
    Input("base-url", "value"),
)
def sync_active_selected_base_url(base_url: str) -> str:
    return _build_controller().sync_active_selected_base_url(base_url)


@app.callback(
    Output("active-drilldown-graph", "figure"),
    Input("active-main-graph", "clickData"),
    State("active-selected-base-url", "data"),
    prevent_initial_call=True,
)
def render_active_drilldown(
    click_data: dict[str, Any] | None,
    base_url: str,
) -> dict[str, Any]:
    return _build_controller().render_active_drilldown(click_data, base_url)


if __name__ == "__main__":
    host = os.getenv("HOST", "127.0.0.1")
    port = int(os.getenv("PORT", "8050"))
    debug = os.getenv("DEBUG", "false").strip().lower() in {"1", "true", "yes", "on"}
    app.run(host=host, port=port, debug=debug)
