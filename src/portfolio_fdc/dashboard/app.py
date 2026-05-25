from __future__ import annotations

import logging
import os
from typing import Any

from dash import Dash, Input, Output, State, dcc, html

from .api_client import (
    APIError,
    create_emergency_change,
    get_charts,
    get_charts_history,
    get_process_waveform_preview,
    parse_utc_millis,
    ratify_emergency_change,
)
from .base_url import DEFAULT_DB_API_BASE_URL, validate_base_url
from .controller import DashboardController, DashboardDependencies
from .tab_renderers import (
    render_active_tab,
    render_charts_tab,
    render_emergency_tab,
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
        render_emergency_tab=_render_emergency_tab,
    )
    return DashboardController(logger, deps)


def _build_filter_group(
    children: list[Any],
    *,
    flex: str,
    class_name: str = "dashboard-filter-group",
    style_overrides: dict[str, str] | None = None,
) -> html.Div:
    style = {
        "display": "flex",
        "flexDirection": "column",
        "gap": "4px",
        "flex": flex,
        "minWidth": "0",
    }
    if style_overrides:
        style.update(style_overrides)
    return html.Div(children, style=style, className=class_name)


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
_render_emergency_tab = render_emergency_tab


def _format_api_error(prefix: str, exc: APIError) -> str:
    suffix = ""
    if exc.code:
        suffix += f" [{exc.code}]"
    if exc.status_code is not None:
        suffix += f" (status={exc.status_code})"
    return f"{prefix}: {exc.message}{suffix}"


def _to_positive_int(raw: str, field_name: str) -> tuple[int | None, str | None]:
    value = (raw or "").strip()
    if not value:
        return None, f"{field_name} is required"
    try:
        parsed = int(value)
    except ValueError:
        return None, f"{field_name} must be integer"
    if parsed <= 0:
        return None, f"{field_name} must be >= 1"
    return parsed, None


def _build_history_preview(rows: list[dict[str, Any]]) -> list[Any]:
    if not rows:
        return [html.Div("No history rows found for target chart")]

    lines = []
    for row in rows[:5]:
        changed_at = parse_utc_millis(str(row.get("changed_at")) if row.get("changed_at") else None)
        lines.append(
            " | ".join(
                [
                    f"history_id={row.get('history_id')}",
                    f"source={row.get('change_source')}",
                    f"changed_by={row.get('changed_by')}",
                    f"changed_at={changed_at}",
                ]
            )
        )
    return [html.Pre("\n".join(lines), style={"backgroundColor": "#f5f5f5", "padding": "8px"})]


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
                _build_filter_group(
                    [
                        html.Label("db_api base URL"),
                        dcc.Input(
                            id="base-url",
                            type="text",
                            value=DEFAULT_DB_API_BASE_URL,
                            style={"width": "100%"},
                        ),
                    ],
                    flex="2 1 320px",
                    class_name="dashboard-filter-group dashboard-filter-base-url",
                ),
                _build_filter_group(
                    [
                        html.Label("recipe_id"),
                        dcc.Input(id="recipe-id", type="text", value="", style={"width": "100%"}),
                    ],
                    flex="1 1 180px",
                ),
                _build_filter_group(
                    [
                        html.Label("chart_id"),
                        dcc.Input(id="chart-id", type="text", value="", style={"width": "100%"}),
                    ],
                    flex="1 1 160px",
                ),
                _build_filter_group(
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
                    flex="3 1 360px",
                    class_name="dashboard-filter-group dashboard-filter-chart-name",
                ),
                _build_filter_group(
                    [
                        html.Label("result_id"),
                        dcc.Input(id="result-id", type="text", value="", style={"width": "100%"}),
                    ],
                    flex="1 1 160px",
                ),
                _build_filter_group(
                    [html.Button("Load", id="load-btn", n_clicks=0, style={"width": "100%"})],
                    flex="1 1 120px",
                    class_name="dashboard-filter-group dashboard-filter-load",
                    style_overrides={"alignItems": "flex-end", "minWidth": "96px"},
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
                    dcc.Tab(label="Emergency", value="emergency"),
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


@app.callback(
    Output("emergency-action-result", "children"),
    Output("emergency-history-preview", "children"),
    Input("emergency-submit-btn", "n_clicks"),
    State("base-url", "value"),
    State("emergency-chart-id", "value"),
    State("emergency-changed-by", "value"),
    State("emergency-changed-by-role", "value"),
    State("emergency-reason", "value"),
    State("emergency-change-payload", "value"),
    prevent_initial_call=True,
)
def submit_emergency_change(
    n_clicks: int,
    base_url: str,
    chart_id: str,
    changed_by: str,
    changed_by_role: str,
    reason: str,
    change_payload: str,
) -> tuple[str, list[Any]]:
    if not n_clicks:
        return "", [html.Div("Apply 実行後に履歴を表示します。")]

    parsed_chart_id, chart_err = _to_positive_int(chart_id, "chart_id")
    if chart_err is not None:
        return chart_err, [html.Div("Apply 実行後に履歴を表示します。")]

    actor = (changed_by or "").strip()
    role = (changed_by_role or "").strip()
    payload_text = (change_payload or "").strip()
    if not actor:
        return "changed_by is required", [html.Div("Apply 実行後に履歴を表示します。")]
    if not role:
        return "changed_by_role is required", [html.Div("Apply 実行後に履歴を表示します。")]
    if not payload_text:
        return "change_payload is required", [html.Div("Apply 実行後に履歴を表示します。")]

    try:
        safe_base_url = validate_base_url(base_url)[0]
        data = create_emergency_change(
            safe_base_url,
            {
                "chart_id": parsed_chart_id,
                "changed_by": actor,
                "changed_by_role": role,
                "reason": (reason or "").strip() or None,
                "change_payload": payload_text,
            },
        )
        history_rows = get_charts_history(
            safe_base_url,
            params={"limit": 20, "chart_id": str(parsed_chart_id)},
        )
    except APIError as exc:
        return _format_api_error("Emergency apply failed", exc), [
            html.Div("履歴取得に失敗しました")
        ]
    except Exception:
        logger.exception("Unexpected error while submitting emergency change")
        return "Unexpected error while submitting emergency change", [
            html.Div("履歴取得に失敗しました")
        ]

    result_text = (
        "Emergency apply success\n"
        f"request_id={data.get('request_id')}\n"
        f"status={data.get('status')}\n"
        f"resulting_version={data.get('resulting_version')}\n"
        f"noop={data.get('noop')}"
    )
    return result_text, _build_history_preview(history_rows)


@app.callback(
    Output("ratify-action-result", "children"),
    Input("ratify-submit-btn", "n_clicks"),
    State("base-url", "value"),
    State("ratify-request-id", "value"),
    State("ratify-by", "value"),
    State("ratify-role", "value"),
    State("ratify-comment", "value"),
    State("ratify-related-pr", "value"),
    prevent_initial_call=True,
)
def submit_emergency_ratify(
    n_clicks: int,
    base_url: str,
    request_id: str,
    ratified_by: str,
    ratified_by_role: str,
    ratification_comment: str,
    related_pr: str,
) -> str:
    if not n_clicks:
        return ""

    parsed_request_id, request_err = _to_positive_int(request_id, "request_id")
    if request_err is not None:
        return request_err
    assert parsed_request_id is not None

    actor = (ratified_by or "").strip()
    role = (ratified_by_role or "").strip()
    if not actor:
        return "ratified_by is required"
    if not role:
        return "ratified_by_role is required"

    try:
        safe_base_url = validate_base_url(base_url)[0]
        data = ratify_emergency_change(
            safe_base_url,
            parsed_request_id,
            {
                "ratified_by": actor,
                "ratified_by_role": role,
                "ratification_comment": (ratification_comment or "").strip() or None,
                "related_pr": (related_pr or "").strip() or None,
            },
        )
    except APIError as exc:
        return _format_api_error("Emergency ratify failed", exc)
    except Exception:
        logger.exception("Unexpected error while ratifying emergency change")
        return "Unexpected error while ratifying emergency change"

    return (
        "Emergency ratify success\n"
        f"request_id={data.get('request_id')}\n"
        f"status={data.get('status')}"
    )


if __name__ == "__main__":
    host = os.getenv("HOST", "127.0.0.1")
    port = int(os.getenv("PORT", "8050"))
    debug = os.getenv("DEBUG", "false").strip().lower() in {"1", "true", "yes", "on"}
    app.run(host=host, port=port, debug=debug)
