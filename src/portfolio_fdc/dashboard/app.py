from __future__ import annotations

import logging
import os
from typing import Any

import dash_ag_grid as dag
from dash import Dash, Input, Output, State, dcc, html

from .api_client import (
    APIError,
    apply_change_request,
    approve_change_request,
    create_change_request,
    create_emergency_change,
    get_change_requests,
    get_charts,
    get_charts_history,
    get_failed_notifications,
    get_process_waveform_preview,
    parse_utc_millis,
    ratify_emergency_change,
    retry_notification,
)
from .base_url import DEFAULT_DB_API_BASE_URL, validate_base_url
from .controller import DashboardController, DashboardDependencies
from .tab_renderers import (
    render_active_tab,
    render_change_requests_tab,
    render_charts_tab,
    render_emergency_tab,
    render_history_tab,
    render_judge_tab,
    render_notification_retry_tab,
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
        render_change_requests_tab=_render_change_requests_tab,
        render_emergency_tab=_render_emergency_tab,
        render_notification_retry_tab=_render_notification_retry_tab,
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
_render_change_requests_tab = render_change_requests_tab
_render_emergency_tab = render_emergency_tab
_render_notification_retry_tab = render_notification_retry_tab


def _format_api_error(prefix: str, exc: APIError) -> str:
    suffix = ""
    if exc.code:
        suffix += f" [{exc.code}]"
    if exc.status_code is not None:
        suffix += f" (status={exc.status_code})"
    if exc.details is not None:
        if exc.status_code == 409 and isinstance(exc.details, dict):
            current = exc.details.get("current")
            if isinstance(current, dict):
                current_parts = [f"{key}={value}" for key, value in current.items()]
                suffix += f" current=({', '.join(current_parts)})"
            else:
                suffix += f" details={exc.details}"
        else:
            suffix += f" details={exc.details}"
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
                    dcc.Tab(label="Change Requests", value="change_requests"),
                    dcc.Tab(label="Emergency", value="emergency"),
                    dcc.Tab(label="Notification Retry", value="notification_retry"),
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
    except APIError as exc:
        return _format_api_error("Emergency apply failed", exc), [html.Div("履歴が未取得です")]
    except Exception:
        logger.exception("Unexpected error while validating base URL for emergency apply")
        return "Unexpected error while submitting emergency change", [html.Div("履歴が未取得です")]

    try:
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
    except APIError as exc:
        return _format_api_error("Emergency apply failed", exc), [html.Div("履歴が未取得です")]
    except Exception:
        logger.exception("Unexpected error while submitting emergency change")
        return "Unexpected error while submitting emergency change", [html.Div("履歴が未取得です")]

    history_rows: list[dict[str, Any]] | None = None
    history_failed = False
    try:
        history_rows = get_charts_history(
            safe_base_url,
            params={"limit": 20, "chart_id": str(parsed_chart_id)},
        )
    except APIError as exc:
        history_failed = True
        logger.exception("API error while loading emergency history preview: %s", exc)
    except Exception:
        history_failed = True
        logger.exception("Unexpected error while loading emergency history preview")

    result_text = (
        "Emergency apply success\n"
        f"request_id={data.get('request_id')}\n"
        f"status={data.get('status')}\n"
        f"resulting_version={data.get('resulting_version')}\n"
        f"noop={data.get('noop')}"
    )
    if history_failed:
        return result_text, [html.Div("履歴取得に失敗しました")]
    assert history_rows is not None
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


def _format_change_request_row(row: dict[str, Any]) -> str:
    return " | ".join(
        [
            f"request_id={row.get('id')}",
            f"status={row.get('status')}",
            f"chart_id={row.get('chart_id')}",
            f"proposed_by={row.get('proposed_by')}",
            f"proposed_at={row.get('proposed_at')}",
            f"expected_version={row.get('expected_version')}",
            f"idempotency_key={row.get('idempotency_key')}",
        ]
    )


def _build_change_request_list_view(
    rows: list[dict[str, Any]],
    detail_request_id: str | None,
) -> tuple[list[Any], list[Any]]:
    if not rows:
        return [html.Div("No change requests found")], [html.Div("No detail available")]

    table_rows: list[dict[str, Any]] = []
    detail_row: dict[str, Any] | None = None
    selected_id = (detail_request_id or "").strip()
    for row in rows:
        if selected_id and str(row.get("id")) == selected_id:
            detail_row = row
        table_rows.append(
            {
                "id": row.get("id"),
                "status": row.get("status"),
                "chart_id": row.get("chart_id"),
                "proposed_by": row.get("proposed_by"),
                "proposed_at": parse_utc_millis(
                    str(row.get("proposed_at")) if row.get("proposed_at") else None
                ),
                "expected_version": row.get("expected_version"),
                "idempotency_key": row.get("idempotency_key"),
            }
        )

    if detail_row is None:
        detail_row = rows[0]

    list_block = [
        html.H5(f"Change Requests: {len(table_rows)} rows"),
        html.Div(
            [
                html.Div(
                    "Filters: status / chart_id / from_ts / to_ts / limit / offset",
                    style={"marginBottom": "8px", "color": "#444"},
                ),
                html.Div(
                    [
                        html.Label("detail request_id"),
                        dcc.Input(
                            id="change-request-detail-request-id",
                            type="text",
                            value=selected_id,
                            placeholder="optional request_id",
                            style={"width": "100%"},
                        ),
                    ],
                    style={"marginBottom": "8px"},
                ),
                dag.AgGrid(
                    id="change-requests-table",
                    rowData=table_rows,
                    columnDefs=[
                        {"headerName": "id", "field": "id"},
                        {"headerName": "status", "field": "status"},
                        {"headerName": "chart_id", "field": "chart_id"},
                        {"headerName": "proposed_by", "field": "proposed_by"},
                        {"headerName": "proposed_at", "field": "proposed_at"},
                        {"headerName": "expected_version", "field": "expected_version"},
                        {"headerName": "idempotency_key", "field": "idempotency_key"},
                    ],
                    defaultColDef={"resizable": True, "sortable": True, "filter": True},
                    dashGridOptions={"pagination": True, "paginationPageSize": 10},
                    style={"width": "100%", "overflowX": "auto"},
                ),
            ]
        ),
    ]
    detail_block = [
        html.H5("Selected Request Detail"),
        html.Pre(
            _format_change_request_row(detail_row),
            style={"backgroundColor": "#f5f5f5", "padding": "8px"},
        ),
    ]
    return list_block, detail_block


@app.callback(
    Output("change-request-create-result", "children"),
    Input("change-request-create-btn", "n_clicks"),
    State("base-url", "value"),
    State("change-request-chart-id", "value"),
    State("change-request-proposed-by", "value"),
    State("change-request-change-payload", "value"),
    State("change-request-expected-version", "value"),
    State("change-request-idempotency-key", "value"),
    prevent_initial_call=True,
)
def submit_change_request_create(
    n_clicks: int,
    base_url: str,
    chart_id: str,
    proposed_by: str,
    change_payload: str,
    expected_version: str,
    idempotency_key: str,
) -> str:
    if not n_clicks:
        return ""

    parsed_chart_id, chart_err = _to_positive_int(chart_id, "chart_id")
    if chart_err is not None:
        return chart_err
    parsed_expected_version, version_err = _to_positive_int(expected_version, "expected_version")
    if version_err is not None:
        return version_err

    actor = (proposed_by or "").strip()
    payload_text = (change_payload or "").strip()
    key = (idempotency_key or "").strip()
    if not actor:
        return "proposed_by is required"
    if not payload_text:
        return "change_payload is required"
    if not key:
        return "idempotency_key is required"

    try:
        safe_base_url = validate_base_url(base_url)[0]
        data = create_change_request(
            safe_base_url,
            {
                "chart_id": parsed_chart_id,
                "proposed_by": actor,
                "change_payload": payload_text,
                "expected_version": parsed_expected_version,
                "idempotency_key": key,
            },
        )
    except APIError as exc:
        return _format_api_error("Change request create failed", exc)
    except Exception:
        logger.exception("Unexpected error while creating change request")
        return "Unexpected error while creating change request"

    return (
        "Change request create success\n"
        f"request_id={data.get('request_id')}\n"
        f"status={data.get('status')}"
    )


@app.callback(
    Output("change-request-approve-result", "children"),
    Input("change-request-approve-btn", "n_clicks"),
    State("base-url", "value"),
    State("change-request-approve-request-id", "value"),
    State("change-request-approved-by", "value"),
    State("change-request-approved-by-role", "value"),
    State("change-request-approve-comment", "value"),
    prevent_initial_call=True,
)
def submit_change_request_approve(
    n_clicks: int,
    base_url: str,
    request_id: str,
    approved_by: str,
    approved_by_role: str,
    comment: str,
) -> str:
    if not n_clicks:
        return ""

    parsed_request_id, request_err = _to_positive_int(request_id, "request_id")
    if request_err is not None:
        return request_err
    assert parsed_request_id is not None
    actor = (approved_by or "").strip()
    role = (approved_by_role or "").strip()
    if not actor:
        return "approved_by is required"
    if not role:
        return "approved_by_role is required"

    try:
        safe_base_url = validate_base_url(base_url)[0]
        data = approve_change_request(
            safe_base_url,
            parsed_request_id,
            {
                "approved_by": actor,
                "approved_by_role": role,
                "comment": (comment or "").strip() or None,
            },
        )
    except APIError as exc:
        return _format_api_error("Change request approve failed", exc)
    except Exception:
        logger.exception("Unexpected error while approving change request")
        return "Unexpected error while approving change request"

    return (
        "Change request approve success\n"
        f"request_id={data.get('request_id')}\n"
        f"status={data.get('status')}"
    )


@app.callback(
    Output("change-request-apply-result", "children"),
    Input("change-request-apply-btn", "n_clicks"),
    State("base-url", "value"),
    State("change-request-apply-request-id", "value"),
    State("change-request-applied-by", "value"),
    State("change-request-applied-by-role", "value"),
    State("change-request-apply-reason", "value"),
    prevent_initial_call=True,
)
def submit_change_request_apply(
    n_clicks: int,
    base_url: str,
    request_id: str,
    applied_by: str,
    applied_by_role: str,
    reason: str,
) -> str:
    if not n_clicks:
        return ""

    parsed_request_id, request_err = _to_positive_int(request_id, "request_id")
    if request_err is not None:
        return request_err
    assert parsed_request_id is not None
    actor = (applied_by or "").strip()
    role = (applied_by_role or "").strip()
    if not actor:
        return "applied_by is required"
    if not role:
        return "applied_by_role is required"

    try:
        safe_base_url = validate_base_url(base_url)[0]
        data = apply_change_request(
            safe_base_url,
            parsed_request_id,
            {
                "applied_by": actor,
                "applied_by_role": role,
                "reason": (reason or "").strip() or None,
            },
        )
    except APIError as exc:
        return _format_api_error("Change request apply failed", exc)
    except Exception:
        logger.exception("Unexpected error while applying change request")
        return "Unexpected error while applying change request"

    return (
        "Change request apply success\n"
        f"request_id={data.get('request_id')}\n"
        f"status={data.get('status')}\n"
        f"resulting_version={data.get('resulting_version')}\n"
        f"noop={data.get('noop')}"
    )


@app.callback(
    Output("change-request-query-result", "children"),
    Output("change-request-list", "children"),
    Output("change-request-detail", "children"),
    Input("change-request-refresh-btn", "n_clicks"),
    State("base-url", "value"),
    State("change-request-status", "value"),
    State("change-request-filter-chart-id", "value"),
    State("change-request-from-ts", "value"),
    State("change-request-to-ts", "value"),
    State("change-request-limit", "value"),
    State("change-request-offset", "value"),
    State("change-request-detail-request-id", "value"),
    prevent_initial_call=True,
)
def refresh_change_request_listing(
    n_clicks: int,
    base_url: str,
    status: str,
    chart_id: str,
    from_ts: str,
    to_ts: str,
    limit: str,
    offset: str,
    detail_request_id: str,
) -> tuple[str, list[Any], list[Any]]:
    if not n_clicks:
        return (
            "",
            [html.Div("Press Refresh to fetch change requests")],
            [html.Div("No detail available")],
        )

    try:
        safe_base_url = validate_base_url(base_url)[0]
        params: dict[str, Any] = {}
        if status:
            params["status"] = status
        parsed_chart_id, chart_err = _to_positive_int(chart_id, "chart_id")
        if chart_err is None and chart_id:
            params["chart_id"] = parsed_chart_id
        elif chart_id:
            return (
                chart_err or "chart_id is invalid",
                [html.Div("No rows loaded")],
                [html.Div("No detail available")],
            )

        if (from_ts or "").strip():
            params["from_ts"] = from_ts.strip()
        if (to_ts or "").strip():
            params["to_ts"] = to_ts.strip()

        parsed_limit, limit_err = _to_positive_int(limit, "limit")
        if limit_err is not None:
            return limit_err, [html.Div("No rows loaded")], [html.Div("No detail available")]
        params["limit"] = parsed_limit

        offset_text = (offset or "").strip()
        if offset_text:
            try:
                parsed_offset = int(offset_text)
            except ValueError:
                return (
                    "offset must be integer",
                    [html.Div("No rows loaded")],
                    [html.Div("No detail available")],
                )
            if parsed_offset < 0:
                return (
                    "offset must be >= 0",
                    [html.Div("No rows loaded")],
                    [html.Div("No detail available")],
                )
            params["offset"] = parsed_offset
        else:
            params["offset"] = 0

        rows = get_change_requests(safe_base_url, params=params)
        list_block, detail_block = _build_change_request_list_view(rows, detail_request_id)
        return f"Loaded {len(rows)} change request(s)", list_block, detail_block
    except APIError as exc:
        return (
            _format_api_error("Change request list failed", exc),
            [html.Div("No rows loaded")],
            [html.Div("No detail available")],
        )
    except Exception:
        logger.exception("Unexpected error while refreshing change request listing")
        return (
            "Unexpected error while loading change request listing",
            [html.Div("No rows loaded")],
            [html.Div("No detail available")],
        )


@app.callback(
    Output("notification-query-result", "children"),
    Output("notification-failed-table", "rowData"),
    Input("notification-refresh-btn", "n_clicks"),
    State("base-url", "value"),
    State("notification-filter-event-id", "value"),
    State("notification-filter-limit", "value"),
    State("notification-filter-offset", "value"),
    prevent_initial_call=True,
)
def refresh_failed_notifications(
    n_clicks: int,
    base_url: str,
    event_id: str,
    limit: str,
    offset: str,
) -> tuple[str, list[dict[str, Any]]]:
    if not n_clicks:
        return "", []

    try:
        safe_base_url = validate_base_url(base_url)[0]
        params: dict[str, Any] = {}

        parsed_event_id, event_err = _to_positive_int(event_id, "event_id")
        if event_id and event_err is not None:
            return event_err, []
        if event_id and parsed_event_id is not None:
            params["event_id"] = parsed_event_id

        parsed_limit, limit_err = _to_positive_int(limit, "limit")
        if limit_err is not None:
            return limit_err, []
        params["limit"] = parsed_limit

        offset_text = (offset or "").strip()
        if offset_text:
            try:
                parsed_offset = int(offset_text)
            except ValueError:
                return "offset must be integer", []
            if parsed_offset < 0:
                return "offset must be >= 0", []
            params["offset"] = parsed_offset
        else:
            params["offset"] = 0

        rows = get_failed_notifications(safe_base_url, params=params)
        table_rows = [
            {
                "event_id": row.get("event_id"),
                "status": row.get("status"),
                "retry_count": row.get("retry_count"),
                "next_retry_at": parse_utc_millis(
                    str(row.get("next_retry_at")) if row.get("next_retry_at") else None
                ),
                "last_attempt_at": parse_utc_millis(
                    str(row.get("last_attempt_at")) if row.get("last_attempt_at") else None
                ),
                "last_error": row.get("last_error"),
            }
            for row in rows
        ]
        return f"Loaded {len(table_rows)} failed notification(s)", table_rows
    except APIError as exc:
        return _format_api_error("Failed notification list failed", exc), []
    except Exception:
        logger.exception("Unexpected error while refreshing failed notifications")
        return "Unexpected error while loading failed notifications", []


@app.callback(
    Output("notification-retry-result", "children"),
    Input("notification-retry-btn", "n_clicks"),
    State("base-url", "value"),
    State("notification-retry-event-id", "value"),
    prevent_initial_call=True,
)
def submit_notification_retry(
    n_clicks: int,
    base_url: str,
    event_id: str,
) -> str:
    if not n_clicks:
        return ""

    parsed_event_id, event_err = _to_positive_int(event_id, "event_id")
    if event_err is not None:
        return event_err
    assert parsed_event_id is not None

    try:
        safe_base_url = validate_base_url(base_url)[0]
        data = retry_notification(safe_base_url, parsed_event_id)
    except APIError as exc:
        return _format_api_error("Notification retry failed", exc)
    except Exception:
        logger.exception("Unexpected error while retrying notification")
        return "Unexpected error while retrying notification"

    return (
        "Notification retry success\n"
        f"event_id={data.get('event_id')}\n"
        f"status={data.get('status')}\n"
        f"retry_count={data.get('retry_count')}\n"
        f"next_retry_at={data.get('next_retry_at')}"
    )


if __name__ == "__main__":
    host = os.getenv("HOST", "127.0.0.1")
    port = int(os.getenv("PORT", "8050"))
    debug = os.getenv("DEBUG", "false").strip().lower() in {"1", "true", "yes", "on"}
    app.run(host=host, port=port, debug=debug)
