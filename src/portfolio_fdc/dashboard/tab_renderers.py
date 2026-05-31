from __future__ import annotations

import json
from typing import Any
from urllib.parse import urlencode

import dash_ag_grid as dag
from dash import dcc, html

from .api_client import (
    get_active_charts,
    get_change_requests,
    get_chart_points,
    get_charts,
    get_charts_history,
    get_failed_notifications,
    get_judge_result,
    get_judge_results,
    parse_utc_millis,
)
from .view_models import (
    LEVEL_COLOR,
    build_chart_name,
    empty_drilldown_figure,
    format_range,
    sort_judge_rows,
    spc_band_with_points_figure,
)


def _ag_grid_columns(keys: list[str]) -> list[dict[str, Any]]:
    return [{"headerName": key, "field": key} for key in keys]


def render_charts_tab(base_url: str, recipe_id: str) -> html.Div:
    params: dict[str, Any] = {}
    if recipe_id:
        params["recipe_id"] = recipe_id

    rows = get_charts(base_url, params=params)
    table_rows = [
        {
            "chart_id": row.get("chart_id"),
            "is_active": row.get("is_active"),
            "chart_name": build_chart_name(row),
            "recipe_id": row.get("recipe_id"),
            "parameter": row.get("parameter"),
            "step_no": row.get("step_no"),
            "feature_type": row.get("feature_type"),
            "warning": format_range(row.get("warning_lcl"), row.get("warning_ucl")),
            "critical": format_range(row.get("critical_lcl"), row.get("critical_ucl")),
            "updated_at": parse_utc_millis(
                str(row.get("updated_at")) if row.get("updated_at") else None
            ),
            "open": (
                (
                    "[Open](?"
                    + urlencode(
                        {
                            "tab": "active",
                            "chart_id": str(row.get("chart_id")),
                            "recipe_id": recipe_id,
                        },
                        doseq=False,
                        safe="",
                    )
                    + ")"
                )
                if row.get("chart_id")
                else ""
            ),
        }
        for row in rows
    ]

    return html.Div(
        [
            html.H4(f"Charts: {len(table_rows)} rows"),
            dag.AgGrid(
                id="charts-table",
                rowData=table_rows,
                columnDefs=(
                    [
                        {
                            **col,
                            "cellRenderer": "markdown",
                            "cellRendererParams": {"linkTarget": "_self"},
                        }
                        if col["field"] == "open"
                        else col
                        for col in _ag_grid_columns(
                            [
                                "chart_id",
                                "is_active",
                                "chart_name",
                                "recipe_id",
                                "parameter",
                                "step_no",
                                "feature_type",
                                "warning",
                                "critical",
                                "updated_at",
                                "open",
                            ]
                        )
                    ]
                    if table_rows
                    else []
                ),
                defaultColDef={"resizable": True, "sortable": True, "filter": True},
                dashGridOptions={"pagination": True, "paginationPageSize": 12},
                style={"width": "100%", "overflowX": "auto"},
            ),
        ]
    )


def render_active_tab(base_url: str, recipe_id: str, chart_id: str) -> html.Div:
    params: dict[str, Any] = {}
    if recipe_id:
        params["recipe_id"] = recipe_id

    data = get_active_charts(base_url, params=params)
    charts = data.get("charts", []) if isinstance(data, dict) else []

    selected_chart = charts[0] if charts else {}
    selected_chart_id = str(selected_chart.get("chart_id")) if selected_chart else ""
    if chart_id:
        matched = [row for row in charts if str(row.get("chart_id")) == chart_id]
        if matched:
            selected_chart = matched[0]
            selected_chart_id = chart_id

    points: list[dict[str, Any]] = []
    if selected_chart_id:
        points = get_chart_points(
            base_url,
            chart_id=selected_chart_id,
            params={"limit": 30},
        )

    figure = spc_band_with_points_figure(selected_chart, points)
    activated_at = parse_utc_millis(
        str(data.get("activated_at")) if data.get("activated_at") else None
    )
    rows = [
        {
            "chart_id": row.get("chart_id"),
            "chart_name": build_chart_name(row),
            "parameter": row.get("parameter"),
            "step_no": row.get("step_no"),
            "feature_type": row.get("feature_type"),
            "warning": format_range(row.get("warning_lcl"), row.get("warning_ucl")),
            "critical": format_range(row.get("critical_lcl"), row.get("critical_ucl")),
        }
        for row in charts
    ]

    return html.Div(
        [
            html.H4(
                "Active chart_set="
                f"{data.get('active_chart_set_id', '-')} / activated_at={activated_at}"
            ),
            html.Div(
                f"Focused chart: {build_chart_name(selected_chart) if selected_chart else '-'}",
                style={"marginBottom": "8px", "fontWeight": "bold"},
            ),
            dcc.Graph(id="active-main-graph", figure=figure),
            dcc.Graph(
                id="active-drilldown-graph",
                figure=empty_drilldown_figure(
                    "Click a point in the top graph to show raw waveform"
                ),
            ),
            dag.AgGrid(
                rowData=rows,
                columnDefs=_ag_grid_columns(list(rows[0].keys())) if rows else [],
                defaultColDef={"resizable": True, "sortable": True, "filter": True},
                dashGridOptions={"pagination": True, "paginationPageSize": 10},
                style={"width": "100%", "overflowX": "auto"},
            ),
        ]
    )


def render_history_tab(base_url: str, chart_id: str) -> html.Div:
    params: dict[str, Any] = {"limit": 100}
    if chart_id:
        params["chart_id"] = chart_id

    rows = get_charts_history(base_url, params=params)
    table_rows = [
        {
            "history_id": row.get("history_id"),
            "chart_id": row.get("chart_id"),
            "change_source": row.get("change_source"),
            "change_reason": row.get("change_reason"),
            "changed_by": row.get("changed_by"),
            "changed_at": parse_utc_millis(
                str(row.get("changed_at")) if row.get("changed_at") else None
            ),
        }
        for row in rows
    ]

    return html.Div(
        [
            html.H4(f"History: {len(table_rows)} rows"),
            dag.AgGrid(
                rowData=table_rows,
                columnDefs=_ag_grid_columns(list(table_rows[0].keys())) if table_rows else [],
                defaultColDef={"resizable": True, "sortable": True, "filter": True},
                dashGridOptions={"pagination": True, "paginationPageSize": 12},
                style={"width": "100%", "overflowX": "auto"},
            ),
        ]
    )


def _build_change_request_detail(row: dict[str, Any]) -> html.Pre:
    detail_payload = {
        "id": row.get("id"),
        "chart_id": row.get("chart_id"),
        "status": row.get("status"),
        "proposed_by": row.get("proposed_by"),
        "proposed_at": parse_utc_millis(
            str(row.get("proposed_at")) if row.get("proposed_at") else None
        ),
        "change_payload": row.get("change_payload"),
        "expected_version": row.get("expected_version"),
        "idempotency_key": row.get("idempotency_key"),
    }
    return html.Pre(
        json.dumps(detail_payload, indent=2, ensure_ascii=False),
        style={"backgroundColor": "#f5f5f5", "padding": "8px"},
    )


def _build_change_request_sections(
    rows: list[dict[str, Any]],
    detail_request_id: str | None = None,
) -> tuple[list[Any], list[Any]]:
    if not rows:
        return [html.Div("No change requests found")], [html.Div("No detail available")]

    selected_id = (detail_request_id or "").strip()
    detail_row = rows[0]
    if selected_id:
        matched = [row for row in rows if str(row.get("id")) == selected_id]
        if matched:
            detail_row = matched[0]

    table_rows = [
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
        for row in rows
    ]

    list_block = [
        html.H5(f"Change Requests: {len(table_rows)} rows"),
        dag.AgGrid(
            id="change-requests-table",
            rowData=table_rows,
            columnDefs=(
                [
                    {"headerName": "id", "field": "id"},
                    {"headerName": "status", "field": "status"},
                    {"headerName": "chart_id", "field": "chart_id"},
                    {"headerName": "proposed_by", "field": "proposed_by"},
                    {"headerName": "proposed_at", "field": "proposed_at"},
                    {"headerName": "expected_version", "field": "expected_version"},
                    {"headerName": "idempotency_key", "field": "idempotency_key"},
                ]
                if table_rows
                else []
            ),
            defaultColDef={"resizable": True, "sortable": True, "filter": True},
            dashGridOptions={"pagination": True, "paginationPageSize": 10},
            style={"width": "100%", "overflowX": "auto"},
        ),
    ]
    detail_block = [html.H5("Selected Request Detail"), _build_change_request_detail(detail_row)]
    return list_block, detail_block


def render_change_requests_tab(base_url: str) -> html.Div:
    rows = get_change_requests(base_url, params={"limit": 100, "offset": 0})
    list_block, detail_block = _build_change_request_sections(rows)
    return html.Div(
        [
            html.H4("Change Requests"),
            html.Div(
                "Create / approve / apply の通常変更フローを dashboard から操作します。"
                "409/422/5xx の envelope をそのまま表示します。",
                style={"marginBottom": "8px", "color": "#444"},
            ),
            html.Div(
                [
                    html.H5("Create Change Request"),
                    html.Label("chart_id"),
                    dcc.Input(
                        id="change-request-chart-id", type="text", value="", style={"width": "100%"}
                    ),
                    html.Label("proposed_by"),
                    dcc.Input(
                        id="change-request-proposed-by",
                        type="text",
                        value="",
                        placeholder="enter actor",
                        style={"width": "100%"},
                    ),
                    html.Label("change_payload (JSON)"),
                    dcc.Textarea(
                        id="change-request-change-payload",
                        value='{"warn_low": 20.0, "warn_high": 30.0}',
                        style={"width": "100%", "height": "96px", "fontFamily": "Consolas"},
                    ),
                    html.Label("expected_version"),
                    dcc.Input(
                        id="change-request-expected-version",
                        type="text",
                        value="1",
                        style={"width": "100%"},
                    ),
                    html.Label("idempotency_key"),
                    dcc.Input(
                        id="change-request-idempotency-key",
                        type="text",
                        value="",
                        placeholder="unique key for retries",
                        style={"width": "100%"},
                    ),
                    html.Button(
                        "Create Change Request",
                        id="change-request-create-btn",
                        n_clicks=0,
                        style={"marginTop": "8px"},
                    ),
                    html.Pre(
                        id="change-request-create-result",
                        style={"backgroundColor": "#f5f5f5", "padding": "8px", "marginTop": "8px"},
                    ),
                ],
                style={"border": "1px solid #ddd", "padding": "10px", "marginBottom": "10px"},
            ),
            html.Div(
                [
                    html.H5("Change Request Filters / Detail"),
                    html.Label("status"),
                    dcc.Dropdown(
                        id="change-request-status",
                        options=[
                            {"label": "pending", "value": "pending"},
                            {"label": "approved", "value": "approved"},
                            {"label": "applied", "value": "applied"},
                            {"label": "apply_failed", "value": "apply_failed"},
                            {"label": "rejected", "value": "rejected"},
                        ],
                        value=None,
                        placeholder="all statuses",
                        clearable=True,
                        style={"width": "100%"},
                    ),
                    html.Label("chart_id"),
                    dcc.Input(
                        id="change-request-filter-chart-id",
                        type="text",
                        value="",
                        style={"width": "100%"},
                    ),
                    html.Label("from_ts (optional)"),
                    dcc.Input(
                        id="change-request-from-ts", type="text", value="", style={"width": "100%"}
                    ),
                    html.Label("to_ts (optional)"),
                    dcc.Input(
                        id="change-request-to-ts", type="text", value="", style={"width": "100%"}
                    ),
                    html.Label("limit"),
                    dcc.Input(
                        id="change-request-limit", type="text", value="100", style={"width": "100%"}
                    ),
                    html.Label("offset"),
                    dcc.Input(
                        id="change-request-offset", type="text", value="0", style={"width": "100%"}
                    ),
                    html.Button(
                        "Refresh Change Requests",
                        id="change-request-refresh-btn",
                        n_clicks=0,
                        style={"marginTop": "8px"},
                    ),
                    html.Pre(
                        id="change-request-query-result",
                        children="Press Refresh to fetch change requests.",
                        style={"backgroundColor": "#f5f5f5", "padding": "8px", "marginTop": "8px"},
                    ),
                    html.Div(id="change-request-list", children=list_block),
                    html.Div(id="change-request-detail", children=detail_block),
                ],
                style={"border": "1px solid #ddd", "padding": "10px", "marginBottom": "10px"},
            ),
            html.Div(
                [
                    html.H5("Approve Change Request"),
                    html.Label("request_id"),
                    dcc.Input(
                        id="change-request-approve-request-id",
                        type="text",
                        value="",
                        style={"width": "100%"},
                    ),
                    html.Label("approved_by"),
                    dcc.Input(
                        id="change-request-approved-by",
                        type="text",
                        value="",
                        style={"width": "100%"},
                    ),
                    html.Label("approved_by_role"),
                    dcc.Input(
                        id="change-request-approved-by-role",
                        type="text",
                        value="",
                        style={"width": "100%"},
                    ),
                    html.Label("comment (optional)"),
                    dcc.Input(
                        id="change-request-approve-comment",
                        type="text",
                        value="",
                        style={"width": "100%"},
                    ),
                    html.Button(
                        "Approve Change Request",
                        id="change-request-approve-btn",
                        n_clicks=0,
                        style={"marginTop": "8px"},
                    ),
                    html.Pre(
                        id="change-request-approve-result",
                        style={"backgroundColor": "#f5f5f5", "padding": "8px", "marginTop": "8px"},
                    ),
                ],
                style={"border": "1px solid #ddd", "padding": "10px", "marginBottom": "10px"},
            ),
            html.Div(
                [
                    html.H5("Apply Change Request"),
                    html.Label("request_id"),
                    dcc.Input(
                        id="change-request-apply-request-id",
                        type="text",
                        value="",
                        style={"width": "100%"},
                    ),
                    html.Label("applied_by"),
                    dcc.Input(
                        id="change-request-applied-by",
                        type="text",
                        value="",
                        style={"width": "100%"},
                    ),
                    html.Label("applied_by_role"),
                    dcc.Input(
                        id="change-request-applied-by-role",
                        type="text",
                        value="",
                        style={"width": "100%"},
                    ),
                    html.Label("reason (optional)"),
                    dcc.Input(
                        id="change-request-apply-reason",
                        type="text",
                        value="",
                        style={"width": "100%"},
                    ),
                    html.Button(
                        "Apply Change Request",
                        id="change-request-apply-btn",
                        n_clicks=0,
                        style={"marginTop": "8px"},
                    ),
                    html.Pre(
                        id="change-request-apply-result",
                        style={"backgroundColor": "#f5f5f5", "padding": "8px", "marginTop": "8px"},
                    ),
                ],
                style={"border": "1px solid #ddd", "padding": "10px", "marginBottom": "10px"},
            ),
            html.Div(
                f"target db_api: {base_url}",
                style={"marginTop": "10px", "color": "#666", "fontSize": "0.9rem"},
            ),
        ]
    )


def _build_judge_table_rows(
    base_url: str,
    params: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows = sort_judge_rows(get_judge_results(base_url, params=params))
    table_rows = [
        {
            "result_id": row.get("result_id"),
            "level": row.get("level"),
            "chart_id": row.get("chart_id"),
            "process_id": row.get("process_id"),
            "lot_id": row.get("lot_id"),
            "feature_type": row.get("feature_type"),
            "feature_value": row.get("feature_value"),
            "judged_at": parse_utc_millis(
                str(row.get("judged_at")) if row.get("judged_at") else None
            ),
            "process_start_ts": parse_utc_millis(
                str(row.get("process_start_ts")) if row.get("process_start_ts") else None
            ),
        }
        for row in rows
    ]
    return rows, table_rows


def _build_judge_drilldown_links(
    rows: list[dict[str, Any]],
    recipe_id: str,
    chart_id: str,
) -> list[html.Li]:
    drilldown_links: list[html.Li] = []
    for row in rows[:20]:
        rid = row.get("result_id")
        if not isinstance(rid, str):
            continue
        params_for_href: dict[str, str] = {
            "tab": "judge",
            "result_id": rid,
        }
        if recipe_id:
            params_for_href["recipe_id"] = recipe_id
        if chart_id:
            params_for_href["chart_id"] = chart_id
        href = f"?{urlencode(params_for_href, doseq=False, safe='')}"
        drilldown_links.append(html.Li(html.A(rid, href=href)))
    return drilldown_links


def _build_judge_detail_block(
    detail: dict[str, Any] | None,
    level_color_map: dict[str, str],
) -> Any:
    if detail is None:
        return html.Div("Select/enter result_id to load detail")

    level = str(detail.get("level", ""))
    judged_at_val = parse_utc_millis(
        str(detail.get("judged_at")) if detail.get("judged_at") else None
    )
    process_start_ts_val = parse_utc_millis(
        str(detail.get("process_start_ts")) if detail.get("process_start_ts") else None
    )

    return html.Pre(
        "\n".join(
            [
                f"result_id: {detail.get('result_id')}",
                f"level: {level}",
                f"chart_id: {detail.get('chart_id')}",
                f"process_id: {detail.get('process_id')}",
                f"feature: {detail.get('feature_type')}={detail.get('feature_value')}",
                (
                    "thresholds: "
                    f"warn[{detail.get('warning_lcl')}, {detail.get('warning_ucl')}], "
                    f"crit[{detail.get('critical_lcl')}, {detail.get('critical_ucl')}]"
                ),
                f"judged_at: {judged_at_val}",
                f"process_start_ts: {process_start_ts_val}",
            ]
        ),
        style={
            "padding": "10px",
            "backgroundColor": "#f5f5f5",
            "borderLeft": f"6px solid {level_color_map.get(level, '#555555')}",
        },
    )


def render_judge_tab(base_url: str, recipe_id: str, chart_id: str, result_id: str) -> html.Div:
    params: dict[str, Any] = {"limit": 200}
    if recipe_id:
        params["recipe_id"] = recipe_id
    if chart_id:
        params["chart_id"] = chart_id

    rows, table_rows = _build_judge_table_rows(base_url, params)
    drilldown_links = _build_judge_drilldown_links(rows, recipe_id, chart_id)

    detail: dict[str, Any] | None = None
    if result_id:
        detail = get_judge_result(base_url, result_id=result_id)

    detail_block = _build_judge_detail_block(detail, LEVEL_COLOR)

    return html.Div(
        [
            html.H4("Judge Results (priority: NG > WARN > OK)"),
            html.Div(
                [
                    html.Div("Drilldown Links (URL parameter navigation)"),
                    html.Ul(drilldown_links),
                ]
            ),
            dag.AgGrid(
                rowData=table_rows,
                columnDefs=_ag_grid_columns(list(table_rows[0].keys())) if table_rows else [],
                defaultColDef={"resizable": True, "sortable": True, "filter": True},
                dashGridOptions={"pagination": True, "paginationPageSize": 12},
                getRowStyle={
                    "styleConditions": [
                        {
                            "condition": "params.data.level === 'NG'",
                            "style": {"backgroundColor": "rgba(176,0,32,0.08)"},
                        },
                        {
                            "condition": "params.data.level === 'WARN'",
                            "style": {"backgroundColor": "rgba(245,124,0,0.08)"},
                        },
                        {
                            "condition": "params.data.level === 'OK'",
                            "style": {"backgroundColor": "rgba(46,125,50,0.08)"},
                        },
                    ]
                },
                style={"width": "100%", "overflowX": "auto"},
            ),
            html.H4("Judge Result Detail"),
            detail_block,
        ]
    )


def _build_emergency_change_form(chart_id: str) -> html.Div:
    return html.Div(
        [
            html.H5("Emergency Change"),
            html.Label("chart_id"),
            dcc.Input(
                id="emergency-chart-id",
                type="text",
                value=chart_id or "",
                style={"width": "100%"},
            ),
            html.Label("changed_by"),
            dcc.Input(
                id="emergency-changed-by",
                type="text",
                value="",
                placeholder="enter actor",
                style={"width": "100%"},
            ),
            html.Label("changed_by_role"),
            dcc.Input(
                id="emergency-changed-by-role",
                type="text",
                value="",
                placeholder="enter role",
                style={"width": "100%"},
            ),
            html.Label("reason (optional)"),
            dcc.Input(
                id="emergency-reason",
                type="text",
                value="",
                style={"width": "100%"},
            ),
            html.Label("change_payload (JSON)"),
            dcc.Textarea(
                id="emergency-change-payload",
                value='{"warn_high": 1.9, "crit_high": 2.0}',
                style={"width": "100%", "height": "96px", "fontFamily": "Consolas"},
            ),
            html.Button(
                "Apply Emergency Change",
                id="emergency-submit-btn",
                n_clicks=0,
                style={"marginTop": "8px"},
            ),
            html.Pre(
                id="emergency-action-result",
                style={"backgroundColor": "#f5f5f5", "padding": "8px", "marginTop": "8px"},
            ),
        ],
        style={"border": "1px solid #ddd", "padding": "10px", "marginBottom": "10px"},
    )


def _build_ratify_form() -> html.Div:
    return html.Div(
        [
            html.H5("Ratify Emergency Change"),
            html.Label("request_id"),
            dcc.Input(
                id="ratify-request-id",
                type="text",
                value="",
                style={"width": "100%"},
            ),
            html.Label("ratified_by"),
            dcc.Input(
                id="ratify-by",
                type="text",
                value="",
                placeholder="enter actor",
                style={"width": "100%"},
            ),
            html.Label("ratified_by_role"),
            dcc.Input(
                id="ratify-role",
                type="text",
                value="",
                placeholder="enter role",
                style={"width": "100%"},
            ),
            html.Label("ratification_comment (optional)"),
            dcc.Input(
                id="ratify-comment",
                type="text",
                value="",
                style={"width": "100%"},
            ),
            html.Label("related_pr (optional)"),
            dcc.Input(
                id="ratify-related-pr",
                type="text",
                value="",
                style={"width": "100%"},
            ),
            html.Button("Ratify", id="ratify-submit-btn", n_clicks=0, style={"marginTop": "8px"}),
            html.Pre(
                id="ratify-action-result",
                style={"backgroundColor": "#f5f5f5", "padding": "8px", "marginTop": "8px"},
            ),
        ],
        style={"border": "1px solid #ddd", "padding": "10px", "marginBottom": "10px"},
    )


def _build_history_block(base_url: str) -> list[html.Div]:
    return [
        html.Div(
            [
                html.H5("History Preview (/charts/history)"),
                html.Div(
                    id="emergency-history-preview", children="Apply 実行後に履歴を表示します。"
                ),
            ],
            style={"border": "1px solid #ddd", "padding": "10px"},
        ),
        html.Div(
            f"target db_api: {base_url}",
            style={"marginTop": "10px", "color": "#666", "fontSize": "0.9rem"},
        ),
    ]


def render_emergency_tab(base_url: str, chart_id: str) -> html.Div:
    return html.Div(
        [
            html.H4("Emergency Change / Ratify"),
            html.Div(
                "403/4xx/5xx の error envelope をそのまま表示します。"
                "実行後に履歴プレビューで反映を確認できます。",
                style={"marginBottom": "8px", "color": "#444"},
            ),
            _build_emergency_change_form(chart_id),
            _build_ratify_form(),
            *_build_history_block(base_url),
        ]
    )


def render_notification_retry_tab(base_url: str) -> html.Div:
    rows = get_failed_notifications(base_url, params={"limit": 100, "offset": 0})
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

    return html.Div(
        [
            html.H4("Notification Retry"),
            html.Div(
                "failed 通知レコードを確認し、event_id 指定で再送します。"
                "4xx/5xx envelope をそのまま表示します。",
                style={"marginBottom": "8px", "color": "#444"},
            ),
            html.Div(
                [
                    html.H5("Failed Notification List"),
                    html.Label("event_id filter (optional)"),
                    dcc.Input(
                        id="notification-filter-event-id",
                        type="text",
                        value="",
                        style={"width": "100%"},
                    ),
                    html.Label("limit"),
                    dcc.Input(
                        id="notification-filter-limit",
                        type="text",
                        value="100",
                        style={"width": "100%"},
                    ),
                    html.Label("offset"),
                    dcc.Input(
                        id="notification-filter-offset",
                        type="text",
                        value="0",
                        style={"width": "100%"},
                    ),
                    html.Button(
                        "Refresh Failed Notifications",
                        id="notification-refresh-btn",
                        n_clicks=0,
                        style={"marginTop": "8px"},
                    ),
                    html.Pre(
                        id="notification-query-result",
                        children="Press Refresh to fetch failed notifications.",
                        style={"backgroundColor": "#f5f5f5", "padding": "8px", "marginTop": "8px"},
                    ),
                    dag.AgGrid(
                        id="notification-failed-table",
                        rowData=table_rows,
                        columnDefs=(
                            [
                                {"headerName": "event_id", "field": "event_id"},
                                {"headerName": "status", "field": "status"},
                                {"headerName": "retry_count", "field": "retry_count"},
                                {"headerName": "next_retry_at", "field": "next_retry_at"},
                                {"headerName": "last_attempt_at", "field": "last_attempt_at"},
                                {"headerName": "last_error", "field": "last_error"},
                            ]
                            if table_rows
                            else []
                        ),
                        defaultColDef={"resizable": True, "sortable": True, "filter": True},
                        dashGridOptions={"pagination": True, "paginationPageSize": 10},
                        style={"width": "100%", "overflowX": "auto"},
                    ),
                ],
                style={"border": "1px solid #ddd", "padding": "10px", "marginBottom": "10px"},
            ),
            html.Div(
                [
                    html.H5("Retry Failed Notification"),
                    html.Label("event_id"),
                    dcc.Input(
                        id="notification-retry-event-id",
                        type="text",
                        value="",
                        style={"width": "100%"},
                    ),
                    html.Button(
                        "Retry Notification",
                        id="notification-retry-btn",
                        n_clicks=0,
                        style={"marginTop": "8px"},
                    ),
                    html.Pre(
                        id="notification-retry-result",
                        style={"backgroundColor": "#f5f5f5", "padding": "8px", "marginTop": "8px"},
                    ),
                ],
                style={"border": "1px solid #ddd", "padding": "10px", "marginBottom": "10px"},
            ),
            html.Div(
                f"target db_api: {base_url}",
                style={"marginTop": "10px", "color": "#666", "fontSize": "0.9rem"},
            ),
        ]
    )
