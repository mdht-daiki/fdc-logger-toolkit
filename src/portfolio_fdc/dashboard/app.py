from __future__ import annotations

import ipaddress
import logging
import os
import socket
from typing import Any, cast
from urllib.parse import parse_qs, urlencode, urlparse

from dash import Dash, Input, Output, State, dash_table, dcc, html, no_update

from .api_client import (
    APIError,
    get_active_charts,
    get_chart_points,
    get_charts,
    get_charts_history,
    get_judge_result,
    get_judge_results,
    get_process_waveform_preview,
    parse_utc_millis,
)
from .view_models import (
    LEVEL_COLOR,
    build_chart_name,
    empty_drilldown_figure,
    format_range,
    sort_judge_rows,
    spc_band_with_points_figure,
    waveform_figure,
)

DEFAULT_DB_API_BASE_URL = os.getenv("PORTFOLIO_DB_API_URL", "http://localhost:8000")
logger = logging.getLogger(__name__)


def _default_allowed_db_api_hosts() -> set[str]:
    hosts = {"localhost", "127.0.0.1", "::1"}
    parsed_default = urlparse(DEFAULT_DB_API_BASE_URL)
    if parsed_default.hostname:
        hosts.add(parsed_default.hostname.lower())
    return hosts


def _allowed_db_api_hosts() -> set[str]:
    env_hosts = os.getenv("PORTFOLIO_DB_API_ALLOWED_HOSTS", "")
    hosts = _default_allowed_db_api_hosts()
    for host in env_hosts.split(","):
        normalized = host.strip().lower()
        if normalized:
            hosts.add(normalized)
    return hosts


def _is_restricted_ip(ip_value: ipaddress.IPv4Address | ipaddress.IPv6Address) -> bool:
    return (
        ip_value.is_private
        or ip_value.is_loopback
        or ip_value.is_link_local
        or ip_value.is_multicast
        or ip_value.is_reserved
        or ip_value.is_unspecified
    )


def validate_base_url(base_url: str) -> str:
    raw_value = (base_url or "").strip()
    if not raw_value:
        logger.warning("Rejected empty db_api base URL")
        raise APIError(message="Invalid db_api base URL", code="INVALID_BASE_URL")

    parsed = urlparse(raw_value)
    log_target = f"{parsed.scheme or '-'}://{parsed.hostname or '-'}"

    has_credentials = bool(parsed.username or parsed.password or "@" in parsed.netloc)
    if has_credentials:
        logger.warning("Rejected db_api base URL with credentials: %s", log_target)
        raise APIError(message="Invalid db_api base URL", code="INVALID_BASE_URL")

    if parsed.scheme not in {"http", "https"}:
        logger.warning("Rejected db_api base URL with unsupported scheme: %s", log_target)
        raise APIError(message="Invalid db_api base URL", code="INVALID_BASE_URL")

    if not parsed.hostname:
        logger.warning("Rejected db_api base URL without hostname: %s", log_target)
        raise APIError(message="Invalid db_api base URL", code="INVALID_BASE_URL")

    if parsed.path not in {"", "/"} or parsed.params or parsed.query or parsed.fragment:
        logger.warning("Rejected db_api base URL with path/query/fragment: %s", log_target)
        raise APIError(message="Invalid db_api base URL", code="INVALID_BASE_URL")

    try:
        parsed_port = parsed.port
    except ValueError:
        logger.warning("Rejected db_api base URL with invalid port: %s", log_target)
        raise APIError(message="Invalid db_api base URL", code="INVALID_BASE_URL") from None

    if parsed_port == 0:
        logger.warning("Rejected db_api base URL with disallowed port: %s", log_target)
        raise APIError(message="Invalid db_api base URL", code="INVALID_BASE_URL")

    hostname = parsed.hostname.lower()
    bracketed_host = f"[{hostname}]" if ":" in hostname else hostname
    safe_base_url = f"{parsed.scheme}://{bracketed_host}"
    if parsed_port is not None:
        safe_base_url = f"{safe_base_url}:{parsed_port}"

    if hostname in _allowed_db_api_hosts():
        return safe_base_url

    resolve_host = bracketed_host[1:-1] if bracketed_host.startswith("[") else bracketed_host

    try:
        resolved = socket.getaddrinfo(resolve_host, parsed_port or 80, type=socket.SOCK_STREAM)
    except OSError:
        logger.warning("Rejected db_api base URL; hostname resolution failed: %s", log_target)
        raise APIError(message="Invalid db_api base URL", code="INVALID_BASE_URL") from None

    for result in resolved:
        sockaddr = result[4]
        if not sockaddr:
            continue
        candidate_ip = sockaddr[0]
        try:
            ip_value = ipaddress.ip_address(candidate_ip)
        except ValueError:
            logger.warning("Rejected db_api base URL; invalid resolved IP: %s", log_target)
            raise APIError(message="Invalid db_api base URL", code="INVALID_BASE_URL") from None
        if _is_restricted_ip(ip_value):
            logger.warning("Rejected db_api base URL; restricted network target: %s", log_target)
            raise APIError(message="Invalid db_api base URL", code="INVALID_BASE_URL")

    return safe_base_url


app = Dash(__name__, suppress_callback_exceptions=True, title="FDC Dashboard Baseline")
typed_dash_table = cast(Any, dash_table)
DataTable: Any = typed_dash_table.DataTable

app.layout = html.Div(
    [
        dcc.Location(id="url", refresh=False),
        html.H2("FDC Dashboard Read-only Baseline"),
        html.Div(
            "Issue #146 scope: /charts /charts/active /charts/history "
            "/judge/results /judge/results/{id}"
        ),
        html.Div(
            [
                html.Label("db_api base URL"),
                dcc.Input(
                    id="base-url",
                    type="text",
                    value=DEFAULT_DB_API_BASE_URL,
                    style={"width": "360px"},
                ),
                html.Label("recipe_id"),
                dcc.Input(id="recipe-id", type="text", value="", style={"width": "220px"}),
                html.Label("chart_id"),
                dcc.Input(id="chart-id", type="text", value="", style={"width": "160px"}),
                html.Label("chart_name"),
                dcc.Dropdown(
                    id="chart-name",
                    options=[],
                    value=None,
                    placeholder="Select chart",
                    style={"width": "500px"},
                ),
                html.Label("result_id"),
                dcc.Input(id="result-id", type="text", value="", style={"width": "160px"}),
                html.Button("Load", id="load-btn", n_clicks=0),
            ],
            style={
                "display": "grid",
                "gridTemplateColumns": "auto auto auto auto auto auto auto auto auto auto auto",
                "gap": "8px",
                "alignItems": "center",
                "margin": "12px 0",
            },
        ),
        html.Div(
            id="error-banner",
            style={"color": "#b00020", "fontWeight": "bold", "marginBottom": "8px"},
        ),
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
        html.Div(id="tab-content"),
    ],
    style={"padding": "12px", "fontFamily": "Segoe UI, sans-serif"},
)


def _get_query_value(query_string: str, key: str) -> str:
    parsed = parse_qs((query_string or "").lstrip("?"))
    values = parsed.get(key)
    if not values:
        return ""
    return values[0]


@app.callback(
    Output("tabs", "value"),
    Output("recipe-id", "value"),
    Output("chart-id", "value"),
    Output("result-id", "value"),
    Input("url", "search"),
    prevent_initial_call=False,
)
def sync_filters_from_url(search: str) -> tuple[str, str, str, str]:
    tab = _get_query_value(search, "tab")
    tab_value = tab if tab in {"charts", "active", "history", "judge"} else "charts"
    return (
        tab_value,
        _get_query_value(search, "recipe_id"),
        _get_query_value(search, "chart_id"),
        _get_query_value(search, "result_id"),
    )


@app.callback(
    Output("tab-content", "children"),
    Output("error-banner", "children"),
    Input("tabs", "value"),
    Input("load-btn", "n_clicks"),
    State("base-url", "value"),
    State("recipe-id", "value"),
    State("chart-id", "value"),
    State("result-id", "value"),
)
def load_data(
    active_tab: str,
    n_clicks: int,
    base_url: str,
    recipe_id: str,
    chart_id: str,
    result_id: str,
) -> tuple[Any, str]:
    if not n_clicks:
        return html.Div("Press Load to fetch data"), ""

    try:
        safe_base_url = validate_base_url(base_url)
        if active_tab == "charts":
            return _render_charts_tab(safe_base_url, recipe_id), ""
        if active_tab == "active":
            return _render_active_tab(safe_base_url, recipe_id, chart_id), ""
        if active_tab == "history":
            return _render_history_tab(safe_base_url, chart_id), ""
        return _render_judge_tab(safe_base_url, recipe_id, chart_id, result_id), ""
    except APIError as exc:
        code = f" [{exc.code}]" if exc.code else ""
        return html.Div(""), f"{exc.message}{code}"


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
    if not n_clicks:
        return [], None

    try:
        safe_base_url = validate_base_url(base_url)
    except APIError:
        return [], None

    params: dict[str, Any] = {}
    if recipe_id:
        params["recipe_id"] = recipe_id

    try:
        rows = get_charts(safe_base_url, params=params)
    except APIError:
        return [], None

    options = [
        {
            "label": build_chart_name(row),
            "value": str(row.get("chart_id") or ""),
        }
        for row in rows
        if row.get("chart_id")
    ]

    if chart_id and any(opt["value"] == chart_id for opt in options):
        return options, chart_id
    return options, None


@app.callback(
    Output("tabs", "value", allow_duplicate=True),
    Output("chart-id", "value", allow_duplicate=True),
    Output("url", "search", allow_duplicate=True),
    Input("chart-name", "value"),
    State("recipe-id", "value"),
    prevent_initial_call=True,
)
def move_to_active_by_chart_name(
    selected_chart_id: str | None,
    recipe_id: str,
) -> tuple[str, str, str] | tuple[Any, Any, Any]:
    if not selected_chart_id:
        return no_update, no_update, no_update

    params: dict[str, str] = {
        "tab": "active",
        "chart_id": selected_chart_id,
    }
    if recipe_id:
        params["recipe_id"] = recipe_id

    return "active", selected_chart_id, f"?{urlencode(params)}"


@app.callback(
    Output("chart-name", "value", allow_duplicate=True),
    Input("charts-table", "active_cell"),
    State("charts-table", "data"),
    prevent_initial_call=True,
)
def select_chart_from_table(
    active_cell: dict[str, Any] | None,
    data: list[dict[str, Any]] | None,
) -> str | Any:
    if not active_cell or not data:
        return no_update

    row_idx = active_cell.get("row")
    if not isinstance(row_idx, int) or row_idx < 0 or row_idx >= len(data):
        return no_update

    selected = data[row_idx].get("chart_id")
    if not selected:
        return no_update
    return str(selected)


def _render_charts_tab(base_url: str, recipe_id: str) -> html.Div:
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
            "center": format_range(row.get("warning_lcl"), row.get("warning_ucl")),
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
            DataTable(
                id="charts-table",
                data=table_rows,
                columns=(
                    [
                        {"name": "chart_id", "id": "chart_id"},
                        {"name": "is_active", "id": "is_active"},
                        {"name": "chart_name", "id": "chart_name"},
                        {"name": "recipe_id", "id": "recipe_id"},
                        {"name": "parameter", "id": "parameter"},
                        {"name": "step_no", "id": "step_no"},
                        {"name": "feature_type", "id": "feature_type"},
                        {"name": "center", "id": "center"},
                        {"name": "critical", "id": "critical"},
                        {"name": "updated_at", "id": "updated_at"},
                        {"name": "open", "id": "open", "presentation": "markdown"},
                    ]
                    if table_rows
                    else []
                ),
                page_size=12,
                markdown_options={"link_target": "_self"},
                style_table={"overflowX": "auto"},
            ),
        ]
    )


def _render_active_tab(base_url: str, recipe_id: str, chart_id: str) -> html.Div:
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
            dcc.Store(id="active-selected-base-url", data=base_url),
            dcc.Graph(id="active-main-graph", figure=figure),
            dcc.Graph(
                id="active-drilldown-graph",
                figure=empty_drilldown_figure(
                    "Click a point in the top graph to show raw waveform"
                ),
            ),
            DataTable(
                data=rows,
                columns=[{"name": col, "id": col} for col in rows[0].keys()] if rows else [],
                page_size=10,
                style_table={"overflowX": "auto"},
            ),
        ]
    )


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
    if not click_data:
        return empty_drilldown_figure("Click a point in the top graph to show raw waveform")

    points = click_data.get("points")
    if not isinstance(points, list) or not points:
        return empty_drilldown_figure("Click a feature point to show raw waveform")

    point = points[0]
    process_id = point.get("customdata")
    if not isinstance(process_id, str) or not process_id:
        return empty_drilldown_figure("Only feature points are clickable for drilldown")

    try:
        safe_base_url = validate_base_url(base_url)
        preview = get_process_waveform_preview(
            safe_base_url,
            process_id,
            params={"limit": 500},
        )
    except APIError as exc:
        return empty_drilldown_figure(f"Failed to load waveform: {exc.message}")

    wave_points = preview.get("points", []) if isinstance(preview, dict) else []
    return waveform_figure(wave_points if isinstance(wave_points, list) else [], process_id)


def _render_history_tab(base_url: str, chart_id: str) -> html.Div:
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
            DataTable(
                data=table_rows,
                columns=[{"name": col, "id": col} for col in table_rows[0].keys()]
                if table_rows
                else [],
                page_size=12,
                style_table={"overflowX": "auto"},
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


def _render_judge_tab(base_url: str, recipe_id: str, chart_id: str, result_id: str) -> html.Div:
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
            DataTable(
                data=table_rows,
                columns=(
                    [{"name": col, "id": col} for col in table_rows[0].keys()] if table_rows else []
                ),
                page_size=12,
                style_data_conditional=[
                    {
                        "if": {"filter_query": "{level} = NG"},
                        "backgroundColor": "rgba(176,0,32,0.08)",
                    },
                    {
                        "if": {"filter_query": "{level} = WARN"},
                        "backgroundColor": "rgba(245,124,0,0.08)",
                    },
                    {
                        "if": {"filter_query": "{level} = OK"},
                        "backgroundColor": "rgba(46,125,50,0.08)",
                    },
                ],
                style_table={"overflowX": "auto"},
            ),
            html.H4("Judge Result Detail"),
            detail_block,
        ]
    )


if __name__ == "__main__":
    host = os.getenv("HOST", "127.0.0.1")
    port = int(os.getenv("PORT", "8050"))
    debug = os.getenv("DEBUG", "false").strip().lower() in {"1", "true", "yes", "on"}
    app.run(host=host, port=port, debug=debug)
