from __future__ import annotations

from typing import Any, cast
from urllib.parse import urlencode

from dash import dash_table, dcc, html

from .api_client import (
    get_active_charts,
    get_chart_points,
    get_charts,
    get_charts_history,
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

typed_dash_table = cast(Any, dash_table)
DataTable: Any = typed_dash_table.DataTable


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
                        {"name": "warning", "id": "warning"},
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
            DataTable(
                data=rows,
                columns=[{"name": col, "id": col} for col in rows[0].keys()] if rows else [],
                page_size=10,
                style_table={"overflowX": "auto"},
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
