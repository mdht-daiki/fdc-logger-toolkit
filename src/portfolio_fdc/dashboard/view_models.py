from __future__ import annotations

from typing import Any, cast

LEVEL_PRIORITY: dict[str, int] = {"NG": 0, "WARN": 1, "OK": 2}
LEVEL_COLOR: dict[str, str] = {
    "NG": "#b00020",
    "WARN": "#f57c00",
    "OK": "#2e7d32",
}


def sort_judge_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """NG > WARN > OK を優先し、同順位内は judged_at 降順に並べる。"""

    def _key(row: dict[str, Any]) -> tuple[int, str]:
        level = str(row.get("level", "")).upper()
        judged_at = str(row.get("judged_at", ""))
        return (LEVEL_PRIORITY.get(level, 9), judged_at)

    return sorted(rows, key=_key)


def format_range(low: Any, high: Any) -> str:
    if low is None or high is None:
        return "-"
    return f"{low} .. {high}"


def build_chart_name(row: dict[str, Any]) -> str:
    """ユーザー向け表示名を返す。API が chart_name を返せば優先利用する。"""
    chart_name = row.get("chart_name")
    if isinstance(chart_name, str) and chart_name.strip():
        return chart_name

    recipe_id = str(row.get("recipe_id") or "-")
    parameter = str(row.get("parameter") or "-")
    step_no = str(row.get("step_no") or "-")
    feature_type = str(row.get("feature_type") or "-")
    chart_id = str(row.get("chart_id") or "-")
    return f"{recipe_id} / {parameter} / step:{step_no} / {feature_type} ({chart_id})"


def chart_band_figure(chart: dict[str, Any]) -> dict[str, Any]:
    """Center/Warning/Critical の color band を表す最小 Figure を返す。"""
    warning_lcl = chart.get("warning_lcl")
    warning_ucl = chart.get("warning_ucl")
    critical_lcl = chart.get("critical_lcl")
    critical_ucl = chart.get("critical_ucl")

    if any(v is None for v in (warning_lcl, warning_ucl, critical_lcl, critical_ucl)):
        return {
            "data": [],
            "layout": {
                "title": "No threshold data",
                "xaxis": {"visible": False},
                "yaxis": {"visible": False},
            },
        }

    warning_lcl_f = float(cast(float, warning_lcl))
    warning_ucl_f = float(cast(float, warning_ucl))
    critical_lcl_f = float(cast(float, critical_lcl))
    critical_ucl_f = float(cast(float, critical_ucl))
    center = (warning_lcl_f + warning_ucl_f) / 2.0
    lower_crit = min(critical_lcl_f, warning_lcl_f)
    upper_crit = max(critical_ucl_f, warning_ucl_f)

    shapes = [
        {
            "type": "rect",
            "x0": 0,
            "x1": 1,
            "y0": lower_crit,
            "y1": warning_lcl_f,
            "fillcolor": "rgba(176,0,32,0.25)",
            "line": {"width": 0},
        },
        {
            "type": "rect",
            "x0": 0,
            "x1": 1,
            "y0": warning_lcl_f,
            "y1": warning_ucl_f,
            "fillcolor": "rgba(46,125,50,0.25)",
            "line": {"width": 0},
        },
        {
            "type": "rect",
            "x0": 0,
            "x1": 1,
            "y0": warning_ucl_f,
            "y1": upper_crit,
            "fillcolor": "rgba(176,0,32,0.25)",
            "line": {"width": 0},
        },
    ]

    return {
        "data": [
            {
                "type": "scatter",
                "mode": "lines",
                "x": [0, 1],
                "y": [center, center],
                "name": "Center",
                "line": {"color": "#2e7d32", "width": 3},
            },
            {
                "type": "scatter",
                "mode": "lines",
                "x": [0, 1],
                "y": [warning_lcl_f, warning_lcl_f],
                "name": "Warning LCL",
                "line": {"color": "#f57c00", "dash": "dot"},
            },
            {
                "type": "scatter",
                "mode": "lines",
                "x": [0, 1],
                "y": [warning_ucl_f, warning_ucl_f],
                "name": "Warning UCL",
                "line": {"color": "#f57c00", "dash": "dot"},
            },
            {
                "type": "scatter",
                "mode": "lines",
                "x": [0, 1],
                "y": [critical_lcl_f, critical_lcl_f],
                "name": "Critical LCL",
                "line": {"color": "#b00020", "dash": "dash"},
            },
            {
                "type": "scatter",
                "mode": "lines",
                "x": [0, 1],
                "y": [critical_ucl_f, critical_ucl_f],
                "name": "Critical UCL",
                "line": {"color": "#b00020", "dash": "dash"},
            },
        ],
        "layout": {
            "title": "Color Band (Center / Warning / Critical)",
            "xaxis": {"visible": False, "range": [0, 1]},
            "yaxis": {"title": "threshold"},
            "shapes": shapes,
            "height": 320,
            "margin": {"l": 40, "r": 20, "t": 50, "b": 30},
        },
    }


def chart_points_figure(chart: dict[str, Any], points: list[dict[str, Any]]) -> dict[str, Any]:
    """特徴量点としきい値線を重ねた Figure を返す。"""
    if not points:
        return {
            "data": [],
            "layout": {
                "title": "No feature points",
                "xaxis": {"visible": False},
                "yaxis": {"visible": False},
            },
        }

    ordered = list(reversed(points))
    x = list(range(1, len(ordered) + 1))
    y = [float(point.get("feature_value", 0.0)) for point in ordered]
    hover = [
        f"process_id={point.get('process_id')}<br>start={point.get('process_start_ts')}"
        for point in ordered
    ]

    warning_lcl = chart.get("warning_lcl")
    warning_ucl = chart.get("warning_ucl")
    critical_lcl = chart.get("critical_lcl")
    critical_ucl = chart.get("critical_ucl")

    traces: list[dict[str, Any]] = [
        {
            "type": "scatter",
            "mode": "markers+lines",
            "x": x,
            "y": y,
            "name": "Feature value",
            "marker": {"color": "#1565c0", "size": 7},
            "line": {"color": "rgba(21,101,192,0.35)", "width": 1},
            "text": hover,
            "hovertemplate": "%{text}<br>value=%{y}<extra></extra>",
        }
    ]

    def _hline(name: str, value: Any, color: str, dash: str) -> dict[str, Any] | None:
        if value is None:
            return None
        return {
            "type": "scatter",
            "mode": "lines",
            "x": [x[0], x[-1]],
            "y": [value, value],
            "name": name,
            "line": {"color": color, "dash": dash},
            "hoverinfo": "skip",
        }

    for threshold_line in (
        _hline("Warning LCL", warning_lcl, "#f57c00", "dot"),
        _hline("Warning UCL", warning_ucl, "#f57c00", "dot"),
        _hline("Critical LCL", critical_lcl, "#b00020", "dash"),
        _hline("Critical UCL", critical_ucl, "#b00020", "dash"),
    ):
        if threshold_line is not None:
            traces.append(threshold_line)

    return {
        "data": traces,
        "layout": {
            "title": "Recent Feature Points",
            "xaxis": {"title": "sample (old -> new)"},
            "yaxis": {"title": "feature_value"},
            "height": 360,
            "margin": {"l": 50, "r": 20, "t": 50, "b": 45},
        },
    }


def spc_band_with_points_figure(
    chart: dict[str, Any], points: list[dict[str, Any]]
) -> dict[str, Any]:
    """上段用: color band と特徴量点を同一グラフに重ねる。"""
    if not points:
        return chart_band_figure(chart)

    ordered = list(reversed(points))
    x = list(range(1, len(ordered) + 1))
    y = [float(point.get("feature_value", 0.0)) for point in ordered]
    process_ids = [str(point.get("process_id", "")) for point in ordered]
    hover = [
        "process_id="
        f"{point.get('process_id')}<br>start={point.get('process_start_ts')}"
        f"<br>value={point.get('feature_value')}"
        for point in ordered
    ]

    warning_lcl = chart.get("warning_lcl")
    warning_ucl = chart.get("warning_ucl")
    critical_lcl = chart.get("critical_lcl")
    critical_ucl = chart.get("critical_ucl")
    center = None
    if warning_lcl is not None and warning_ucl is not None:
        center = (float(cast(float, warning_lcl)) + float(cast(float, warning_ucl))) / 2.0

    traces: list[dict[str, Any]] = [
        {
            "type": "scatter",
            "mode": "markers+lines",
            "x": x,
            "y": y,
            "name": "Feature value",
            "marker": {"color": "#1565c0", "size": 8},
            "line": {"color": "rgba(21,101,192,0.35)", "width": 1},
            "text": hover,
            "customdata": process_ids,
            "hovertemplate": "%{text}<extra></extra>",
        }
    ]

    def _hline(name: str, value: Any, color: str, dash: str, width: int = 2) -> None:
        if value is None:
            return
        traces.append(
            {
                "type": "scatter",
                "mode": "lines",
                "x": [x[0], x[-1]],
                "y": [value, value],
                "name": name,
                "line": {"color": color, "dash": dash, "width": width},
                "hoverinfo": "skip",
            }
        )

    _hline("Center", center, "#2e7d32", "solid", 3)
    _hline("Warning LCL", warning_lcl, "#f57c00", "dot")
    _hline("Warning UCL", warning_ucl, "#f57c00", "dot")
    _hline("Critical LCL", critical_lcl, "#b00020", "dash")
    _hline("Critical UCL", critical_ucl, "#b00020", "dash")

    shapes: list[dict[str, Any]] = []
    if None not in (warning_lcl, warning_ucl, critical_lcl, critical_ucl):
        warning_lcl_f = float(cast(float, warning_lcl))
        warning_ucl_f = float(cast(float, warning_ucl))
        critical_lcl_f = float(cast(float, critical_lcl))
        critical_ucl_f = float(cast(float, critical_ucl))
        lower_crit = min(critical_lcl_f, warning_lcl_f)
        upper_crit = max(critical_ucl_f, warning_ucl_f)
        x0 = x[0]
        x1 = x[-1]
        shapes = [
            {
                "type": "rect",
                "x0": x0,
                "x1": x1,
                "y0": lower_crit,
                "y1": warning_lcl_f,
                "fillcolor": "rgba(176,0,32,0.15)",
                "line": {"width": 0},
                "layer": "below",
            },
            {
                "type": "rect",
                "x0": x0,
                "x1": x1,
                "y0": warning_lcl_f,
                "y1": warning_ucl_f,
                "fillcolor": "rgba(46,125,50,0.15)",
                "line": {"width": 0},
                "layer": "below",
            },
            {
                "type": "rect",
                "x0": x0,
                "x1": x1,
                "y0": warning_ucl_f,
                "y1": upper_crit,
                "fillcolor": "rgba(176,0,32,0.15)",
                "line": {"width": 0},
                "layer": "below",
            },
        ]

    return {
        "data": traces,
        "layout": {
            "title": "SPC Band + Feature Points",
            "xaxis": {"title": "sample (old -> new)"},
            "yaxis": {"title": "feature_value"},
            "height": 360,
            "margin": {"l": 50, "r": 20, "t": 50, "b": 45},
            "shapes": shapes,
        },
    }


def empty_drilldown_figure(message: str) -> dict[str, Any]:
    return {
        "data": [],
        "layout": {
            "title": "Raw Waveform Drilldown",
            "xaxis": {"visible": False},
            "yaxis": {"visible": False},
            "annotations": [
                {
                    "text": message,
                    "x": 0.5,
                    "y": 0.5,
                    "showarrow": False,
                }
            ],
            "height": 320,
            "margin": {"l": 40, "r": 20, "t": 50, "b": 30},
        },
    }


def waveform_figure(wave_points: list[dict[str, Any]], process_id: str) -> dict[str, Any]:
    if not wave_points:
        return empty_drilldown_figure("No waveform preview data for this process")

    x = [str(p.get("x", "")) for p in wave_points]
    y = [float(p.get("y", 0.0)) for p in wave_points]
    return {
        "data": [
            {
                "type": "scatter",
                "mode": "lines",
                "x": x,
                "y": y,
                "name": "Raw waveform",
                "line": {"color": "#263238", "width": 1.5},
            }
        ],
        "layout": {
            "title": f"Raw Waveform Preview ({process_id})",
            "xaxis": {"title": "timestamp"},
            "yaxis": {"title": "value"},
            "height": 320,
            "margin": {"l": 50, "r": 20, "t": 50, "b": 45},
        },
    }
