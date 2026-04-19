from __future__ import annotations

from typing import Any

LEVEL_PRIORITY: dict[str, int] = {"NG": 0, "WARN": 1, "OK": 2}
LEVEL_COLOR: dict[str, str] = {
    "NG": "#b00020",
    "WARN": "#f57c00",
    "OK": "#2e7d32",
}


def sort_judge_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """NG > WARN > OK を優先し、同順位内は judged_at 降順に並べる。"""

    rows_by_judged_at = sorted(
        rows,
        key=lambda row: str(row.get("judged_at") or ""),
        reverse=True,
    )

    def _key(row: dict[str, Any]) -> int:
        level = str(row.get("level", "")).upper()
        return LEVEL_PRIORITY.get(level, 9)

    return sorted(rows_by_judged_at, key=_key)


def format_range(low: Any, high: Any) -> str:
    if low is None or high is None:
        return "-"
    return f"{low} .. {high}"


def safe_cast_float(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, str):
        trimmed = value.strip()
        if not trimmed:
            return None
        value = trimmed
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def build_chart_name(row: dict[str, Any]) -> str:
    """ユーザー向け表示名を返す。API が chart_name を返せば優先利用する。"""
    chart_name = row.get("chart_name")
    if isinstance(chart_name, str) and chart_name.strip():
        return chart_name

    recipe_id_raw = row.get("recipe_id")
    recipe_id = "-" if recipe_id_raw is None else str(recipe_id_raw)

    parameter_raw = row.get("parameter")
    parameter = "-" if parameter_raw is None else str(parameter_raw)

    step_no_raw = row.get("step_no")
    step_no = "-" if step_no_raw is None else str(step_no_raw)

    feature_type_raw = row.get("feature_type")
    feature_type = "-" if feature_type_raw is None else str(feature_type_raw)

    chart_id_raw = row.get("chart_id")
    chart_id = "-" if chart_id_raw is None else str(chart_id_raw)

    return f"{recipe_id} / {parameter} / step:{step_no} / {feature_type} ({chart_id})"


def chart_band_figure(chart: dict[str, Any]) -> dict[str, Any]:
    """Center/Warning/Critical の color band を表す最小 Figure を返す。"""
    warning_lcl = chart.get("warning_lcl")
    warning_ucl = chart.get("warning_ucl")
    critical_lcl = chart.get("critical_lcl")
    critical_ucl = chart.get("critical_ucl")

    warning_lcl_f = safe_cast_float(warning_lcl)
    warning_ucl_f = safe_cast_float(warning_ucl)
    critical_lcl_f = safe_cast_float(critical_lcl)
    critical_ucl_f = safe_cast_float(critical_ucl)

    if any(v is None for v in (warning_lcl_f, warning_ucl_f, critical_lcl_f, critical_ucl_f)):
        return {
            "data": [],
            "layout": {
                "title": "No threshold data",
                "xaxis": {"visible": False},
                "yaxis": {"visible": False},
            },
        }

    assert warning_lcl_f is not None
    assert warning_ucl_f is not None
    assert critical_lcl_f is not None
    assert critical_ucl_f is not None

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

    filtered_points: list[tuple[dict[str, Any], float]] = []
    for point in ordered:
        feature_value = safe_cast_float(point.get("feature_value"))
        if feature_value is None:
            continue
        filtered_points.append((point, feature_value))

    if not filtered_points:
        return {
            "data": [],
            "layout": {
                "title": "No feature points",
                "xaxis": {"visible": False},
                "yaxis": {"visible": False},
            },
        }

    x = list(range(1, len(filtered_points) + 1))
    y = [feature_value for _, feature_value in filtered_points]
    hover = [
        f"process_id={point.get('process_id')}<br>start={point.get('process_start_ts')}"
        for point, _ in filtered_points
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
        value_f = safe_cast_float(value)
        if value_f is None:
            return None
        return {
            "type": "scatter",
            "mode": "lines",
            "x": [x[0], x[-1]],
            "y": [value_f, value_f],
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
    filtered_points: list[tuple[dict[str, Any], float]] = []
    for point in ordered:
        feature_value = safe_cast_float(point.get("feature_value"))
        if feature_value is None:
            continue
        filtered_points.append((point, feature_value))

    if not filtered_points:
        return chart_band_figure(chart)

    x = list(range(1, len(filtered_points) + 1))
    y = [feature_value for _, feature_value in filtered_points]
    process_ids = [str(point.get("process_id", "")) for point, _ in filtered_points]
    hover = [
        "process_id="
        f"{point.get('process_id')}<br>start={point.get('process_start_ts')}"
        f"<br>value={feature_value}"
        for point, feature_value in filtered_points
    ]

    warning_lcl = chart.get("warning_lcl")
    warning_ucl = chart.get("warning_ucl")
    critical_lcl = chart.get("critical_lcl")
    critical_ucl = chart.get("critical_ucl")

    warning_lcl_f = safe_cast_float(warning_lcl)
    warning_ucl_f = safe_cast_float(warning_ucl)
    critical_lcl_f = safe_cast_float(critical_lcl)
    critical_ucl_f = safe_cast_float(critical_ucl)

    center = None
    if warning_lcl_f is not None and warning_ucl_f is not None:
        center = (warning_lcl_f + warning_ucl_f) / 2.0

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
    _hline("Warning LCL", warning_lcl_f, "#f57c00", "dot")
    _hline("Warning UCL", warning_ucl_f, "#f57c00", "dot")
    _hline("Critical LCL", critical_lcl_f, "#b00020", "dash")
    _hline("Critical UCL", critical_ucl_f, "#b00020", "dash")

    shapes: list[dict[str, Any]] = []
    if None not in (warning_lcl_f, warning_ucl_f, critical_lcl_f, critical_ucl_f):
        assert warning_lcl_f is not None
        assert warning_ucl_f is not None
        assert critical_lcl_f is not None
        assert critical_ucl_f is not None
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

    valid_wave_points: list[dict[str, Any]] = []
    for point in wave_points:
        y_val = safe_cast_float(point.get("y"))
        if y_val is None:
            continue
        valid_wave_points.append(
            {
                "x": str(point.get("x", "")),
                "y": y_val,
            }
        )

    if not valid_wave_points:
        return empty_drilldown_figure("No valid waveform preview data for this process")

    x = [point["x"] for point in valid_wave_points]
    y = [point["y"] for point in valid_wave_points]
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
