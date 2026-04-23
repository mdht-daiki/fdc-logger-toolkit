from typing import Any

from portfolio_fdc.dashboard.view_models import (
    build_chart_name,
    chart_band_figure,
    chart_points_figure,
    empty_drilldown_figure,
    format_range,
    sort_judge_rows,
    spc_band_with_points_figure,
    waveform_figure,
)


def test_sort_judge_rows_prioritizes_ng_warn_ok() -> None:
    rows = [
        {"result_id": "JR_1", "level": "OK", "judged_at": "2026-04-17T00:00:00.000Z"},
        {"result_id": "JR_2", "level": "WARN", "judged_at": "2026-04-17T00:00:01.000Z"},
        {"result_id": "JR_3", "level": "NG", "judged_at": "2026-04-17T00:00:02.000Z"},
    ]

    sorted_rows = sort_judge_rows(rows)

    assert [row["result_id"] for row in sorted_rows] == ["JR_3", "JR_2", "JR_1"]


def test_sort_judge_rows_same_level_descending_judged_at() -> None:
    rows = [
        {"result_id": "JR_1", "level": "WARN", "judged_at": "2026-04-17T00:00:00.000Z"},
        {"result_id": "JR_2", "level": "WARN", "judged_at": "2026-04-17T00:00:02.000Z"},
        {"result_id": "JR_3", "level": "WARN", "judged_at": "2026-04-17T00:00:01.000Z"},
    ]

    sorted_rows = sort_judge_rows(rows)

    assert [row["result_id"] for row in sorted_rows] == ["JR_2", "JR_3", "JR_1"]


def test_sort_judge_rows_places_missing_judged_at_last_within_level() -> None:
    rows: list[dict[str, Any]] = [
        {"result_id": "JR_1", "level": "WARN", "judged_at": None},
        {"result_id": "JR_2", "level": "WARN", "judged_at": "2026-04-17T00:00:02.000Z"},
        {"result_id": "JR_3", "level": "WARN", "judged_at": "2026-04-17T00:00:01.000Z"},
    ]

    sorted_rows = sort_judge_rows(rows)

    assert [row["result_id"] for row in sorted_rows] == ["JR_2", "JR_3", "JR_1"]


def test_chart_band_figure_contains_center_warning_critical() -> None:
    figure = chart_band_figure(
        {
            "warning_lcl": 1.4,
            "warning_ucl": 2.6,
            "critical_lcl": 1.2,
            "critical_ucl": 2.8,
        }
    )

    assert "data" in figure
    names = [trace["name"] for trace in figure["data"]]
    assert "Center" in names
    assert "Warning LCL" in names
    assert "Critical UCL" in names


def test_build_chart_name_uses_explicit_name_when_present() -> None:
    row = {
        "chart_name": "Etch dc_bias step1 mean",
        "chart_id": "CHART_10",
    }

    assert build_chart_name(row) == "Etch dc_bias step1 mean"


def test_build_chart_name_falls_back_to_composed_name() -> None:
    row = {
        "recipe_id": "RCP_A",
        "parameter": "dc_bias",
        "step_no": 1,
        "feature_type": "mean",
        "chart_id": "CHART_10",
    }

    assert build_chart_name(row) == "RCP_A / dc_bias / step:1 / mean (CHART_10)"


def test_build_chart_name_preserves_falsy_value_step_no_zero() -> None:
    """Verify that step_no=0 is displayed as 0, not as "-"."""
    row = {
        "recipe_id": "RCP_B",
        "parameter": "voltage",
        "step_no": 0,
        "feature_type": "median",
        "chart_id": "CHART_20",
    }

    assert build_chart_name(row) == "RCP_B / voltage / step:0 / median (CHART_20)"


def test_chart_points_figure_contains_feature_and_threshold_lines() -> None:
    figure = chart_points_figure(
        {
            "warning_lcl": 1.4,
            "warning_ucl": 2.6,
            "critical_lcl": 1.2,
            "critical_ucl": 2.8,
        },
        [
            {
                "process_id": "P1",
                "feature_value": 2.1,
                "process_start_ts": "2026-04-19T00:00:00.000Z",
            },
            {
                "process_id": "P2",
                "feature_value": 2.4,
                "process_start_ts": "2026-04-19T00:10:00.000Z",
            },
        ],
    )

    names = [trace["name"] for trace in figure["data"]]
    assert "Feature value" in names
    assert "Warning LCL" in names
    assert "Critical UCL" in names


def test_chart_points_figure_ignores_non_numeric_feature_values() -> None:
    figure = chart_points_figure(
        {
            "warning_lcl": "1.4",
            "warning_ucl": "2.6",
            "critical_lcl": "1.2",
            "critical_ucl": "2.8",
        },
        [
            {
                "process_id": "P1",
                "feature_value": "abc",
                "process_start_ts": "2026-04-19T00:00:00.000Z",
            },
            {
                "process_id": "P2",
                "feature_value": "",
                "process_start_ts": "2026-04-19T00:10:00.000Z",
            },
            {
                "process_id": "P3",
                "feature_value": "2.4",
                "process_start_ts": "2026-04-19T00:20:00.000Z",
            },
        ],
    )

    feature_trace = figure["data"][0]
    assert feature_trace["name"] == "Feature value"
    assert feature_trace["y"] == [2.4]


def test_spc_band_with_points_figure_contains_feature_trace_and_customdata() -> None:
    figure = spc_band_with_points_figure(
        {
            "warning_lcl": 1.4,
            "warning_ucl": 2.6,
            "critical_lcl": 1.2,
            "critical_ucl": 2.8,
        },
        [
            {
                "process_id": "P1",
                "feature_value": 2.1,
                "process_start_ts": "2026-04-19T00:00:00.000Z",
            }
        ],
    )

    assert figure["data"][0]["name"] == "Feature value"
    assert figure["data"][0]["customdata"] == ["P1"]


def test_waveform_figure_and_empty_drilldown() -> None:
    empty = empty_drilldown_figure("Click")
    assert empty["layout"]["title"] == "Raw Waveform Drilldown"

    wf = waveform_figure([{"x": "t1", "y": 1.0}, {"x": "t2", "y": 2.0}], "P1")
    assert wf["layout"]["title"] == "Raw Waveform Preview (P1)"
    assert wf["data"][0]["name"] == "Raw waveform"


def test_waveform_figure_ignores_invalid_points() -> None:
    wf = waveform_figure(
        [
            {"x": "t1", "y": "abc"},
            {"x": "t2", "y": ""},
            {"x": "t3", "y": "1.5"},
        ],
        "P1",
    )

    assert wf["data"][0]["y"] == [1.5]


# --- #153: sort_judge_rows 未知のレベル ---


def test_sort_judge_rows_places_unknown_level_at_end() -> None:
    rows = [
        {"result_id": "JR_1", "level": "UNKNOWN", "judged_at": "2026-04-17T00:00:00.000Z"},
        {"result_id": "JR_2", "level": "OK", "judged_at": "2026-04-17T00:00:01.000Z"},
    ]

    sorted_rows = sort_judge_rows(rows)

    # OK は priority=2, UNKNOWN は priority=9 → OK が先
    assert sorted_rows[0]["result_id"] == "JR_2"
    assert sorted_rows[1]["result_id"] == "JR_1"


# --- #153: format_range falsy だが有効な値 ---


def test_format_range_handles_zero_values() -> None:
    assert format_range(0.0, 0.0) == "0.0 .. 0.0"


def test_format_range_returns_dash_for_none() -> None:
    assert format_range(None, 1.0) == "-"
    assert format_range(1.0, None) == "-"
