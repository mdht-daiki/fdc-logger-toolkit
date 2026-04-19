from __future__ import annotations

from typing import Any

import pytest

from portfolio_fdc.dashboard.app import refresh_chart_name_options


def test_refresh_chart_name_options_keeps_dropdown_unselected_without_chart_id(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def _fake_get_charts(*_args: Any, **_kwargs: Any) -> list[dict[str, Any]]:
        return [
            {"chart_id": "CHART_1", "chart_name": "Chart One"},
            {"chart_id": "CHART_2", "chart_name": "Chart Two"},
        ]

    monkeypatch.setattr("portfolio_fdc.dashboard.app.get_charts", _fake_get_charts)

    options, value = refresh_chart_name_options(0, "http://localhost:8000", "", "")

    assert [option["value"] for option in options] == ["CHART_1", "CHART_2"]
    assert value is None


def test_refresh_chart_name_options_preserves_explicit_chart_id(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def _fake_get_charts(*_args: Any, **_kwargs: Any) -> list[dict[str, Any]]:
        return [
            {"chart_id": "CHART_1", "chart_name": "Chart One"},
            {"chart_id": "CHART_2", "chart_name": "Chart Two"},
        ]

    monkeypatch.setattr("portfolio_fdc.dashboard.app.get_charts", _fake_get_charts)

    options, value = refresh_chart_name_options(
        0,
        "http://localhost:8000",
        "",
        "CHART_2",
    )

    assert [option["value"] for option in options] == ["CHART_1", "CHART_2"]
    assert value == "CHART_2"
