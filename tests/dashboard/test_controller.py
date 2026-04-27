"""Smoke tests for DashboardController delegation to services."""

from __future__ import annotations

import logging
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from portfolio_fdc.dashboard.controller import DashboardController
from portfolio_fdc.dashboard.dependencies import DashboardDependencies


@pytest.fixture
def logger() -> logging.Logger:
    """Return test logger."""
    return logging.getLogger("test")


@pytest.fixture
def deps() -> MagicMock:
    """Return mocked DashboardDependencies."""
    return MagicMock(spec=DashboardDependencies)


@pytest.fixture
def controller(logger: logging.Logger, deps: MagicMock) -> DashboardController:
    """Return DashboardController instance."""
    return DashboardController(logger, deps)


class TestSyncFiltersFromUrl:
    """Tests for sync_filters_from_url delegation."""

    def test_delegates_to_url_filter_service(self, controller: DashboardController) -> None:
        """sync_filters_from_url should delegate to UrlFilterService.sync_filters_from_url."""
        search = "recipe_id=R1&chart_id=C1"
        expected = ("recipe", "chart", "result", "selected")
        with patch.object(
            controller._url_filter,
            "sync_filters_from_url",
            return_value=expected,
        ) as mock_sync:
            result = controller.sync_filters_from_url(search)

        mock_sync.assert_called_once_with(search)
        assert result == expected

    def test_returns_url_filter_result(self, controller: DashboardController) -> None:
        """sync_filters_from_url should return the result from UrlFilterService."""
        search = "recipe_id=R1&chart_id=C1"
        expected = ("recipe_id", "chart_id", "result_id", "selected_chart_id")
        with patch.object(
            controller._url_filter,
            "sync_filters_from_url",
            return_value=expected,
        ):
            result = controller.sync_filters_from_url(search)

        assert result == expected


class TestLoadData:
    """Tests for load_data delegation."""

    def test_delegates_with_correct_arguments(self, controller: DashboardController) -> None:
        """load_data should delegate all arguments to TabLoadService."""
        with patch.object(
            controller._tab_loader, "load_data", return_value=({"data": []}, "")
        ) as mock_load:
            controller.load_data(
                active_tab="spc",
                n_clicks=1,
                base_url="http://localhost:8050",
                recipe_id="recipe_1",
                chart_id="CHART_1",
                result_id="JR_1",
                selected_chart_id="CHART_2",
            )

            mock_load.assert_called_once_with(
                "spc",
                1,
                "http://localhost:8050",
                "recipe_1",
                "CHART_1",
                "JR_1",
                "CHART_2",
            )

    def test_returns_loader_result(self, controller: DashboardController) -> None:
        """load_data should return the result from TabLoadService."""
        expected = ({"data": "test"}, "content")
        with patch.object(controller._tab_loader, "load_data", return_value=expected):
            result = controller.load_data(
                active_tab="spc",
                n_clicks=1,
                base_url="http://localhost:8050",
                recipe_id="recipe_1",
                chart_id="CHART_1",
                result_id="JR_1",
                selected_chart_id=None,
            )

            assert result == expected


class TestRefreshChartNameOptions:
    """Tests for refresh_chart_name_options delegation."""

    def test_delegates_with_correct_arguments(self, controller: DashboardController) -> None:
        """refresh_chart_name_options should delegate arguments to ChartNameOptionService."""
        with patch.object(
            controller._chart_name,
            "refresh_chart_name_options",
            return_value=([], None),
        ) as mock_refresh:
            controller.refresh_chart_name_options(
                n_clicks=2,
                base_url="http://localhost:8050",
                recipe_id="recipe_1",
                chart_id="CHART_1",
            )

            mock_refresh.assert_called_once_with(
                2,
                "http://localhost:8050",
                "recipe_1",
                "CHART_1",
            )

    def test_returns_options_result(self, controller: DashboardController) -> None:
        """refresh_chart_name_options should return result from ChartNameOptionService."""
        expected_options = [{"label": "Chart 1", "value": "CHART_1"}]
        expected = (expected_options, "recipe_1")
        with patch.object(
            controller._chart_name, "refresh_chart_name_options", return_value=expected
        ):
            result = controller.refresh_chart_name_options(
                n_clicks=1,
                base_url="http://localhost:8050",
                recipe_id="recipe_1",
                chart_id="CHART_1",
            )

            assert result == expected


class TestMoveToActiveByChartName:
    """Tests for move_to_active_by_chart_name delegation."""

    def test_delegates_with_correct_arguments(self, controller: DashboardController) -> None:
        """move_to_active_by_chart_name should delegate arguments to NavigationService."""
        with patch.object(
            controller._navigation,
            "move_to_active_by_chart_name",
            return_value=("tab", "url", "search"),
        ) as mock_move:
            controller.move_to_active_by_chart_name(
                selected_chart_id="CHART_1",
                recipe_id="recipe_1",
                current_search="recipe_id=R1",
            )

            mock_move.assert_called_once_with(
                "CHART_1",
                "recipe_1",
                "recipe_id=R1",
            )

    def test_returns_navigation_result(self, controller: DashboardController) -> None:
        """move_to_active_by_chart_name should return result from NavigationService."""
        expected = ("tab", "http://localhost:8050", "recipe_id=R1")
        with patch.object(
            controller._navigation, "move_to_active_by_chart_name", return_value=expected
        ):
            result = controller.move_to_active_by_chart_name(
                selected_chart_id="CHART_1",
                recipe_id="recipe_1",
                current_search="recipe_id=R1",
            )

            assert result == expected


class TestSelectChartFromTable:
    """Tests for select_chart_from_table delegation."""

    def test_delegates_with_active_cell_and_data(self, controller: DashboardController) -> None:
        """select_chart_from_table should delegate arguments to NavigationService."""
        active_cell = {"row": 0, "column": 0}
        data = [{"chart_id": "CHART_1"}]

        with patch.object(
            controller._navigation,
            "select_chart_from_table",
            return_value="CHART_1",
        ) as mock_select:
            controller.select_chart_from_table(active_cell, data)

            mock_select.assert_called_once_with(active_cell, data)

    def test_handles_none_arguments(self, controller: DashboardController) -> None:
        """select_chart_from_table should handle None arguments."""
        with patch.object(
            controller._navigation,
            "select_chart_from_table",
            return_value=None,
        ) as mock_select:
            result = controller.select_chart_from_table(None, None)

            mock_select.assert_called_once_with(None, None)
            assert result is None


class TestSyncActiveSelectedBaseUrl:
    """Tests for sync_active_selected_base_url delegation."""

    def test_delegates_base_url(self, controller: DashboardController) -> None:
        """sync_active_selected_base_url should delegate to NavigationService."""
        with patch.object(
            controller._navigation,
            "sync_active_selected_base_url",
            return_value="http://localhost:8050",
        ) as mock_sync:
            result = controller.sync_active_selected_base_url("http://localhost:8050")

            mock_sync.assert_called_once_with("http://localhost:8050")
            assert result == "http://localhost:8050"


class TestRenderActiveDrilldown:
    """Tests for render_active_drilldown delegation."""

    def test_delegates_with_click_data_and_base_url(self, controller: DashboardController) -> None:
        """render_active_drilldown should delegate arguments to ActiveDrilldownService."""
        click_data: dict[str, Any] = {"points": [{"customdata": "process_1"}]}

        with patch.object(
            controller._drilldown,
            "render_active_drilldown",
            return_value={"data": []},
        ) as mock_render:
            controller.render_active_drilldown(click_data, "http://localhost:8050")

            mock_render.assert_called_once_with(click_data, "http://localhost:8050")

    def test_handles_none_click_data(self, controller: DashboardController) -> None:
        """render_active_drilldown should handle None click_data."""
        with patch.object(
            controller._drilldown,
            "render_active_drilldown",
            return_value={"data": []},
        ) as mock_render:
            result = controller.render_active_drilldown(None, "http://localhost:8050")

            mock_render.assert_called_once_with(None, "http://localhost:8050")
            assert isinstance(result, dict)


class TestControllerInitialization:
    """Tests for DashboardController initialization."""

    def test_initializes_all_services(self, logger: logging.Logger, deps: MagicMock) -> None:
        """DashboardController should initialize all required services."""
        controller = DashboardController(logger, deps)

        assert hasattr(controller, "_url_filter")
        assert hasattr(controller, "_tab_loader")
        assert hasattr(controller, "_chart_name")
        assert hasattr(controller, "_navigation")
        assert hasattr(controller, "_drilldown")

    def test_service_initialization_with_deps(
        self, logger: logging.Logger, deps: MagicMock
    ) -> None:
        """DashboardController should pass deps to services that require them."""
        with (
            patch("portfolio_fdc.dashboard.controller.TabLoadService") as mock_tab_load,
            patch("portfolio_fdc.dashboard.controller.ChartNameOptionService") as mock_chart_name,
            patch("portfolio_fdc.dashboard.controller.ActiveDrilldownService") as mock_drilldown,
        ):
            DashboardController(logger, deps)

        mock_tab_load.assert_called_once_with(logger, deps)
        mock_chart_name.assert_called_once_with(logger, deps)
        mock_drilldown.assert_called_once_with(logger, deps)
