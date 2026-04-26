from __future__ import annotations

import logging
from typing import Any

from .dependencies import DashboardDependencies
from .services import (
    ActiveDrilldownService,
    ChartNameOptionService,
    NavigationService,
    TabLoadService,
    UrlFilterService,
)


class DashboardController:
    """Facade that delegates callback behaviors to focused services."""

    def __init__(self, logger: logging.Logger, deps: DashboardDependencies) -> None:
        self._url_filter = UrlFilterService()
        self._tab_loader = TabLoadService(logger, deps)
        self._chart_name = ChartNameOptionService(logger, deps)
        self._navigation = NavigationService()
        self._drilldown = ActiveDrilldownService(logger, deps)

    def sync_filters_from_url(self, search: str) -> tuple[str, str, str, str]:
        return self._url_filter.sync_filters_from_url(search)

    def load_data(
        self,
        active_tab: str,
        n_clicks: int,
        base_url: str,
        recipe_id: str,
        chart_id: str,
        result_id: str,
        selected_chart_id: str | None,
    ) -> tuple[Any, str]:
        return self._tab_loader.load_data(
            active_tab,
            n_clicks,
            base_url,
            recipe_id,
            chart_id,
            result_id,
            selected_chart_id,
        )

    def refresh_chart_name_options(
        self,
        n_clicks: int,
        base_url: str,
        recipe_id: str,
        chart_id: str,
    ) -> tuple[list[dict[str, str]], str | None]:
        return self._chart_name.refresh_chart_name_options(
            n_clicks,
            base_url,
            recipe_id,
            chart_id,
        )

    def move_to_active_by_chart_name(
        self,
        selected_chart_id: str | None,
        recipe_id: str,
        current_search: str,
    ) -> tuple[str, str, str] | tuple[Any, Any, Any]:
        return self._navigation.move_to_active_by_chart_name(
            selected_chart_id,
            recipe_id,
            current_search,
        )

    def select_chart_from_table(
        self,
        active_cell: dict[str, Any] | None,
        data: list[dict[str, Any]] | None,
    ) -> str | Any:
        return self._navigation.select_chart_from_table(active_cell, data)

    def sync_active_selected_base_url(self, base_url: str) -> str:
        return self._navigation.sync_active_selected_base_url(base_url)

    def render_active_drilldown(
        self,
        click_data: dict[str, Any] | None,
        base_url: str,
    ) -> dict[str, Any]:
        return self._drilldown.render_active_drilldown(click_data, base_url)
