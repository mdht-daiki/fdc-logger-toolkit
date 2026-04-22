from __future__ import annotations

import logging
from typing import Any

from dash import html

from ..api_client import APIError
from ..dependencies import DashboardDependencies


class TabLoadService:
    def __init__(self, logger: logging.Logger, deps: DashboardDependencies) -> None:
        self._logger = logger
        self._deps = deps

    def load_data(
        self,
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
            result = self._deps.validate_base_url(base_url)
            if isinstance(result, tuple):
                safe_base_url = result[0]
            else:
                safe_base_url = result
            if active_tab == "charts":
                return self._deps.render_charts_tab(safe_base_url, recipe_id), ""
            if active_tab == "active":
                return self._deps.render_active_tab(safe_base_url, recipe_id, chart_id), ""
            if active_tab == "history":
                return self._deps.render_history_tab(safe_base_url, chart_id), ""
            return self._deps.render_judge_tab(safe_base_url, recipe_id, chart_id, result_id), ""
        except APIError as exc:
            code = f" [{exc.code}]" if exc.code else ""
            return html.Div(""), f"{exc.message}{code}"
        except Exception:
            self._logger.exception("Unexpected error in load_data callback")
            return html.Div(""), "Unexpected error while loading dashboard data"
