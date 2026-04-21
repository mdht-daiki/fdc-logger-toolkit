from __future__ import annotations

import logging
from typing import Any

from ..api_client import APIError
from ..dependencies import DashboardDependencies
from ..view_models import build_chart_name


class ChartNameOptionService:
    def __init__(self, logger: logging.Logger, deps: DashboardDependencies) -> None:
        self._logger = logger
        self._deps = deps

    def refresh_chart_name_options(
        self,
        n_clicks: int,
        base_url: str,
        recipe_id: str,
        chart_id: str,
    ) -> tuple[list[dict[str, str]], str | None]:
        if not n_clicks:
            return [], None

        try:
            result = self._deps.validate_base_url(base_url)
            if isinstance(result, tuple):
                safe_base_url = result[0]
            else:
                safe_base_url = result
        except APIError:
            return [], None

        params: dict[str, Any] = {}
        if recipe_id:
            params["recipe_id"] = recipe_id

        try:
            rows = self._deps.get_charts(safe_base_url, params=params)
        except APIError:
            return [], None
        except Exception:
            self._logger.exception("Unexpected error while refreshing chart options")
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
