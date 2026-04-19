from __future__ import annotations

from typing import Any
from urllib.parse import urlencode

from dash import no_update

from ..base_url import DEFAULT_DB_API_BASE_URL


class NavigationService:
    def move_to_active_by_chart_name(
        self,
        selected_chart_id: str | None,
        recipe_id: str,
        current_search: str,
    ) -> tuple[str, str, str] | tuple[Any, Any, Any]:
        if not selected_chart_id:
            return no_update, no_update, no_update

        params: dict[str, str] = {
            "tab": "active",
            "chart_id": selected_chart_id,
        }
        if recipe_id:
            params["recipe_id"] = recipe_id

        next_search = f"?{urlencode(params)}"
        if next_search == (current_search or ""):
            return no_update, no_update, no_update

        return "active", selected_chart_id, next_search

    def select_chart_from_table(
        self,
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

    def sync_active_selected_base_url(self, base_url: str) -> str:
        return base_url or DEFAULT_DB_API_BASE_URL
