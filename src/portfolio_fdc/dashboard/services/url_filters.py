from __future__ import annotations

from urllib.parse import parse_qs


class UrlFilterService:
    def sync_filters_from_url(self, search: str) -> tuple[str, str, str, str]:
        tab = self._get_query_value(search, "tab")
        tab_value = tab if tab in {"charts", "active", "history", "judge"} else "charts"
        return (
            tab_value,
            self._get_query_value(search, "recipe_id"),
            self._get_query_value(search, "chart_id"),
            self._get_query_value(search, "result_id"),
        )

    def _get_query_value(self, query_string: str, key: str) -> str:
        parsed = parse_qs((query_string or "").lstrip("?"))
        values = parsed.get(key)
        if not values:
            return ""
        return values[0]
