from __future__ import annotations

import logging
from typing import Any

from ..api_client import APIError
from ..dependencies import DashboardDependencies
from ..view_models import empty_drilldown_figure, waveform_figure


class ActiveDrilldownService:
    def __init__(self, logger: logging.Logger, deps: DashboardDependencies) -> None:
        self._logger = logger
        self._deps = deps

    def render_active_drilldown(
        self,
        click_data: dict[str, Any] | None,
        base_url: str,
    ) -> dict[str, Any]:
        if not click_data:
            return empty_drilldown_figure("Click a point in the top graph to show raw waveform")

        points = click_data.get("points")
        if not isinstance(points, list) or not points:
            return empty_drilldown_figure("Click a feature point to show raw waveform")

        point = points[0]
        if not isinstance(point, dict):
            return empty_drilldown_figure("Only feature points are clickable for drilldown")
        process_id = point.get("customdata")
        if not isinstance(process_id, str) or not process_id:
            return empty_drilldown_figure("Only feature points are clickable for drilldown")

        try:
            result = self._deps.validate_base_url(base_url)
            if isinstance(result, tuple):
                safe_base_url = result[0]
            else:
                safe_base_url = result
            preview = self._deps.get_process_waveform_preview(
                safe_base_url,
                process_id,
                params={"limit": 500},
            )
        except APIError as exc:
            return empty_drilldown_figure(f"Failed to load waveform: {exc.message}")
        except Exception:
            self._logger.exception("Unexpected error while rendering active drilldown")
            return empty_drilldown_figure("Failed to load waveform: unexpected error")

        wave_points = preview.get("points", []) if isinstance(preview, dict) else []
        return waveform_figure(wave_points if isinstance(wave_points, list) else [], process_id)
