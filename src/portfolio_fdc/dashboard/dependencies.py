from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from dash import html

TabRenderer = Callable[..., html.Div]


@dataclass(frozen=True)
class DashboardDependencies:
    validate_base_url: Callable[[str], str]
    get_charts: Callable[..., list[dict[str, Any]]]
    get_process_waveform_preview: Callable[..., dict[str, Any]]
    render_charts_tab: TabRenderer
    render_active_tab: TabRenderer
    render_history_tab: TabRenderer
    render_judge_tab: TabRenderer
