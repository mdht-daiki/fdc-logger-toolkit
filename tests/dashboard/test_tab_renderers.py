from typing import Any
from unittest.mock import MagicMock, patch

from dash import html

from portfolio_fdc.dashboard import tab_renderers


def _find_by_id(node: Any, target_id: str) -> Any:
    node_id = getattr(node, "id", None)
    if node_id == target_id:
        return node

    children = getattr(node, "children", None)
    if children is None:
        return None

    if not isinstance(children, (list, tuple)):
        children = [children]

    for child in children:
        found = _find_by_id(child, target_id)
        if found is not None:
            return found
    return None


# --- render_charts_tab ---
def test_render_charts_tab_minimal():
    with patch("portfolio_fdc.dashboard.tab_renderers.get_charts") as mock_get_charts:
        mock_get_charts.return_value = [
            {
                "chart_id": "C1",
                "is_active": True,
                "recipe_id": "R1",
                "parameter": "P",
                "step_no": 1,
                "feature_type": "F",
                "warning_lcl": 1,
                "warning_ucl": 2,
                "critical_lcl": 3,
                "critical_ucl": 4,
                "updated_at": 1710000000000,
            }
        ]
        div = tab_renderers.render_charts_tab("base", "R1")
        assert isinstance(div, html.Div)
        # get_chartsの呼び出し引数を検証
        mock_get_charts.assert_called_once_with("base", params={"recipe_id": "R1"})
        # ヘッダ文言の厳密一致
        assert div.children[0].children == "Charts: 1 rows"


# --- render_active_tab ---
def test_render_active_tab_minimal():
    with (
        patch("portfolio_fdc.dashboard.tab_renderers.get_active_charts") as mock_get_active_charts,
        patch("portfolio_fdc.dashboard.tab_renderers.get_chart_points") as mock_get_chart_points,
        patch("portfolio_fdc.dashboard.tab_renderers.spc_band_with_points_figure") as mock_fig,
        patch("portfolio_fdc.dashboard.tab_renderers.empty_drilldown_figure") as mock_empty,
        patch("portfolio_fdc.dashboard.tab_renderers.parse_utc_millis") as mock_parse_utc_millis,
    ):
        mock_get_active_charts.return_value = {
            "charts": [
                {
                    "chart_id": "C1",
                    "parameter": "P",
                    "step_no": 1,
                    "feature_type": "F",
                    "warning_lcl": 1,
                    "warning_ucl": 2,
                    "critical_lcl": 3,
                    "critical_ucl": 4,
                }
            ],
            "activated_at": 1710000000000,
        }
        mock_get_chart_points.return_value = [{"x": 1, "y": 2}]
        mock_fig.return_value = MagicMock()
        mock_empty.return_value = MagicMock()
        mock_parse_utc_millis.return_value = "2024-03-10 09:20:00"
        div = tab_renderers.render_active_tab("base", "R1", "C1")
        assert isinstance(div, html.Div)
        mock_get_active_charts.assert_called_once_with("base", params={"recipe_id": "R1"})
        mock_get_chart_points.assert_called_once_with("base", chart_id="C1", params={"limit": 30})
        assert div.children[0].children == "Active chart_set=- / activated_at=2024-03-10 09:20:00"


# --- render_history_tab ---
def test_render_history_tab_minimal():
    with patch(
        "portfolio_fdc.dashboard.tab_renderers.get_charts_history"
    ) as mock_get_charts_history:
        mock_get_charts_history.return_value = [
            {
                "history_id": "H1",
                "chart_id": "C1",
                "change_source": "src",
                "change_reason": "reason",
                "changed_by": "user",
                "changed_at": 1710000000000,
            }
        ]
        div = tab_renderers.render_history_tab("base", "C1")
        assert isinstance(div, html.Div)
        mock_get_charts_history.assert_called_once_with(
            "base", params={"limit": 100, "chart_id": "C1"}
        )
        assert div.children[0].children == "History: 1 rows"


# --- render_judge_tab ---
def test_render_judge_tab_minimal():
    with (
        patch("portfolio_fdc.dashboard.tab_renderers._build_judge_table_rows") as mock_table_rows,
        patch("portfolio_fdc.dashboard.tab_renderers._build_judge_drilldown_links") as mock_links,
        patch("portfolio_fdc.dashboard.tab_renderers._build_judge_detail_block") as mock_detail,
        patch("portfolio_fdc.dashboard.tab_renderers.get_judge_result") as mock_get_judge_result,
    ):
        mock_table_rows.return_value = (
            [{"result_id": "R1", "level": "OK"}],
            [{"result_id": "R1", "level": "OK"}],
        )
        mock_links.return_value = [html.Li("link1")]
        mock_detail.return_value = html.Div("detail")
        mock_get_judge_result.return_value = {"result_id": "RID1", "level": "OK"}
        div = tab_renderers.render_judge_tab("base", "R1", "C1", "RID1")
        assert isinstance(div, html.Div)
        mock_table_rows.assert_called_once_with(
            "base", {"limit": 200, "recipe_id": "R1", "chart_id": "C1"}
        )
        mock_links.assert_called_once()
        mock_detail.assert_called_once()
        mock_get_judge_result.assert_called_once_with("base", result_id="RID1")
        assert div.children[0].children == "Judge Results (priority: NG > WARN > OK)"


def test_render_emergency_tab_minimal():
    div = tab_renderers.render_emergency_tab("http://localhost:8000", "C1")
    assert isinstance(div, html.Div)
    assert div.children[0].children == "Emergency Change / Ratify"

    emergency_card = _find_by_id(div, "emergency-chart-id")
    assert emergency_card is not None, "Could not find emergency form element"
    emergency_chart_id_input = emergency_card
    assert emergency_chart_id_input is not None, "Could not find emergency-chart-id"
    assert emergency_chart_id_input.id == "emergency-chart-id"
    assert emergency_chart_id_input.value == "C1"


def test_render_change_requests_tab_minimal():
    with patch(
        "portfolio_fdc.dashboard.tab_renderers.get_change_requests"
    ) as mock_get_change_requests:
        mock_get_change_requests.return_value = [
            {
                "id": 11,
                "chart_id": 21,
                "status": "pending",
                "proposed_by": "tester",
                "proposed_at": "2026-05-29T00:00:00.000Z",
                "change_payload": '{"warn_low": 20.0}',
                "expected_version": 1,
                "idempotency_key": "idem-11",
            }
        ]
        div = tab_renderers.render_change_requests_tab("http://localhost:8000")

    assert isinstance(div, html.Div)
    assert div.children[0].children == "Change Requests"
    mock_get_change_requests.assert_called_once_with(
        "http://localhost:8000", params={"limit": 100, "offset": 0}
    )

    create_button = _find_by_id(div, "change-request-create-btn")
    assert create_button is not None
    assert create_button.children == "Create Change Request"

    refresh_button = _find_by_id(div, "change-request-refresh-btn")
    assert refresh_button is not None
    assert refresh_button.children == "Refresh Change Requests"


def test_render_notification_retry_tab_minimal():
    with patch(
        "portfolio_fdc.dashboard.tab_renderers.get_failed_notifications"
    ) as mock_get_failed_notifications:
        mock_get_failed_notifications.return_value = [
            {
                "event_id": 101,
                "status": "failed",
                "retry_count": 1,
                "next_retry_at": "2026-05-31T00:10:00.000Z",
                "last_attempt_at": "2026-05-31T00:05:00.000Z",
                "last_error": "smtp timeout",
            }
        ]
        div = tab_renderers.render_notification_retry_tab("http://localhost:8000")

    assert isinstance(div, html.Div)
    assert div.children[0].children == "Notification Retry"
    mock_get_failed_notifications.assert_called_once_with(
        "http://localhost:8000", params={"limit": 100, "offset": 0}
    )

    refresh_button = _find_by_id(div, "notification-refresh-btn")
    assert refresh_button is not None
    assert refresh_button.children == "Refresh Failed Notifications"

    retry_button = _find_by_id(div, "notification-retry-btn")
    assert retry_button is not None
    assert retry_button.children == "Retry Notification"
