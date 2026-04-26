from __future__ import annotations

from typing import Any

import pytest

from portfolio_fdc.dashboard.app import app


def _collect_component_ids(node: Any) -> set[str]:
    ids: set[str] = set()
    if isinstance(node, dict):
        if "props" in node and isinstance(node["props"], dict):
            node_id = node["props"].get("id")
            if isinstance(node_id, str):
                ids.add(node_id)
            children = node["props"].get("children")
            if children is not None:
                ids.update(_collect_component_ids(children))
        for value in node.values():
            ids.update(_collect_component_ids(value))
    elif isinstance(node, list):
        for item in node:
            ids.update(_collect_component_ids(item))
    return ids


@pytest.mark.integration
def test_dashboard_layout_exposes_static_store_and_root_components() -> None:
    client = app.server.test_client()

    response = client.get("/_dash-layout")

    assert response.status_code == 200
    layout = response.get_json()
    component_ids = _collect_component_ids(layout)
    assert "active-selected-base-url" in component_ids
    assert "tabs" in component_ids
    assert "tab-content" in component_ids


@pytest.mark.integration
def test_load_data_callback_depends_on_chart_name_state() -> None:
    client = app.server.test_client()

    response = client.get("/_dash-dependencies")

    assert response.status_code == 200
    dependencies = response.get_json()
    assert isinstance(dependencies, list)

    load_callback = next(
        (
            dep
            for dep in dependencies
            if isinstance(dep, dict)
            and isinstance(dep.get("output"), str)
            and "tab-content.children" in dep["output"]
            and "error-banner.children" in dep["output"]
        ),
        None,
    )
    if load_callback is None:
        pytest.fail(
            "expected dependency with 'tab-content.children' and 'error-banner.children' not found"
        )
    assert load_callback is not None

    states = load_callback.get("state", [])
    assert any(
        isinstance(state, dict)
        and state.get("id") == "chart-name"
        and state.get("property") == "value"
        for state in states
    )
