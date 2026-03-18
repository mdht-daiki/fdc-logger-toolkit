from __future__ import annotations

import importlib
from collections.abc import Iterator
from pathlib import Path

import pytest

import portfolio_fdc.main.aggregate as aggregate_module


def _reload_aggregate_module() -> None:
    importlib.reload(aggregate_module)


@pytest.fixture(autouse=True)
def _restore_recipe_path_env(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """各テスト後に環境変数とモジュール状態を既定へ戻す。"""
    yield
    monkeypatch.delenv(aggregate_module.RECIPE_RULES_PATH_ENV_VAR, raising=False)
    _reload_aggregate_module()


def test_recipe_rules_path_uses_default_when_env_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """PORTFOLIO_RECIPE_RULES_PATH 未設定時は既定の configs/recipe_rules.yaml を使う。"""
    monkeypatch.delenv(aggregate_module.RECIPE_RULES_PATH_ENV_VAR, raising=False)

    _reload_aggregate_module()

    expected = (
        Path(aggregate_module.__file__).resolve().parents[1] / "configs" / "recipe_rules.yaml"
    )
    assert aggregate_module.RECIPE_RULES_PATH == expected


def test_recipe_rules_path_uses_env_override_when_set(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """PORTFOLIO_RECIPE_RULES_PATH が設定されていればそのパスを優先する。"""
    custom_path = tmp_path / "custom_rules" / "recipe_rules.yaml"
    monkeypatch.setenv(aggregate_module.RECIPE_RULES_PATH_ENV_VAR, str(custom_path))

    _reload_aggregate_module()

    assert aggregate_module.RECIPE_RULES_PATH == custom_path.resolve()


def test_recipe_rules_path_resolves_relative_env_path(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """相対パス指定の PORTFOLIO_RECIPE_RULES_PATH は絶対パスに解決される。"""
    monkeypatch.chdir(tmp_path)
    relative_path = Path("relative/subdir/recipe_rules.yaml")
    monkeypatch.setenv(aggregate_module.RECIPE_RULES_PATH_ENV_VAR, str(relative_path))

    _reload_aggregate_module()

    resolved_path = (tmp_path / relative_path).resolve()
    assert aggregate_module.RECIPE_RULES_PATH == resolved_path
