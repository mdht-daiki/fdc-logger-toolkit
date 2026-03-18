from __future__ import annotations

import importlib
from collections.abc import Iterator
from pathlib import Path

import pytest

import portfolio_fdc.db_api.db as db_module


def _reload_db_module() -> None:
    importlib.reload(db_module)


@pytest.fixture(autouse=True)
def _restore_db_dir_env(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """各テスト後に環境変数とモジュール状態を既定へ戻す。"""
    yield
    monkeypatch.delenv(db_module.DB_DIR_ENV_VAR, raising=False)
    _reload_db_module()


def test_db_dir_uses_default_path_when_env_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    """PORTFOLIO_DB_DIR 未設定時は既定の data/db を使う。"""
    monkeypatch.delenv(db_module.DB_DIR_ENV_VAR, raising=False)

    _reload_db_module()

    expected = Path(db_module.__file__).resolve().parents[3] / "data" / "db"
    assert db_module.DB_DIR == expected
    assert db_module.MAIN_DB == expected / "main.db"
    assert db_module.TEMP_DB == expected / "temp.db"


def test_db_dir_uses_env_override_when_set(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """PORTFOLIO_DB_DIR が設定されていればそのパスを優先する。"""
    custom_dir = tmp_path / "custom_db_dir"
    monkeypatch.setenv(db_module.DB_DIR_ENV_VAR, str(custom_dir))

    _reload_db_module()

    assert db_module.DB_DIR == custom_dir.resolve()
    assert db_module.MAIN_DB == custom_dir.resolve() / "main.db"
    assert db_module.TEMP_DB == custom_dir.resolve() / "temp.db"

    db_module._init_schema(db_module.MAIN_DB)
    assert custom_dir.exists()
    assert db_module.MAIN_DB.exists()


def test_db_dir_resolves_relative_env_path(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """相対パス指定の PORTFOLIO_DB_DIR は絶対パスに解決される。"""
    monkeypatch.chdir(tmp_path)
    relative_dir = Path("relative/subdir")
    monkeypatch.setenv(db_module.DB_DIR_ENV_VAR, str(relative_dir))

    _reload_db_module()

    resolved_dir = (tmp_path / relative_dir).resolve()
    assert db_module.DB_DIR == resolved_dir
    assert db_module.MAIN_DB == resolved_dir / "main.db"
    assert db_module.TEMP_DB == resolved_dir / "temp.db"
