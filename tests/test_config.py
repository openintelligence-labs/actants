from __future__ import annotations

import sys

from agentic_kit.config import AppSettings, app_cache_dir, app_config_dir, app_data_dir


def test_app_config_dir_creates(tmp_path, monkeypatch):
    if sys.platform == "linux":
        monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
        d = app_config_dir("myapp")
        assert d == tmp_path / "myapp"
        assert d.exists()
    else:
        # On macOS / Windows just confirm path is under the right base and gets created.
        d = app_config_dir("agentic-kit-test-myapp")
        try:
            assert d.exists()
        finally:
            if d.exists():
                d.rmdir()


def test_app_data_dir_returns_path():
    d = app_data_dir("agentic-kit-test-data", create=False)
    assert d.name == "agentic-kit-test-data"


def test_app_cache_dir_returns_path():
    d = app_cache_dir("agentic-kit-test-cache", create=False)
    assert d.name == "agentic-kit-test-cache"


class _MyAppSettings(AppSettings):
    model_config = AppSettings.config_for("myapp")
    max_results: int = 10
    name: str = "default"


def test_app_settings_subclass_uses_env_prefix(monkeypatch):
    monkeypatch.setenv("MYAPP_MAX_RESULTS", "42")
    monkeypatch.setenv("MYAPP_NAME", "from-env")
    s = _MyAppSettings()
    assert s.max_results == 42
    assert s.name == "from-env"


def test_app_settings_overrides_win():
    s = _MyAppSettings.load(name="explicit", max_results=5)
    assert s.name == "explicit"
    assert s.max_results == 5
