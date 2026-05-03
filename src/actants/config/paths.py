from __future__ import annotations

import os
import sys
from pathlib import Path


def _platform_base(macos: str, windows_env: str, xdg_env: str, xdg_default: str) -> Path:
    if sys.platform == "darwin":
        return Path.home() / "Library" / macos
    if sys.platform == "win32":
        return Path(os.environ.get(windows_env) or Path.home() / "AppData" / "Local")
    explicit = os.environ.get(xdg_env)
    if explicit:
        return Path(explicit).expanduser()
    return Path.home() / xdg_default


def app_config_dir(app_name: str, *, create: bool = True) -> Path:
    """Per-user config directory for ``app_name``.

    macOS: ``~/Library/Application Support/<app>``
    Linux: ``$XDG_CONFIG_HOME/<app>`` or ``~/.config/<app>``
    Windows: ``%APPDATA%/<app>``
    """
    if sys.platform == "win32":
        base = Path(os.environ.get("APPDATA") or Path.home() / "AppData" / "Roaming")
    elif sys.platform == "darwin":
        base = Path.home() / "Library" / "Application Support"
    else:
        explicit = os.environ.get("XDG_CONFIG_HOME")
        base = Path(explicit).expanduser() if explicit else Path.home() / ".config"
    path = base / app_name
    if create:
        path.mkdir(parents=True, exist_ok=True)
    return path


def app_data_dir(app_name: str, *, create: bool = True) -> Path:
    """Per-user data directory (databases, models, large state)."""
    base = _platform_base(
        macos="Application Support",
        windows_env="LOCALAPPDATA",
        xdg_env="XDG_DATA_HOME",
        xdg_default=".local/share",
    )
    path = base / app_name
    if create:
        path.mkdir(parents=True, exist_ok=True)
    return path


def app_cache_dir(app_name: str, *, create: bool = True) -> Path:
    """Per-user cache directory (regenerable artifacts)."""
    base = _platform_base(
        macos="Caches",
        windows_env="LOCALAPPDATA",
        xdg_env="XDG_CACHE_HOME",
        xdg_default=".cache",
    )
    path = base / app_name
    if create:
        path.mkdir(parents=True, exist_ok=True)
    return path
