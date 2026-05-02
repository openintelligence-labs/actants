"""App-level settings via AppSettings + per-app paths.

Run::

    MYAPP_MAX_RESULTS=42 python examples/11_app_settings.py
"""

from __future__ import annotations

from agentic_kit import AppSettings, app_cache_dir, app_config_dir, app_data_dir


class MySettings(AppSettings):
    model_config = AppSettings.config_for("myapp")
    max_results: int = 10
    cache_ttl: int = 3600


def main() -> None:
    s = MySettings()
    print("max_results:", s.max_results, "(env: MYAPP_MAX_RESULTS)")
    print("cache_ttl:", s.cache_ttl)
    print()
    print("config dir:", app_config_dir("myapp", create=False))
    print("data dir:", app_data_dir("myapp", create=False))
    print("cache dir:", app_cache_dir("myapp", create=False))


if __name__ == "__main__":
    main()
