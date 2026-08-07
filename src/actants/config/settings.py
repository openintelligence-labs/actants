from __future__ import annotations

from pathlib import Path
from typing import Any, Self

from pydantic_settings import BaseSettings, SettingsConfigDict


class AppSettings(BaseSettings):
    """Base class for app-level settings.

    Subclass and add your fields. Reads from environment variables, ``.env`` in CWD,
    and (optionally) ``settings.toml`` in the app config dir. Subclasses should set
    ``app_name`` and use it to derive paths.

    Example::

        class MySettings(AppSettings):
            model_config = AppSettings.config_for("deepdive")
            search_provider: str = "ddg"
            max_results: int = 10

        s = MySettings()  # loads from env + .env
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        env_nested_delimiter="__",
        extra="ignore",
    )

    @classmethod
    def config_for(cls, app_name: str, *, env_prefix: str | None = None) -> SettingsConfigDict:
        """Build a SettingsConfigDict scoped to a named app."""
        return SettingsConfigDict(
            env_file=".env",
            env_file_encoding="utf-8",
            env_prefix=env_prefix or f"{app_name.upper().replace('-', '_')}_",
            env_nested_delimiter="__",
            extra="ignore",
        )

    @classmethod
    def load(cls, *, env_file: str | Path | None = None, **overrides: Any) -> Self:
        """Load settings, optionally from a specific env file. Overrides win."""
        if env_file is not None:
            # pydantic-settings reads `_env_file` off the init kwargs at runtime, but
            # BaseSettings.__init__ is typed to accept only the model's own fields, so
            # there is no annotation under which this call type-checks.
            return cls(_env_file=str(env_file), **overrides)  # type: ignore[call-arg]
        return cls(**overrides)
