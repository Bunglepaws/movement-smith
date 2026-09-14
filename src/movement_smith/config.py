from __future__ import annotations

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="MOVEMENT_SMITH_", env_file=".env", extra="ignore")

    worker_url: str = ""
    worker_token: str = ""
    modal_key: str = ""
    modal_secret: str = ""
    allow_stub: bool = False
    data_dir: Path = Path("data")
    cors_origins: str = "http://localhost:5173,http://127.0.0.1:5173"


def get_settings() -> Settings:
    return Settings()
