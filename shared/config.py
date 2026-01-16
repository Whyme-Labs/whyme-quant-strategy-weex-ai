"""Configuration management."""

from functools import lru_cache
from typing import Optional
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings loaded from environment."""

    # WEEX API
    weex_api_key: str = ""
    weex_secret_key: str = ""
    weex_passphrase: str = ""
    weex_api_url: str = "https://api-contract.weex.com"

    # WhyMe Quant Main Server (optional)
    main_server_url: str = "https://whyme-quant.swmengappdev.workers.dev"
    node_api_key: str = ""

    # Redis
    redis_url: str = "redis://localhost:6379"

    # AI Models
    openai_api_key: str = ""
    anthropic_api_key: str = ""

    # Logging
    log_level: str = "INFO"

    # Strategy Configuration
    max_position_size: float = 1000.0
    max_leverage: int = 20
    default_symbol: str = "BTCUSDT"

    # Heartbeat
    heartbeat_interval: int = 30

    # AI Log Upload
    ai_log_upload_interval: int = 30
    ai_log_batch_size: int = 10

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


@lru_cache()
def get_settings() -> Settings:
    """Get cached settings instance.

    Returns:
        Settings singleton
    """
    return Settings()
