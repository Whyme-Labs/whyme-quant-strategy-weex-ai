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
    openrouter_api_key: str = ""
    openrouter_api_url: str = "https://openrouter.ai/api/v1/chat/completions"
    llm_model: str = "xiaomi/mimo-v2-flash:free"  # MiMo-v2-flash #1 Finance on OpenRouter
    llm_http_referer: str = "https://whymelabs.com"
    llm_app_name: str = "WhyMe Quant Strategy Engine"

    # Discord
    discord_webhook_url: str = ""
    discord_bot_name: str = "WhyMe Quant Bot"
    discord_bot_avatar: str = ""  # Optional: URL to bot avatar image

    # Trading Loop
    main_loop_interval: int = 5  # seconds between market data polls
    error_backoff_interval: int = 10  # seconds to wait after error

    # Logging
    log_level: str = "INFO"

    # Strategy Configuration
    max_position_size: float = 1000.0
    max_leverage: int = 20
    default_symbol: str = "BTCUSDT"
    # Trading symbols - multiple pairs for more opportunities
    trading_symbols: str = "BTCUSDT,ETHUSDT,SOLUSDT,BNBUSDT,XRPUSDT"  # Comma-separated

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
