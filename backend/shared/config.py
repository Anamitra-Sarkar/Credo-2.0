"""
Shared configuration module for VeriFy AI backend services.
Loads environment variables and provides typed configuration objects.
"""
from functools import lru_cache
from typing import List
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
        protected_namespaces=()
    )

    # Environment
    environment: str = "development"
    debug: bool = False

    # API Configuration
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    api_version: str = "v1"
    cors_origins: List[str] = ["http://localhost:3000"]

    # Security
    jwt_secret_key: str = "dev-secret-key-change-in-production-min-32-characters-long"
    jwt_algorithm: str = "HS256"
    jwt_access_token_expire_minutes: int = 30
    jwt_refresh_token_expire_days: int = 7

    # Database
    database_url: str = "sqlite+aiosqlite:///./verifyai.db"
    database_pool_size: int = 20
    database_max_overflow: int = 40

    # Redis
    redis_url: str = "redis://localhost:6379/0"
    redis_max_connections: int = 50
    cache_ttl_seconds: int = 3600

    # Google Cloud
    gcp_project_id: str = "verifyai-dev"
    gcp_region: str = "us-central1"
    gcs_bucket: str = "verifyai-storage"
    pubsub_video_topic: str = "video-processing"
    pubsub_video_subscription: str = "video-processing-sub"

    # AI APIs
    gemini_api_key: str = "mock-gemini-key"
    gemini_model: str = "gemini-2.0-flash"
    tavily_api_key: str = "mock-tavily-key"

    # Model Service URLs
    model_text_liar_url: str = "http://localhost:8001"
    model_text_brain2_url: str = "http://localhost:8002"
    model_image_url: str = "http://localhost:8003"
    model_video_url: str = "http://localhost:8004"
    model_voice_url: str = "http://localhost:8005"

    # Rate Limiting
    rate_limit_per_minute: int = 60
    rate_limit_per_hour: int = 1000
    rate_limit_per_day: int = 10000

    # File Limits
    max_text_length: int = 50000
    max_image_size_mb: int = 10
    max_video_size_mb: int = 100
    max_audio_size_mb: int = 20

    # Monitoring
    enable_metrics: bool = True
    enable_tracing: bool = True
    log_level: str = "INFO"

    # Translation
    default_language: str = "en"
    supported_languages: List[str] = [
        "en", "hi", "bn", "ta", "te", "mr", "gu", "kn", "ml", "pa", "or", "as"
    ]

    # Trending
    trending_time_window_hours: int = 24
    trending_min_reports: int = 5
    trending_update_interval_minutes: int = 15

    # Worker
    video_worker_concurrency: int = 4
    video_worker_timeout_seconds: int = 600
    video_worker_retry_attempts: int = 3

    # Feature Flags
    enable_crowdsourced_reports: bool = True
    enable_geographic_trending: bool = True
    enable_real_time_search: bool = True
    enable_explanation_generation: bool = True

    @property
    def is_production(self) -> bool:
        return self.environment == "production"

    @property
    def max_image_size_bytes(self) -> int:
        return self.max_image_size_mb * 1024 * 1024

    @property
    def max_video_size_bytes(self) -> int:
        return self.max_video_size_mb * 1024 * 1024

    @property
    def max_audio_size_bytes(self) -> int:
        return self.max_audio_size_mb * 1024 * 1024


@lru_cache()
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()


# Global settings instance
settings = get_settings()
