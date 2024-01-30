from typing import Any, List, Optional

from pydantic import Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class AuthenticationSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="AUTH_")

    enabled: bool = True
    verify_ssl: bool = True
    certs_endpoint: Optional[str] = None
    encryption_algorithm: str = "RS256"

    @model_validator(mode="before")
    @classmethod
    def check_auth_enabled_and_certs_endpoint_set(cls, data: Any) -> Any:
        if data.get("enabled") and (not data.get("certs_endpoint")):
            raise ValueError("Please provide OAUTH certificates endpoint via 'AUTH_CERTS_ENDPOINT'")
        return data


class CORSSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="CORS_")

    enabled: bool = True
    allow_origins: List[str] = ["*"]
    allow_headers: List[str] = ["*"]
    allow_methods: List[str] = ["*"]


class CelerySettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="CELERY_")

    broker_url: str
    backend_url: str
    concurrency: int = 1
    task_acks_late: bool = True
    task_reject_on_worker_lost: bool = True
    task_track_started: bool = True


class RedisSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="REDIS_")

    url: str


class JupyterhubSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="JUPYTERHUB_")

    ui_url: str
    service_url: str
    token: str


class OptimizationSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="OPTIMIZATION_")

    storage_url: str
    tasks_per_study: int


class Settings(BaseSettings):
    env: str = Field("prod", env="ENV")
    logger_level: str = Field("INFO", env="LOG_LEVEL")

    authentication: AuthenticationSettings = AuthenticationSettings()
    cors: CORSSettings = CORSSettings()

    limits_enabled: bool = Field(True, env="ALERTS_BACKEND_SERVICE_LIMITS_ENABLED")
    celery: CelerySettings = CelerySettings()

    redis: RedisSettings = RedisSettings()

    jupyterhub: JupyterhubSettings = JupyterhubSettings()

    optimization: OptimizationSettings = OptimizationSettings()

    documentation_enabled: bool = True
