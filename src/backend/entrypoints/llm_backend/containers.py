import optuna
import redis
from redis import asyncio as aioredis

from market_alerts.entrypoints.llm_backend import Settings
from market_alerts.entrypoints.llm_backend.celery_ import make_celery
from market_alerts.entrypoints.llm_backend.infrastructure.services.auth import (
    AuthService,
)
from market_alerts.entrypoints.llm_backend.infrastructure.session import SessionManager
from market_alerts.infrastructure.services.jupyter import JupyterService

settings = Settings()
auth_settings = settings.authentication

auth_service_singleton = AuthService.get_instance(
    auth_settings.encryption_algorithm, auth_settings.certs_endpoint, auth_settings.verify_ssl
)

celery_settings = settings.celery
celery_app = make_celery(
    celery_settings.broker_url,
    celery_settings.backend_url,
    celery_settings.task_acks_late,
    celery_settings.task_reject_on_worker_lost,
    celery_settings.task_track_started,
)

redis_settings = settings.redis
sync_redis_client = redis.Redis.from_url(redis_settings.url)
async_redis_client = aioredis.from_url(redis_settings.url)
session_manager_singleton = SessionManager.get_instance(
    sync_redis_client,
)

jupyterhub_settings = settings.jupyterhub
jupyter_service_singleton = JupyterService.get_instance(
    jupyterhub_settings.ui_url, jupyterhub_settings.service_url, jupyterhub_settings.token
)

optimization_storage = None


def get_optimization_storage(**kwargs):
    global optimization_storage
    if optimization_storage is None:
        optimization_storage = optuna.storages.RDBStorage(
            settings.optimization.storage_url,
            engine_kwargs=kwargs,
        )

    return optimization_storage
