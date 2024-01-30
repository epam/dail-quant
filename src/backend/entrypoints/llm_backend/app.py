import logging
import os
from typing import Awaitable, Callable, List

import uvicorn
from celery.schedules import crontab
from celery.signals import after_setup_logger
from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from uvicorn.config import LOGGING_CONFIG

from market_alerts.containers import alerts_backend_proxy_singleton
from market_alerts.domain.exceptions import AuthError
from market_alerts.domain.exceptions.base import MarketAlertsError
from market_alerts.entrypoints.llm_backend import api, constants, middleware
from market_alerts.entrypoints.llm_backend.config import Settings
from market_alerts.entrypoints.llm_backend.constants import (
    MARKET_ALERTS_NO_SESSION_ENDPOINTS,
    MARKET_ALERTS_PUBLIC_ENDPOINTS,
    MARKET_ALERTS_READ_ONLY_SESSION_ENDPOINTS,
)
from market_alerts.entrypoints.llm_backend.containers import (
    celery_app,
    session_manager_singleton,
)
from market_alerts.entrypoints.llm_backend.domain.exceptions import ResourcesLimitsError
from market_alerts.entrypoints.llm_backend.error_handlers import (
    handle_all_errors,
    json_error_handler,
    register_error_handlers,
)
from market_alerts.entrypoints.llm_backend.infrastructure.access_management.context_vars import (
    user,
)
from market_alerts.entrypoints.llm_backend.middleware import SessionMiddleware
from market_alerts.entrypoints.llm_backend.openapi import add_auth_to_openapi


def create_fastapi() -> FastAPI:
    config = Settings()

    fastapi_app = FastAPI(
        title=constants.PROJECT_NAME,
        version=constants.VERSION,
        docs_url=f"{constants.MARKET_ALERTS_API_PREFIX}{constants.SWAGGER_DOC_URL}",
        description=constants.DESCRIPTION,
        openapi_url=f"{constants.MARKET_ALERTS_API_PREFIX}/openapi.json",
    )
    fastapi_app.include_router(api.tickers_router, prefix=constants.MARKET_ALERTS_API_PREFIX)
    fastapi_app.include_router(api.llm_router, prefix=constants.MARKET_ALERTS_API_PREFIX)
    fastapi_app.include_router(api.plots_router, prefix=constants.MARKET_ALERTS_API_PREFIX)
    fastapi_app.include_router(api.jupyter_router, prefix=constants.MARKET_ALERTS_API_PREFIX)
    fastapi_app.include_router(api.session_router, prefix=constants.MARKET_ALERTS_API_PREFIX)
    fastapi_app.include_router(api.tasks_router, prefix=constants.MARKET_ALERTS_API_PREFIX)
    fastapi_app.include_router(api.ws_router, prefix=constants.MARKET_ALERTS_API_PREFIX)
    fastapi_app.include_router(api.files_router, prefix=constants.MARKET_ALERTS_API_PREFIX)
    fastapi_app.include_router(api.ui_router, prefix=constants.MARKET_ALERTS_API_PREFIX)
    fastapi_app.include_router(api.alerts_backend_router, prefix=constants.MARKET_ALERTS_API_PREFIX)
    fastapi_app.include_router(api.optimization_router, prefix=constants.MARKET_ALERTS_API_PREFIX)

    if config.authentication.enabled:
        if config.limits_enabled:
            alerts_backend_proxy_singleton.set_user_context(user)
            register_limits_middleware(fastapi_app)

        fastapi_app.add_middleware(
            SessionMiddleware,
            session_manager=session_manager_singleton,
            public_endpoints=MARKET_ALERTS_PUBLIC_ENDPOINTS + MARKET_ALERTS_NO_SESSION_ENDPOINTS,
            readonly_endpoints=MARKET_ALERTS_READ_ONLY_SESSION_ENDPOINTS,
        )
        register_auth_middleware(fastapi_app, MARKET_ALERTS_PUBLIC_ENDPOINTS)

    if config.cors.enabled:
        register_cors_middleware(fastapi_app, config)

    register_error_handlers(fastapi_app)

    return fastapi_app


def register_auth_middleware(app: FastAPI, public_endpoints: List[str]):
    add_auth_to_openapi(app)

    @app.middleware("http")
    async def handle_authorization(request: Request, call_next: Callable[[Request], Awaitable[Response]]) -> Response:
        try:
            middleware.set_user_from_jwt_middleware(request, public_endpoints)
        except AuthError as err:
            return json_error_handler(err, err.status_code)

        try:
            return await call_next(request)
        except MarketAlertsError as err:
            return json_error_handler(err, err.status_code)
        except Exception as err:
            return handle_all_errors(request, err)


def register_cors_middleware(app: FastAPI, config: Settings):
    app.add_middleware(
        CORSMiddleware,
        allow_origins=config.cors.allow_origins,
        allow_methods=config.cors.allow_methods,
        allow_headers=config.cors.allow_headers,
    )


def register_limits_middleware(app: FastAPI):
    @app.middleware("http")
    async def handle_limits(request: Request, call_next: Callable[[Request], Awaitable[Response]]) -> Response:
        try:
            middleware.check_limits_middleware(request.url.path)
        except ResourcesLimitsError as err:
            return json_error_handler(err, err.status_code)

        try:
            return await call_next(request)
        except MarketAlertsError as err:
            return json_error_handler(err, err.status_code)
        except Exception as err:
            return handle_all_errors(request, err)


def run_api():
    config = Settings()
    use_web_concurrency = "WEB_CONCURRENCY" in os.environ
    LOGGING_CONFIG["formatters"]["access"][
        "fmt"
    ] = '%(asctime)s - %(levelprefix)s %(client_addr)s - "%(request_line)s" %(status_code)s'

    options = {
        "host": "0.0.0.0",
        "port": 8000,
        "log_level": "debug",
        "workers": os.getenv("WEB_CONCURRENCY") if use_web_concurrency else 2,
        "reload": config.env == "development",
        "factory": True,
    }

    uvicorn.run("market_alerts.entrypoints.llm_backend.app:create_fastapi", **options)


def run_celery() -> None:
    config = Settings()

    if config.authentication.enabled:
        if config.limits_enabled:
            alerts_backend_proxy_singleton.set_user_context(user)

    celery_app.worker_main(
        ["worker", "-l", "info", "-n", "task-consumer", "-Q", "alerts_default", "-c", str(config.celery.concurrency)]
    )


def run_celery_beat() -> None:
    @after_setup_logger.connect
    def setup_task_logger(logger, **kwargs):
        for handler in logger.handlers:
            logger.removeHandler(handler)

        formatter = logging.Formatter("[%(asctime)s: %(levelname)s/%(processName)s] %(message)s")
        handler = logging.StreamHandler()
        handler.setLevel(logging.INFO)
        handler.setFormatter(formatter)

        logger.setLevel(logging.INFO)
        logger.addHandler(handler)

    celery_app.conf.update(
        beat_schedule={
            "clear-expired-sessions-every-hour": {
                "task": "clear_expired_sessions",
                "schedule": crontab(minute="*/30"),
                "options": {"queue": "alerts_default"},
            }
        }
    )
    celery_app.Beat().run()
