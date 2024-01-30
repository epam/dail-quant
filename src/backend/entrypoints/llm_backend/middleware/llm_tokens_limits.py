from http import HTTPStatus
from typing import Optional

from market_alerts.containers import alerts_backend_proxy_singleton
from market_alerts.entrypoints.llm_backend.domain.exceptions import ResourcesLimitsError
from market_alerts.infrastructure.services.proxy.alerts_backend import (
    LimitsExceededError,
    LimitsInfoFetchError,
)

PROTECTED_LLM_ENDPOINTS = ("/api/market_alerts/v1/llm/chat/v2",)

PROTECTED_CPU_ENDPOINTS = (
    "/api/market_alerts/v1/llm/calculate_indicators",
    "/api/market_alerts/v1/llm/calculate_backtesting",
    "/api/market_alerts/v1/optimization/run",
)


def check_limits_middleware(
    url_path: Optional[str] = None,
) -> None:
    if url_path is not None and url_path not in PROTECTED_LLM_ENDPOINTS and url_path not in PROTECTED_CPU_ENDPOINTS:
        return

    try:
        if url_path in PROTECTED_LLM_ENDPOINTS:
            alerts_backend_proxy_singleton.check_token_user_limits()
        if url_path is None or url_path in PROTECTED_CPU_ENDPOINTS:
            alerts_backend_proxy_singleton.check_cpu_user_limits()
    except LimitsExceededError as e:
        raise ResourcesLimitsError(
            detail=str(e),
            status_code=HTTPStatus.TOO_MANY_REQUESTS,
        )
    except LimitsInfoFetchError:
        raise ResourcesLimitsError(
            detail="Couldn't fetch limits",
            status_code=HTTPStatus.INTERNAL_SERVER_ERROR,
        )
    except Exception as e:
        raise ResourcesLimitsError(
            detail=str(e),
            status_code=HTTPStatus.INTERNAL_SERVER_ERROR,
        )
