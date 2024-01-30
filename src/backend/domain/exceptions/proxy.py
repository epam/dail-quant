from .base import MarketAlertsError


class ServiceProxyError(MarketAlertsError):
    code = "service_proxy_error"

    def __init__(self, status_code: int, message: str, service_name: str = None):
        if service_name is not None:
            message = f"[{service_name}] Service failed with code {status_code}: {message}"

        super().__init__(message)
