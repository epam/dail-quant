import http

from market_alerts.domain.exceptions.base import MarketAlertsError


class ResourcesLimitsError(MarketAlertsError):
    code = "resources_limits_error"

    def __init__(self, detail: str, status_code: int = http.HTTPStatus.TOO_MANY_REQUESTS):
        super().__init__(detail)
        self.status_code = status_code
