import http

from market_alerts.domain.exceptions.base import MarketAlertsError


class AuthError(MarketAlertsError):
    code = "authentication_error"

    def __init__(self, detail: str, status_code: int = http.HTTPStatus.UNAUTHORIZED):
        super().__init__(detail)
        self.status_code = status_code
