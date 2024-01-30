import http

from market_alerts.domain.exceptions.base import MarketAlertsError


class JupyterError(MarketAlertsError):
    code = "jupyter_error"

    def __init__(self, detail: str, status_code: int = http.HTTPStatus.INTERNAL_SERVER_ERROR):
        super().__init__(detail)
        self.status_code = status_code
