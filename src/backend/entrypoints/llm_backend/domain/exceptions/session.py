import http

from market_alerts.domain.exceptions.base import (
    ForbiddenError,
    MarketAlertsError,
    NotFoundError,
)


class XTabSessionIDHeaderNotSetError(MarketAlertsError):
    code = "x_tab_session_id_header_not_set_error"

    def __init__(self, detail: str = "X-Tab-Session-ID header not set", status_code: int = http.HTTPStatus.BAD_REQUEST):
        super().__init__(detail)
        self.status_code = status_code


class SessionExpiredError(ForbiddenError):
    code = "session_expired_error"


class SessionDoesNotExist(NotFoundError):
    code = "session_does_not_exist_error"
