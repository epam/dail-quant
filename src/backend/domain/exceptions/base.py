import http


class MarketAlertsError(Exception):
    code = "market_alerts_error"

    def __init__(self, detail: str, status_code: int = http.HTTPStatus.INTERNAL_SERVER_ERROR):
        super().__init__(detail)
        self.status_code = status_code


class APILimitsError(MarketAlertsError):
    code = "api_limits_error"


class ForbiddenError(MarketAlertsError):
    code = "forbidden_error"


class NotFoundError(MarketAlertsError):
    code = "not_found_error"


class AlreadyExistsError(MarketAlertsError):
    code = "already_exists_error"


class InputError(MarketAlertsError):
    code = "input_error"


class NotImplementedError(MarketAlertsError):
    code = "not_implemented_error"
