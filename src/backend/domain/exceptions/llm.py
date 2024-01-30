import http

from .base import MarketAlertsError


class LLMBadResponseError(MarketAlertsError):
    code = "llm_bad_response_error"

    def __init__(self, detail: str, status_code: int = http.HTTPStatus.INTERNAL_SERVER_ERROR):
        super().__init__(detail)
        self.status_code = status_code
