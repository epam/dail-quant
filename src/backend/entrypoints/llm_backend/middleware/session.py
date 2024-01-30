import re
from typing import List

from fastapi import FastAPI
from starlette.middleware.base import BaseHTTPMiddleware

from market_alerts.domain.exceptions.base import MarketAlertsError
from market_alerts.entrypoints.llm_backend.domain.exceptions import (
    XTabSessionIDHeaderNotSetError,
)
from market_alerts.entrypoints.llm_backend.error_handlers import (
    handle_all_errors,
    json_error_handler,
)
from market_alerts.entrypoints.llm_backend.infrastructure.access_management.context_vars import (
    user,
)
from market_alerts.entrypoints.llm_backend.infrastructure.session import SessionManager


class SessionMiddleware(BaseHTTPMiddleware):
    def __init__(
        self, app: FastAPI, session_manager: SessionManager, public_endpoints: List[str], readonly_endpoints: List[str], **kwargs
    ) -> None:
        self._public_endpoints = public_endpoints
        self._readonly_endpoints = readonly_endpoints
        self._session_manager = session_manager
        super().__init__(app, **kwargs)

    async def dispatch(self, request, call_next):
        for pattern in self._public_endpoints:
            if re.match(pattern, request.url.path):
                return await call_next(request)

        current_user = user.get()
        session_id = request.headers.get("X-Tab-Session-ID")

        try:
            if not session_id:
                raise XTabSessionIDHeaderNotSetError

            request.state.client_session = self._session_manager.get(current_user.email, session_id)
            response = await call_next(request)

            if not any(re.match(pattern, request.url.path) for pattern in self._readonly_endpoints):
                if response.status_code < 400:
                    self._session_manager.save(current_user.email, request.state.client_session)

        except MarketAlertsError as err:
            return json_error_handler(err, err.status_code)
        except Exception as err:
            return handle_all_errors(request, err)

        return response
