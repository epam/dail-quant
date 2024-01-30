from contextvars import ContextVar
from typing import Any, Dict, Optional

from requests import Session

from market_alerts.constants import AUTH_HEADER, JWT_PREFIX
from market_alerts.domain.exceptions import AuthError


class AuthDriverMixin:
    def __init__(
        self,
    ):
        self._session = Session()
        self._user_context: Optional[ContextVar] = None
        self._auth_header: Optional[Dict[str, str]] = None
        self._email: Optional[str] = None

    def set_user_context(self, user_context: ContextVar) -> None:
        self._user_context = user_context

    def set_auth_header(self, headers: Dict[str, str]) -> None:
        self._auth_header = headers

    def set_email(self, email: str) -> None:
        self._email = email

    @property
    def email(self) -> str:
        if self._user_context is not None:
            return self._user_context.get().email
        if not self._email:
            raise ValueError(f"Email field was not set in {self.__class__}")
        return self._email

    @property
    def session(self) -> Session:
        header = self._get_auth_header()
        if header:
            self._set_auth_header(header)
        return self._session

    @session.setter
    def session(self, session: Session) -> None:
        self._session = session

    def _get_auth_header(self) -> Optional[Dict[str, str]]:
        if self._auth_header is not None:
            return self._auth_header
        user_context = self._get_user_context()
        if user_context:
            token = getattr(user_context, "token", "")
            if token:
                return self._create_auth_token_header(AUTH_HEADER, JWT_PREFIX, token)
            else:
                raise AuthError("Authorization token of current user is empty")
        return None

    def _get_user_context(self) -> Optional[Dict[str, Any]]:
        return self._user_context.get() if self._user_context and self._user_context.get(None) else None

    def _set_auth_header(self, headers: Dict[str, str]) -> None:
        self._session.headers.update(headers)

    @staticmethod
    def _create_auth_token_header(header: str, prefix: str, token: str) -> Dict[str, str]:
        return {header: f"{prefix}{token}"}


class SingletonMixin:
    _instance = None

    @classmethod
    def get_instance(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = cls(*args, **kwargs)
        return cls._instance
