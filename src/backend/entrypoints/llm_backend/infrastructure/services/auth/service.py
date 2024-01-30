import logging
from functools import lru_cache
from typing import Any, Dict, Mapping

import requests
from authlib.jose import JoseError, JsonWebKey, JsonWebToken

from market_alerts.config import KEYCLOAK_CLIENT_ID
from market_alerts.constants import AUTH_HEADER, JWT_PREFIX
from market_alerts.entrypoints.llm_backend.infrastructure.access_management.context_vars import (
    UserValueObject,
)
from market_alerts.infrastructure.mixins import SingletonMixin

from .exceptions import (
    EmptyAuthHeaderError,
    InvalidAuthHeaderFormatError,
    InvalidJWTError,
    JWKFetchError,
)

logger = logging.getLogger(__name__)


class AuthService(SingletonMixin):
    ADMIN_ROLE = "admin"

    def __init__(self, encryption_algorithm: str, certs_endpoint: str, verify_ssl: bool) -> None:
        self._encryption_algorithm = encryption_algorithm
        self._certs_endpoint = certs_endpoint
        self._verify_ssl = verify_ssl

    def authorize(self, request_headers: Mapping[str, Any]) -> UserValueObject:
        token = request_headers.get(AUTH_HEADER)

        if token is None:
            raise EmptyAuthHeaderError("No token provided inside Authorization header")

        if token.startswith(JWT_PREFIX):
            jwt = (lambda t: t.split(" ")[1])(token)
        else:
            raise InvalidAuthHeaderFormatError(f"Token format is invalid. Must start with {JWT_PREFIX}")

        decoded_token = self.decode_jwt(jwt)
        return UserValueObject(
            email=decoded_token["email"],
            token=jwt,
            is_admin=self.ADMIN_ROLE in decoded_token.get("resource_access", {}).get(KEYCLOAK_CLIENT_ID, {}).get("roles", []),
        )

    def decode_jwt(self, token: str) -> Dict[str, Any]:
        public_keys = self.fetch_jwks()
        try:
            token = JsonWebToken([self._encryption_algorithm]).decode(token, key=public_keys)
            token.validate()
        except JoseError as e:
            logger.debug(e)
            raise InvalidJWTError

        return token

    @lru_cache(maxsize=1)
    def fetch_jwks(self):
        try:
            res = requests.get(self._certs_endpoint, verify=self._verify_ssl)
            res.raise_for_status()
            keys = JsonWebKey.import_key_set(res.json()["keys"])
        except Exception as e:
            logger.error(e)
            raise JWKFetchError(f"Error while fetching public JWKS on '{self._certs_endpoint}'")

        return keys
