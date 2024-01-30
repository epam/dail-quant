from http import HTTPStatus
from typing import List

from fastapi import Request

from market_alerts.domain.exceptions import AuthError
from market_alerts.entrypoints.llm_backend.containers import auth_service_singleton
from market_alerts.entrypoints.llm_backend.infrastructure.access_management.context_vars import (
    user,
)
from market_alerts.entrypoints.llm_backend.infrastructure.services.auth import (
    EmptyAuthHeaderError,
    InvalidAuthHeaderFormatError,
    InvalidJWTError,
    JWKFetchError,
)


def set_user_from_jwt_middleware(
    request: Request,
    public_endpoints: List[str],
) -> None:
    if request.url.path in public_endpoints:
        return None

    try:
        user_credentials = auth_service_singleton.authorize(request.headers)
        user.set(user_credentials)
    except (EmptyAuthHeaderError, InvalidAuthHeaderFormatError) as e:
        raise AuthError(
            detail=str(e),
            status_code=HTTPStatus.UNAUTHORIZED,
        )
    except InvalidJWTError:
        raise AuthError(
            detail="Invalid JWT",
            status_code=HTTPStatus.UNAUTHORIZED,
        )
    except JWKFetchError as e:
        raise AuthError(
            detail=str(e),
            status_code=HTTPStatus.INTERNAL_SERVER_ERROR,
        )
    except Exception as e:
        raise AuthError(
            detail=str(e),
            status_code=HTTPStatus.INTERNAL_SERVER_ERROR,
        )
