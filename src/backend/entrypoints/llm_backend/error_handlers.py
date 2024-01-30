import logging
import traceback
from http import HTTPStatus

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import ValidationError
from starlette import status

from market_alerts.domain.exceptions.base import (
    AlreadyExistsError,
    ForbiddenError,
    InputError,
    MarketAlertsError,
    NotFoundError,
)
from market_alerts.entrypoints.llm_backend.api.models.error import ErrorModel

logger = logging.getLogger(__name__)


def json_error_handler(error: MarketAlertsError, status_code: int):
    return JSONResponse(status_code=status_code, content=ErrorModel(code=error.code, message=str(error)).model_dump())


def handle_all_errors(req: Request, error: Exception):
    tb_str = "".join(traceback.format_exc())

    logger.error(f"Unhandled error {error}:\n{tb_str}")

    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content=ErrorModel(code="unhandled_error", message=str(error)).model_dump(),
    )


def register_error_handlers(app: FastAPI) -> None:
    @app.exception_handler(MarketAlertsError)
    def handle_mas_exception(req: Request, error: MarketAlertsError):
        mapper = [
            (ForbiddenError, HTTPStatus.FORBIDDEN),
            (NotFoundError, HTTPStatus.NOT_FOUND),
            (AlreadyExistsError, HTTPStatus.CONFLICT),
            (InputError, HTTPStatus.UNPROCESSABLE_ENTITY),
            (MarketAlertsError, HTTPStatus.BAD_REQUEST),
        ]

        for error_type, status_code in mapper:
            if issubclass(type(error), error_type):
                return json_error_handler(error, status_code)

    @app.exception_handler(ValidationError)
    def bad_request(req: Request, exc: ValidationError):
        return JSONResponse(
            status_code=HTTPStatus.BAD_REQUEST,
            content=ErrorModel(code="bad_request", message=str(exc)).model_dump(),
        )

    @app.exception_handler(RequestValidationError)
    def bad_request_(req: Request, exc: RequestValidationError):
        return JSONResponse(
            status_code=HTTPStatus.BAD_REQUEST,
            content=ErrorModel(code="bad_request", message=str(exc)).model_dump(),
        )

    app.add_exception_handler(Exception, handle_all_errors)
