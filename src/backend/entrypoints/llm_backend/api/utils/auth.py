from fastapi import Request

from market_alerts.entrypoints.llm_backend.infrastructure.session import Session


def extract_authorization_token(request: Request):
    return request.headers.get("Authorization").replace("Bearer ", "")


def get_session(request: Request) -> Session:
    session = request.state.client_session
    return session
