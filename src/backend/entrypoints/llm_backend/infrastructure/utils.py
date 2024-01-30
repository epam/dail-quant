from dataclasses import asdict
from typing import Any, Dict

from market_alerts.entrypoints.llm_backend.infrastructure.access_management.context_vars import (
    user,
)


def create_current_user_header() -> Dict[str, Dict[str, Any]]:
    return {"current_user": asdict(user.get())}


def create_session_id_header(session_id: str) -> Dict[str, str]:
    return {"sessionId": session_id}


def create_task_headers(session_id: str) -> Dict[str, Any]:
    return {**create_session_id_header(session_id), **create_current_user_header()}
