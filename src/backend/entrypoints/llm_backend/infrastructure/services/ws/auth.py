import uuid
from datetime import datetime
from typing import Tuple

from market_alerts.entrypoints.llm_backend.containers import async_redis_client


async def generate_ws_auth_ticket(task_id: str, ttl: int = 60) -> Tuple[str, str]:
    ticket_id = str(uuid.uuid4())

    await async_redis_client.set(ticket_id, task_id, ex=ttl)

    return ticket_id, datetime.now().strftime("%Y-%m-%d %H:%M:%S")
