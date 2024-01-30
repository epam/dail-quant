import asyncio
import json
import logging
from typing import Optional

from fastapi import WebSocket, status
from redis import asyncio as aioredis
from starlette.websockets import WebSocketDisconnect
from websockets.exceptions import ConnectionClosedError

from market_alerts.domain.constants import PUBSUB_END_OF_DATA
from market_alerts.entrypoints.llm_backend.containers import async_redis_client

from .handlers import BaseTaskProgressHandler
from .task_meta_info import get_task_metadata

logger = logging.getLogger(__name__)


async def consume_progress_updates(
    websocket: WebSocket,
    task_id: str,
    task_progress_handler: BaseTaskProgressHandler,
    task_ids: Optional[list[str]] = None,
):
    try:
        async with async_redis_client.pubsub() as pubsub:
            await pubsub.subscribe(f"task-{task_id}-progress")

            loop = asyncio.get_running_loop()
            websocket_alive_checker = loop.create_task(
                _is_websocket_alive_checker(websocket),
                name=f"WS alive checker: {websocket.client}",
            )
            consumer = loop.create_task(
                _consume_data(pubsub, websocket, task_progress_handler, no_data_timeout=10 * 60),
                name=f"WS data consumer: {websocket.client}",
            )

            stop_tasks_callback = lambda tasks: [task.cancel() for task in tasks]
            websocket_alive_checker.add_done_callback(lambda f: stop_tasks_callback([consumer] + task_status_checkers))
            consumer.add_done_callback(lambda f: stop_tasks_callback([websocket_alive_checker] + task_status_checkers))

            task_status_checkers = []
            tasks_to_check = task_ids if task_ids is not None else [task_id]
            for idx, t_id in enumerate(tasks_to_check):
                is_last_task = idx == len(tasks_to_check) - 1
                task_status_checker = loop.create_task(
                    _is_celery_task_finished_checker(t_id, is_last_task, websocket),
                    name=f"Task finished checker: {t_id}",
                )
                task_status_checker.add_done_callback(lambda f: stop_tasks_callback([websocket_alive_checker, consumer]))
                task_status_checkers.append(task_status_checker)

            await asyncio.wait({websocket_alive_checker, consumer, *task_status_checkers})
            logger.info("Finished consuming progress data")

    except Exception as e:
        logger.error(e)
        raise
    finally:
        await pubsub.unsubscribe(f"task-{task_id}-progress")


async def _is_websocket_alive_checker(websocket: WebSocket):
    try:
        await websocket.receive_text()
    except (WebSocketDisconnect, ConnectionClosedError):
        logger.info("[alive_checker] Client %s:%s closed websocket connection", websocket.client.host, websocket.client.port)


async def _is_celery_task_finished_checker(t_id: str, is_last_task: bool, websocket: WebSocket):
    while True:
        logger.debug("[task_finished_checker] Checking if task %s is finished...", t_id)
        task_finished_status = await _is_celery_task_finished(t_id, check_success=is_last_task)
        if task_finished_status:
            logger.debug("[task_finished_checker] Task %s is finished", t_id)
            await websocket.close(code=status.WS_1000_NORMAL_CLOSURE, reason="Task completed")
            logger.info(
                "[task_finished_checker] Client %s:%s closed websocket connection", websocket.client.host, websocket.client.port
            )
            break
        await asyncio.sleep(1)


async def _is_celery_task_finished(task_id: str, check_success: bool = False) -> bool:
    task_meta = await get_task_metadata(task_id)
    if task_meta:
        task_meta = json.loads(task_meta)
        check_statuses = ("SUCCESS", "FAILURE", "REVOKED") if check_success else ("FAILURE", "REVOKED")
        return task_meta["status"] in check_statuses
    return False


async def _consume_data(
    pubsub: aioredis.client.PubSub,
    websocket: WebSocket,
    task_progress_handler: BaseTaskProgressHandler,
    no_data_timeout: int = 3,
):
    try:
        await task_progress_handler.pre_process()

        while True:
            try:
                message = await asyncio.wait_for(
                    pubsub.get_message(ignore_subscribe_messages=True, timeout=no_data_timeout + 1), timeout=no_data_timeout
                )
            except asyncio.TimeoutError:
                logger.info(
                    "No data from Redis channel after timeout of %s sec, closing websocket connection...", no_data_timeout
                )
                await websocket.close(code=status.WS_1000_NORMAL_CLOSURE, reason="Server closed connection due to end of data")
                return

            if message is not None and message["type"] == "message":
                decoded_message = message["data"].decode("utf-8")
                message_data = json.loads(decoded_message)

                if message_data.get("progress") == PUBSUB_END_OF_DATA:
                    logger.info("End of data, closing websocket connection...")
                    await websocket.close(
                        code=status.WS_1000_NORMAL_CLOSURE, reason="Server closed connection due to end of data"
                    )
                    return

                await task_progress_handler.post_process(message_data)
    except (WebSocketDisconnect, ConnectionClosedError):
        logger.info("[consume_data] Client %s:%s closed websocket connection", websocket.client.host, websocket.client.port)
    except aioredis.RedisError as e:
        await websocket.close(code=status.WS_1011_INTERNAL_ERROR, reason="Internal server error")
        logger.error(e)
