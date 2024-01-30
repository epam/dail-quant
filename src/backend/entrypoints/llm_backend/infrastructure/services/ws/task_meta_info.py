import json
from typing import Optional, Type, Union

from pydantic import BaseModel, ValidationError

from market_alerts.entrypoints.llm_backend.containers import (
    async_redis_client,
    sync_redis_client,
)


class BaseTaskWSMetaInfo(BaseModel):
    start_time: float


class WorkUnitBasedTaskWSMetaInfo(BaseTaskWSMetaInfo):
    work_units: int


class PipelineTaskWSMetaInfo(BaseTaskWSMetaInfo):
    task_ids: list[str]


class PipelineWorkUnitBasedWSTaskMetaInfo(PipelineTaskWSMetaInfo, WorkUnitBasedTaskWSMetaInfo):
    pass


class OptimizationTaskWSMetaInfo(PipelineWorkUnitBasedWSTaskMetaInfo):
    studies_names: list[str]
    stop_flag: bool = False


def save_task_ws_meta_info(task_id: str, ttl: int = 3600, **info) -> None:
    sync_redis_client.set(f"celery-task-meta-{task_id}-ws", json.dumps(info), ex=ttl)


def _try_to_load_as_types(json_data: str, types: tuple[Type[BaseModel], ...]) -> Optional[BaseModel]:
    for t in types:
        try:
            return t.model_validate_json(json_data)
        except ValidationError:
            pass


task_ws_meta_info_resolution_order = (
    OptimizationTaskWSMetaInfo,
    PipelineWorkUnitBasedWSTaskMetaInfo,
    PipelineTaskWSMetaInfo,
    WorkUnitBasedTaskWSMetaInfo,
    BaseTaskWSMetaInfo,
)


def get_task_ws_meta_info(
    task_id: str,
) -> Union[BaseTaskWSMetaInfo, WorkUnitBasedTaskWSMetaInfo, PipelineTaskWSMetaInfo, OptimizationTaskWSMetaInfo, None]:
    task_info_json = sync_redis_client.get(f"celery-task-meta-{task_id}-ws")
    if task_info_json is not None:
        return _try_to_load_as_types(task_info_json, task_ws_meta_info_resolution_order)


async def get_task_ws_meta_info_async(
    task_id: str,
) -> Union[BaseTaskWSMetaInfo, WorkUnitBasedTaskWSMetaInfo, PipelineTaskWSMetaInfo, OptimizationTaskWSMetaInfo, None]:
    task_info_json = await async_redis_client.get(f"celery-task-meta-{task_id}-ws")
    if task_info_json is not None:
        return _try_to_load_as_types(task_info_json, task_ws_meta_info_resolution_order)


async def get_task_metadata(task_id: str) -> Optional[bytes]:
    return await async_redis_client.get(f"celery-task-meta-{task_id}")
