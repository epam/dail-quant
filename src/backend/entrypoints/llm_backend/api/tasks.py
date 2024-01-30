import json

from billiard.exceptions import SoftTimeLimitExceeded
from celery.states import FAILURE, PENDING, SUCCESS
from fastapi import APIRouter, Depends

from market_alerts.domain.exceptions import DataNotFoundError, LLMBadResponseError
from market_alerts.domain.services.steps import delete_optimization_study
from market_alerts.entrypoints.llm_backend.api.utils import get_session
from market_alerts.entrypoints.llm_backend.containers import celery_app, settings
from market_alerts.entrypoints.llm_backend.infrastructure.services.ws import (
    OptimizationTaskWSMetaInfo,
    PipelineTaskWSMetaInfo,
    get_task_metadata,
    get_task_ws_meta_info,
)
from market_alerts.entrypoints.llm_backend.infrastructure.services.ws.task_meta_info import (
    save_task_ws_meta_info,
)
from market_alerts.entrypoints.llm_backend.infrastructure.session import Session

tasks_router = APIRouter(prefix="/tasks")


@tasks_router.get("/{task_id}", tags=["Tasks"], status_code=200)
async def get_task_status(task_id: str, session: Session = Depends(get_session)):
    task_meta = await get_task_metadata(task_id)

    final_result = {
        "task_id": task_id,
        "state": PENDING,
        "result": None,
    }

    if task_meta:
        task_meta = json.loads(task_meta)
        task_status = task_meta["status"]

        final_result.update(state=task_status)

        if task_status == SUCCESS:
            result = task_meta["result"]
        elif task_status == FAILURE:
            result = get_task_error_msg(task_meta["result"])
        else:
            # We are interested in statuses other than successful and failed only in the case
            # when we run our tasks in a chain
            result = session.flow_status.get_pipeline_status()
    else:
        result = session.flow_status.get_pipeline_status()

    final_result.update(result=result)

    return final_result


# TODO: handle only specific exceptions in prod
def get_task_error_msg(exc: Exception) -> str:
    if isinstance(exc, SoftTimeLimitExceeded):
        try:
            return exc.args[0]
        except IndexError:
            return "Timed out"
    return str(exc) if isinstance(exc, (DataNotFoundError, LLMBadResponseError)) else _get_error_msg_with_type(exc)


def _get_error_msg_with_type(exc: Exception) -> str:
    return f"{exc.__class__.__name__}: {exc}"


@tasks_router.delete("/{task_id}", tags=["Tasks"], status_code=200)
def revoke(task_id: str):
    task_info = get_task_ws_meta_info(task_id)

    if isinstance(task_info, OptimizationTaskWSMetaInfo):
        task_info.stop_flag = True
        save_task_ws_meta_info(task_id, **task_info.model_dump())

    if isinstance(task_info, PipelineTaskWSMetaInfo):
        for tid in task_info.task_ids:
            celery_app.control.revoke(tid)
    else:
        celery_app.control.revoke(task_id, terminate=True)
