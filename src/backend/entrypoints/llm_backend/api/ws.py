from fastapi import APIRouter, Body, Query, WebSocket, status

from market_alerts.entrypoints.llm_backend.api.models.tasks import TaskIDRequestModel
from market_alerts.entrypoints.llm_backend.api.models.ws import (
    WSAuthTicketResponseModel,
)
from market_alerts.entrypoints.llm_backend.infrastructure.services.ws import (
    BaseTaskProgressHandler,
    OptimizationTaskProgressHandler,
    OptimizationTaskWSMetaInfo,
    PipelineWorkUnitBasedWSTaskMetaInfo,
    WorkUnitBasedProgressHandler,
    WorkUnitBasedTaskWSMetaInfo,
    consume_progress_updates,
    generate_ws_auth_ticket,
    get_task_ws_meta_info_async,
)

ws_router = APIRouter(prefix="/websockets")


@ws_router.post("/auth", tags=["Websockets"], response_model=WSAuthTicketResponseModel, status_code=201)
async def create_ws_auth_ticket(request_model: TaskIDRequestModel = Body(...)):
    ticket_id, creation_timestamp = await generate_ws_auth_ticket(
        task_id=request_model.task_id,
        ttl=60,
    )

    return WSAuthTicketResponseModel(
        ticket_id=ticket_id,
        task_id=request_model.task_id,
        creation_timestamp=creation_timestamp,
    )


@ws_router.websocket("/progress")
async def websocket_task_progress_updates(websocket: WebSocket, task_id: str = Query(...)):
    await websocket.accept()

    task_ids = None

    task_info = await get_task_ws_meta_info_async(task_id)
    if isinstance(task_info, OptimizationTaskWSMetaInfo):
        task_handler = OptimizationTaskProgressHandler(
            task_info.studies_names, task_info.work_units, task_info.start_time, websocket
        )
        task_ids = task_info.task_ids
    elif isinstance(task_info, PipelineWorkUnitBasedWSTaskMetaInfo):
        task_handler = WorkUnitBasedProgressHandler(task_info.work_units, task_info.start_time, websocket)
        task_ids = task_info.task_ids
    elif isinstance(task_info, WorkUnitBasedTaskWSMetaInfo):
        task_handler = WorkUnitBasedProgressHandler(task_info.work_units, task_info.start_time, websocket)
    else:
        task_handler = BaseTaskProgressHandler(task_info.start_time, websocket)

    try:
        await consume_progress_updates(websocket, task_id, task_handler, task_ids)
    except Exception:
        await websocket.close(code=status.WS_1011_INTERNAL_ERROR, reason="Internal error")
