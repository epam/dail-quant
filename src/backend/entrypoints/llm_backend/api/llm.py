import time
import uuid

from fastapi import APIRouter, Body, Depends

from market_alerts.domain.services import indicator_chat
from market_alerts.entrypoints.llm_backend.api.models.llm import (
    BacktestingRequestModel,
    LLMResponseModel,
    SubmitLLMQueryRequestModel,
)
from market_alerts.entrypoints.llm_backend.containers import (
    celery_app,
    session_manager_singleton,
)
from market_alerts.entrypoints.llm_backend.domain.exceptions import (
    DataNotFetchedError,
    IndicatorsBlockIsNotPresentError,
    LLMChatHistoryClearedError,
    LLMChatNotSubmittedError,
    TradingBlockIsNotPresentError,
)
from market_alerts.entrypoints.llm_backend.infrastructure.session import Session, Steps
from market_alerts.infrastructure.services.code import get_code_sections
from market_alerts.utils import time_profile

from ..infrastructure.access_management.context_vars import user
from ..infrastructure.services.ws.task_meta_info import save_task_ws_meta_info
from ..infrastructure.utils import create_task_headers
from .utils import get_session

llm_router = APIRouter(prefix="/llm")


@llm_router.post("/chat", response_model=LLMResponseModel, tags=["LLM"])
def submit_llm_chat(
    request_model: SubmitLLMQueryRequestModel = Body(...),
    session: Session = Depends(get_session),
):
    if not session.flow_status.is_step_done(Steps.FETCH_TICKERS):
        raise DataNotFetchedError

    session.setdefault("indicators_dialogue", []).append(request_model.llm_query)

    result, execution_time = time_profile(indicator_chat, session, request_model.engine)

    llm_response, token_usage, request_timestamp = result

    indicators_code, trading_code = get_code_sections(llm_response)

    session.flow_status.promote_submit_llm_chat_step(indicators_code, trading_code)

    return LLMResponseModel(
        flow_status=session.flow_status.to_dict(),
        llm_response=llm_response,
        engine=request_model.engine,
        token_usage=token_usage,
        request_timestamp=request_timestamp,
        execution_time=execution_time,
    )


@llm_router.post("/chat/v2", tags=["LLM"])
def submit_llm_chat_(
    request_model: SubmitLLMQueryRequestModel = Body(...),
    session: Session = Depends(get_session),
):
    if not session.flow_status.is_step_done(Steps.FETCH_TICKERS):
        raise DataNotFetchedError

    current_user = user.get()

    task = celery_app.signature(
        "submit_llm_chat",
        kwargs={
            "llm_query": request_model.llm_query,
            "user_prompt_ids": request_model.prompt_ids,
            "engine": request_model.engine,
        },
        queue="alerts_default",
        routing_key="alerts_default",
        headers=create_task_headers(session.session_id),
    )

    task_id = str(uuid.uuid4())
    task_scheduled_timestamp = time.time()

    session.add_action(task_id, task_scheduled_timestamp)
    session_manager_singleton.save(current_user.email, session)

    task.apply_async(soft_time_limit=600, time_limit=620, task_id=task_id)
    return {
        "task_id": task_id,
    }


@llm_router.post("/calculate_indicators", tags=["LLM"])
def calculate_indicators(session: Session = Depends(get_session)):
    if not session.flow_status.is_step_done(Steps.SUBMIT_LLM_CHAT):
        raise LLMChatNotSubmittedError
    if session.flow_status.is_llm_chat_history_cleared:
        raise LLMChatHistoryClearedError
    if not session.flow_status.is_indicators_block_present:
        raise IndicatorsBlockIsNotPresentError

    current_user = user.get()

    task = celery_app.signature(
        "calculate_indicators",
        queue="alerts_default",
        routing_key="alerts_default",
        headers=create_task_headers(session.session_id),
    )

    task_id = str(uuid.uuid4())
    task_scheduled_timestamp = time.time()

    session.add_action(task_id, task_scheduled_timestamp)
    session_manager_singleton.save(current_user.email, session)

    soft_time_limit = 10 * 60
    task.apply_async(soft_time_limit=soft_time_limit, time_limit=soft_time_limit + 20, task_id=task_id)
    return {
        "task_id": task_id,
    }


@llm_router.post("/calculate_backtesting", tags=["LLM"])
def calculate_backtesting(request_model: BacktestingRequestModel = Body(), session: Session = Depends(get_session)):
    if not session.flow_status.is_step_done(Steps.SUBMIT_LLM_CHAT):
        raise LLMChatNotSubmittedError
    if session.flow_status.is_llm_chat_history_cleared:
        raise LLMChatHistoryClearedError
    if not session.flow_status.is_trading_block_present:
        raise TradingBlockIsNotPresentError

    current_user = user.get()

    task = celery_app.signature(
        "calculate_backtesting",
        kwargs={
            "actual_currency": request_model.actual_currency,
            "bet_size": request_model.bet_size,
            "per_instrument_gross_limit": request_model.per_instrument_gross_limit,
            "total_gross_limit": request_model.total_gross_limit,
            "nop_limit": request_model.nop_limit,
            "account_for_dividends": request_model.account_for_dividends,
            "trade_fill_price": request_model.trade_fill_price,
            "execution_cost_bps": request_model.execution_cost_bps,
        },
        queue="alerts_default",
        routing_key="alerts_default",
        headers=create_task_headers(session.session_id),
    )

    task_id = str(uuid.uuid4())
    task_scheduled_timestamp = time.time()

    save_task_ws_meta_info(
        task_id,
        start_time=time.time(),
    )

    session.add_action(task_id, task_scheduled_timestamp)
    session_manager_singleton.save(current_user.email, session)

    soft_time_limit = 10 * 60
    task.apply_async(soft_time_limit=soft_time_limit, time_limit=soft_time_limit + 20, task_id=task_id)

    return {"task_id": task_id}
