from datetime import datetime
from typing import List

from fastapi import APIRouter

from market_alerts.entrypoints.llm_backend.api.models.common import (
    FlowStatusResponseModel,
)
from market_alerts.entrypoints.llm_backend.api.models.session import (
    DefaultSession,
    SessionInfoResponseModel,
    SessionResponseModel,
    UpdateCodeRequestModel,
)
from market_alerts.entrypoints.llm_backend.containers import (
    celery_app,
    session_manager_singleton,
)
from market_alerts.entrypoints.llm_backend.domain.exceptions import (
    BacktestingNotPerformedError,
    IndicatorsNotGeneratedError,
    LLMChatNotSubmittedError,
)
from market_alerts.entrypoints.llm_backend.infrastructure.access_management.context_vars import (
    user,
)
from market_alerts.entrypoints.llm_backend.infrastructure.session import Steps
from market_alerts.infrastructure.services.code import update_code_sections

session_router = APIRouter(prefix="/session")


@session_router.post("", tags=["Session"], response_model=SessionResponseModel, status_code=201)
def create_session():
    current_user = user.get()

    session = session_manager_singleton.create(current_user.email, data=DefaultSession().model_dump())

    return SessionResponseModel(sessionId=session.session_id, expires_in=session.expires_in)


@session_router.get("/{session_id}", response_model=SessionInfoResponseModel, tags=["Session"], status_code=200)
def get_data(session_id: str):
    current_user = user.get()

    session = session_manager_singleton.get(current_user.email, session_id)

    return SessionInfoResponseModel(
        sessionId=session_id,
        flow_status=session.flow_status.to_dict(),
        datasets=session.get("datasets_keys"),
        time_range=session.get("time_period"),
        periodicity=session.get("interval"),
        account_for_dividends=session.get("use_dividends_trading"),
        trade_fill_price=session.get("fill_trade_price"),
        **session.data,
    )


@session_router.put("/{session_id}", tags=["Session"], response_model=SessionResponseModel, status_code=200)
def prolong_session(session_id: str):
    current_user = user.get()

    session = session_manager_singleton.prolong(current_user.email, session_id)

    return SessionResponseModel(sessionId=session.session_id, expires_in=session.expires_in)


@session_router.patch("/{session_id}/clear_dialogue", tags=["Session"], response_model=SessionResponseModel, status_code=200)
def clear_llm_dialogue(session_id: str):
    current_user = user.get()

    session = session_manager_singleton.get(current_user.email, session_id)

    session.data["indicators_dialogue"] = []

    session.flow_status.set_llm_chat_history_cleared()

    session_manager_singleton.save(current_user.email, session)

    return SessionResponseModel(sessionId=session.session_id, expires_in=session.expires_in)


@session_router.put("/{session_id}/update_code", tags=["Session"], response_model=FlowStatusResponseModel, status_code=200)
def update_code(session_id: str, request_model: UpdateCodeRequestModel):
    current_user = user.get()

    session = session_manager_singleton.get(current_user.email, session_id)

    if not session.flow_status.is_step_done(Steps.SUBMIT_LLM_CHAT):
        raise LLMChatNotSubmittedError

    session["indicators_dialogue"][-1] = update_code_sections(
        session["indicators_dialogue"][-1], request_model.indicators_code, request_model.trading_code
    )

    session.flow_status.promote_submit_llm_chat_step(request_model.indicators_code, request_model.trading_code)

    session_manager_singleton.save(current_user.email, session)

    return FlowStatusResponseModel(flow_status=session.flow_status.to_dict())


@session_router.get("/{session_id}/indicators_logs", response_model=List[str], tags=["Session"], status_code=200)
def get_indicators_logs(session_id: str):
    current_user = user.get()

    session = session_manager_singleton.get(current_user.email, session_id)

    if not session.flow_status.is_step_done(Steps.CALCULATE_INDICATORS):
        raise IndicatorsNotGeneratedError

    return session["indicators_code_log"]


@session_router.get("/{session_id}/trading_logs", response_model=List[str], tags=["Session"], status_code=200)
def get_trading_logs(session_id: str):
    current_user = user.get()

    session = session_manager_singleton.get(current_user.email, session_id)

    if not session.flow_status.is_step_done(Steps.PERFORM_BACKTESTING):
        raise BacktestingNotPerformedError

    return session["trading_code_log"]


@session_router.put("/{session_id}/reset", response_model=SessionInfoResponseModel, tags=["Session"], status_code=200)
def reset_session(session_id: str):
    current_user = user.get()

    session = session_manager_singleton.get(current_user.email, session_id)

    session.reset_flow_status()
    session.data = DefaultSession().model_dump()

    session_manager_singleton.save(current_user.email, session)

    return SessionInfoResponseModel(
        sessionId=session_id,
        flow_status=session.flow_status.to_dict(),
        datasets=session.get("datasets_keys"),
        time_range=session.get("time_period"),
        periodicity=session.get("interval"),
        account_for_dividends=session.get("use_dividends_trading"),
        trade_fill_price=session.get("fill_trade_price"),
        **session.data,
    )


@session_router.delete("/{session_id}", tags=["Session"], status_code=204)
def delete_session(session_id: str):
    current_user = user.get()

    session_manager_singleton.delete(current_user.email, session_id)


@session_router.get("/action_history/{session_id}", tags=["Session"], status_code=200)
def get_action_history(session_id: str):
    current_user = user.get()

    session = session_manager_singleton.get(current_user.email, session_id)

    actions = []

    for task_id, timestamp in session.actions_history:
        task = celery_app.AsyncResult(task_id)

        args_str = ", ".join([str(arg) for arg in task.args])

        kwargs_str = ", ".join([f'{k}="{v}"' if isinstance(v, str) else f"{k}={v}" for k, v in task.kwargs.items()])

        all_args_str = ", ".join(filter(None, [args_str, kwargs_str]))
        action = f"{task.name}({all_args_str})"
        actions.append(
            dict(
                start_date=datetime.fromtimestamp(timestamp).strftime("%Y-%m-%dT%H:%M:%S.%f"),
                end_date__=task.date_done,
                action=action,
                result=str(task.result) if isinstance(task.result, Exception) else task.result,
            )
        )
        if task.traceback:
            actions[-1].update(traceback=task.traceback)

    return actions
