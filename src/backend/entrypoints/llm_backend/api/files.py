import json
import logging
import time
from urllib.parse import quote

from fastapi import APIRouter, Depends, File, UploadFile
from fastapi.responses import Response

from market_alerts.domain.exceptions import FileImportError
from market_alerts.domain.services import (
    create_pnl_report,
    create_stats_by_symbol_report,
)
from market_alerts.entrypoints.llm_backend.api.models.common import StrategyTitleModel
from market_alerts.entrypoints.llm_backend.api.models.session import DefaultSession
from market_alerts.entrypoints.llm_backend.api.utils import get_session
from market_alerts.entrypoints.llm_backend.containers import session_manager_singleton
from market_alerts.entrypoints.llm_backend.domain.exceptions import (
    BacktestingNotPerformedError,
    DataNotFetchedError,
    LLMChatNotSubmittedError,
)
from market_alerts.entrypoints.llm_backend.domain.services import (
    load_model_into_session,
)
from market_alerts.entrypoints.llm_backend.infrastructure.access_management.context_vars import (
    user,
)
from market_alerts.entrypoints.llm_backend.infrastructure.session import Session, Steps
from market_alerts.utils import session_state_getter

logger = logging.getLogger(__name__)

files_router = APIRouter(prefix="/files")


@files_router.get("/export", tags=["Files"])
def export_model_as_json(
    request_model: StrategyTitleModel = Depends(),
    session: Session = Depends(get_session),
):
    if not session.flow_status.is_step_done(Steps.FETCH_TICKERS):
        raise DataNotFetchedError

    data = session_state_getter(session)

    model_json = json.dumps(data, indent=4)

    filename = f"{request_model.strategy_title}.json" if request_model.strategy_title else "model.json"

    return Response(
        content=model_json,
        media_type="application/json",
        headers={f"Content-Disposition": f"attachment; filename={quote(filename)}"},
    )


@files_router.post("/import", tags=["Files"])
def import_model_as_json(
    file: UploadFile = File(...),
    session: Session = Depends(get_session),
):
    current_user = user.get()

    file_bytes = file.file.read()

    try:
        model_json = json.loads(file_bytes.decode())
    except (UnicodeDecodeError, json.JSONDecodeError):
        raise FileImportError("Invalid JSON file")

    logger.info("Importing the model from file...")

    session.reset_flow_status()
    session.data = DefaultSession().model_dump()
    session_manager_singleton.save(current_user.email, session)

    pipeline_start_time = time.time()

    pipeline_task_ids = load_model_into_session(model_json, session.session_id)

    for task_id in pipeline_task_ids:
        session.add_action(task_id, pipeline_start_time)
    session_manager_singleton.save(current_user.email, session)

    return {
        "task_id": pipeline_task_ids[-1],
    }


@files_router.get("/download_pnl", tags=["Files"])
def download_pnl_report(
    request_model: StrategyTitleModel = Depends(),
    session: Session = Depends(get_session),
):
    if not session.flow_status.is_step_done(Steps.PERFORM_BACKTESTING):
        raise BacktestingNotPerformedError

    csv_report = create_pnl_report(session)
    filename = f"{request_model.strategy_title}_pnl.csv" if request_model.strategy_title else "pnl.csv"

    headers = {"Content-Disposition": f"attachment; filename={quote(filename)}", "Content-Type": "text/csv"}

    return Response(content=csv_report, media_type="text/csv", headers=headers)


@files_router.get("/download_trading_stats_by_symbol", tags=["Files"])
def download_trading_stats_by_symbol_report(
    request_model: StrategyTitleModel = Depends(),
    session: Session = Depends(get_session),
):
    if not session.flow_status.is_step_done(Steps.PERFORM_BACKTESTING):
        raise BacktestingNotPerformedError

    csv_report = create_stats_by_symbol_report(session["trading_stats_by_symbol"])
    filename = (
        f"{request_model.strategy_title}_trading_stats_by_symbol.csv"
        if request_model.strategy_title
        else "trading_stats_by_symbol.csv"
    )

    headers = {"Content-Disposition": f"attachment; filename={quote(filename)}", "Content-Type": "text/csv"}

    return Response(content=csv_report, media_type="text/csv", headers=headers)


@files_router.get("/download_trades", tags=["Files"])
def download_trades_report(
    request_model: StrategyTitleModel = Depends(),
    session: Session = Depends(get_session),
):
    if not session.flow_status.is_step_done(Steps.PERFORM_BACKTESTING):
        raise BacktestingNotPerformedError

    csv_report = create_stats_by_symbol_report(session["trades_by_symbol"])
    filename = f"{request_model.strategy_title}_trades.csv" if request_model.strategy_title else "trades.csv"

    headers = {"Content-Disposition": f"attachment; filename={quote(filename)}", "Content-Type": "text/csv"}

    return Response(content=csv_report, media_type="text/csv", headers=headers)


@files_router.get("/download_prompt", tags=["Files"])
def download_latest_prompt(
    request_model: StrategyTitleModel = Depends(),
    session: Session = Depends(get_session),
):
    if not session.flow_status.is_step_done(Steps.SUBMIT_LLM_CHAT) or not session["last_llm_context"]:
        raise LLMChatNotSubmittedError

    chat_llm_input = session["last_llm_context"]
    filename = f"{request_model.strategy_title}_prompt.json" if request_model.strategy_title else "prompt.json"
    headers = {"Content-Disposition": f"attachment; filename={quote(filename)}", "Content-Type": "application/json"}

    return Response(content=chat_llm_input, headers=headers)
