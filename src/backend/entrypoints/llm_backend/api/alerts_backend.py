import json
import logging
import re
import time
from functools import partial
from typing import Any, Dict

from fastapi import APIRouter, Body, Depends

from market_alerts.containers import alerts_backend_proxy_singleton
from market_alerts.domain.constants import EMAIL_REGEXP, PROMPTS_SEPARATOR
from market_alerts.domain.exceptions import TradingRuleAlreadyExistsError
from market_alerts.domain.exceptions.alerts_backend import (
    SaveModelInputError,
    TradingRuleNotFoundError,
)
from market_alerts.entrypoints.llm_backend.api.models.alerts_backend import (
    SaveTradingStrategyRequestModel,
    ShareTradingStrategyRequestModel,
)
from market_alerts.entrypoints.llm_backend.api.models.session import DefaultSession
from market_alerts.entrypoints.llm_backend.api.utils import get_session
from market_alerts.entrypoints.llm_backend.containers import session_manager_singleton
from market_alerts.entrypoints.llm_backend.domain.exceptions import DataNotFetchedError
from market_alerts.entrypoints.llm_backend.domain.services import (
    load_model_into_session,
)
from market_alerts.entrypoints.llm_backend.infrastructure.access_management.context_vars import (
    user,
)
from market_alerts.entrypoints.llm_backend.infrastructure.session import Session, Steps
from market_alerts.infrastructure.services.proxy.alerts_backend.exceptions import (
    GetTradingRuleError,
)

logger = logging.getLogger(__name__)

alerts_backend_router = APIRouter(prefix="/alerts_backend")


@alerts_backend_router.get("/trading_model/open/{model_id}", tags=["Alerts Backend"])
def open_trading_model(
    model_id: int,
    session: Session = Depends(get_session),
):
    current_user = user.get()

    try:
        model_json = alerts_backend_proxy_singleton.get_trading_rule_by_id(model_id)
    except GetTradingRuleError:
        raise TradingRuleNotFoundError(f"Trading model with id {model_id} not found")

    logger.info("Starting to open the model with id %s...", model_id)

    session.reset_flow_status()
    session.data = DefaultSession().model_dump()
    session_manager_singleton.save(current_user.email, session)

    pipeline_start_time = time.time()

    pipeline_task_ids = load_model_into_session(model_json, session.session_id, model_id)

    for task_id in pipeline_task_ids:
        session.add_action(task_id, pipeline_start_time)
    session_manager_singleton.save(current_user.email, session)

    return {
        "task_id": pipeline_task_ids[-1],
    }


@alerts_backend_router.post("/trading_model/save", tags=["Alerts Backend"])
def save_trading_model(
    request_model: SaveTradingStrategyRequestModel = Body(None),
    session: Session = Depends(get_session),
):
    if not session.flow_status.is_step_done(Steps.FETCH_TICKERS):
        raise DataNotFetchedError
    if request_model is None or not request_model.strategy_title:
        raise SaveModelInputError("Please, provide strategy title")

    current_user = user.get()
    empty_description = ""

    prepared_body_getter = partial(_get_save_model_request_body, session, current_user.email, empty_description)
    create_request_body = prepared_body_getter(request_model.strategy_title, False)

    if session.flow_status.is_model_opened:
        try:
            trading_rule = alerts_backend_proxy_singleton.get_trading_rule_by_id(session.flow_status.opened_model_id)

            if trading_rule["title"] != request_model.strategy_title:
                trading_rule = None
        except GetTradingRuleError:
            trading_rule = None

        _save_trading_rule(trading_rule, request_model.strategy_title, prepared_body_getter, create_request_body)
    else:
        _save_new_trading_rule(
            request_model.strategy_title, request_model.ignore_exists, False, prepared_body_getter, create_request_body
        )

    session.flow_status.unset_model_opened()
    session_manager_singleton.save(current_user.email, session)


@alerts_backend_router.post("/trading_model/share", tags=["Alerts Backend"])
def share_trading_model(
    request_model: ShareTradingStrategyRequestModel = Body(None),
    session: Session = Depends(get_session),
):
    if not session.flow_status.is_step_done(Steps.FETCH_TICKERS):
        raise DataNotFetchedError
    if request_model is None or not request_model.strategy_title:
        raise SaveModelInputError("Please, provide strategy title")
    if not request_model.recipient and not request_model.make_public:
        raise SaveModelInputError("Either a recipient must be present, or the model must be public")

    current_user = user.get()

    prepared_body_getter = partial(_get_save_model_request_body, session, current_user.email, request_model.description)
    create_request_body = prepared_body_getter(request_model.strategy_title, request_model.make_public)

    if request_model.make_public:
        _save_new_trading_rule(
            request_model.strategy_title,
            request_model.ignore_exists,
            request_model.make_public,
            prepared_body_getter,
            create_request_body,
        )
    else:
        if not re.match(EMAIL_REGEXP, request_model.recipient):
            raise SaveModelInputError(f"Recipient's email '{request_model.recipient}' is invalid")
        alerts_backend_proxy_singleton.share_trading_rule(create_request_body, request_model.recipient)


def _save_new_trading_rule(
    strategy_title: str, ignore_exists: bool, is_public: bool, prepared_body_getter, create_request_body
) -> None:
    if is_public:
        trading_rule = alerts_backend_proxy_singleton.get_trading_rule_by_title_from_public(strategy_title)
    else:
        trading_rule = alerts_backend_proxy_singleton.get_trading_rule_by_title_from_personal(strategy_title)

    if trading_rule and not ignore_exists:
        raise TradingRuleAlreadyExistsError(f"Model '{strategy_title}' already exists. Do you want to overwrite it?")

    _save_trading_rule(trading_rule, strategy_title, prepared_body_getter, create_request_body)


def _save_trading_rule(
    trading_rule: Dict[str, Any], strategy_title: str, prepared_body_getter, create_request_body: Dict[str, Any]
) -> None:
    if trading_rule:
        update_request_body = _get_update_request_body(prepared_body_getter, strategy_title, trading_rule)
        alerts_backend_proxy_singleton.update_trading_rule(update_request_body)
        logger.info("Model with id %s was updated successfully", update_request_body["id"])
    else:
        model_response = alerts_backend_proxy_singleton.create_trading_rule(create_request_body)
        logger.info("Model with id %s was created successfully", model_response["id"])


def _get_save_model_request_body(
    session: Session, email: str, description: str, strategy_title: str, is_public: bool
) -> Dict[str, Any]:
    return {
        "model": session.get("engine", "gpt-4"),
        "title": strategy_title,
        "user_id": email,
        "data_source": session.get("data_provider", "TW"),
        "datasets": session.get("datasets_keys", None),
        "periodicity": session.get("interval"),
        "time_range": session.get("time_period"),
        "tickers_prompt": session.get("tradable_symbols_prompt"),
        "tickers": list(session["true_symbols"].values()),
        "economic_indicators": session.get("economic_indicators"),
        "additional_dividend_fields": session.get("dividend_fields"),
        "account_for_dividends": session.get("use_dividends_trading", False),
        "active": True,
        "end_time": None,
        "public": is_public,
        "indicators_prompt": PROMPTS_SEPARATOR.join(session.get("indicators_dialogue", [])),
        "indicators_logic": session.get("indicators_code", ""),
        "trading_prompt": "",
        "trading_logic": "",
        "account_currency": session.get("actual_currency", "USD"),
        "bet_size": str(session.get("bet_size", 0)),
        "total_gross_limit": str(session.get("total_gross_limit", 0)),
        "per_instrument_gross_limit": str(session.get("per_instrument_gross_limit", 0)),
        "nop_limit": str(session.get("nop_limit", 0)),
        "description": description,
        "trade_fill_price": session.get("trade_fill_price", "next_day_open"),
        "execution_cost_bps": str(session.get("execution_cost_bps", 0)),
        "optimization_trials": session.get("optimization_trials", 5),
        "optimization_train_size": session.get("optimization_train_size", 1.0),
        "optimization_params": json.dumps(session.get("optimization_params", [])),
        "optimization_minimize": session.get("optimization_minimize", True),
        "optimization_maximize": session.get("optimization_maximize", True),
        "optimization_sampler": session.get("optimization_sampler", "tpe"),
        "optimization_target_func": session.get("optimization_target_func", "gpl"),
    }


def _get_update_request_body(prepared_body_getter, strategy_title: str, trading_rule: Dict[str, Any]) -> Dict[str, Any]:
    update_request_body = prepared_body_getter(strategy_title, trading_rule["public"])
    update_request_body["id"] = trading_rule["id"]
    update_request_body["created"] = trading_rule["created"]
    return update_request_body
