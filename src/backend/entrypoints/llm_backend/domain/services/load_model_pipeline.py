import json
import uuid
from typing import Any, Dict, Optional

from celery import chain

from market_alerts.domain.constants import PROMPTS_SEPARATOR
from market_alerts.entrypoints.llm_backend.containers import celery_app
from market_alerts.entrypoints.llm_backend.infrastructure.utils import (
    create_task_headers,
)
from market_alerts.infrastructure.services.code import get_code_sections


def load_model_into_session(model_json: Dict[str, Any], session_id: str, model_id: Optional[int] = None) -> list[str]:
    try:
        llm_chat = model_json["indicators_prompt"]
        indicators_dialogue = llm_chat.split(PROMPTS_SEPARATOR) if llm_chat else []
    except KeyError:
        indicators_dialogue = model_json["indicators_dialogue"]

    tasks = [
        celery_app.signature(
            "chained_load_model_into_session",
            kwargs={
                "model_id": model_id,
                "is_public": model_json.get("public", False),
                "indicators_dialogue": indicators_dialogue,
                "data_provider": model_json.get("data_source") or model_json.get("data_provider"),
                "datasets": model_json.get("datasets") or model_json.get("datasets_keys"),
                "periodicity": model_json.get("periodicity") or model_json.get("interval"),
                "tradable_symbols_prompt": model_json.get("tickers_prompt") or model_json.get("tradable_symbols_prompt"),
                "supplementary_symbols_prompt": model_json.get("supplementary_symbols_prompt", ""),
                "economic_indicators": model_json.get("economic_indicators") or [],
                "dividend_fields": model_json.get("additional_dividend_fields") or [],
                "time_range": model_json.get("time_range") or model_json.get("time_period"),
                "strategy_title": model_json.get("title") or model_json.get("strategy_title"),
                "strategy_description": model_json.get("description") or model_json.get("strategy_description"),
                "actual_currency": model_json.get("account_currency") or model_json.get("actual_currency"),
                "bet_size": model_json["bet_size"],
                "per_instrument_gross_limit": model_json["per_instrument_gross_limit"],
                "total_gross_limit": model_json["total_gross_limit"],
                "nop_limit": model_json["nop_limit"],
                "account_for_dividends": model_json.get("use_dividends_trading") or model_json.get("account_for_dividends"),
                "trade_fill_price": model_json.get("fill_trade_price") or model_json.get("trade_fill_price"),
                "execution_cost_bps": model_json.get("execution_cost_bps"),
                "optimization_trials": model_json.get("optimization_trials"),
                "optimization_train_size": model_json.get("optimization_train_size") or 1.0,
                "optimization_params": json.loads(model_json.get("optimization_params") or "[]")
                if type(model_json.get("optimization_params")) is str
                else model_json.get("optimization_params", []),
                "optimization_minimize": model_json.get("optimization_minimize", True),
                "optimization_maximize": model_json.get("optimization_maximize", True),
                "optimization_sampler": model_json.get("optimization_sampler"),
                "optimization_target_func": model_json.get("optimization_target_func"),
            },
            queue="alerts_default",
            routing_key="alerts_default",
            headers=create_task_headers(session_id),
            task_id=str(uuid.uuid4()),
            immutable=True,
        ),
        celery_app.signature(
            "fetch_tickers",
            kwargs={
                "is_chained": True,
                "data_provider": model_json.get("data_source") or model_json.get("data_provider"),
                "datasets": model_json.get("datasets") or model_json.get("datasets_keys"),
                "periodicity": model_json.get("periodicity") or model_json.get("interval"),
                "tradable_symbols_prompt": model_json.get("tickers_prompt") or model_json.get("tradable_symbols_prompt"),
                "supplementary_symbols_prompt": model_json.get("supplementary_symbols_prompt", ""),
                "economic_indicators": model_json.get("economic_indicators") or [],
                "dividend_fields": model_json.get("additional_dividend_fields") or [],
                "time_range": model_json.get("time_range") or model_json.get("time_period"),
            },
            queue="alerts_default",
            routing_key="alerts_default",
            headers=create_task_headers(session_id),
            task_id=str(uuid.uuid4()),
            immutable=True,
        ),
    ]

    if indicators_dialogue:
        indicators_code, trading_code = get_code_sections(indicators_dialogue[-1])

        tasks.extend(
            (
                celery_app.signature(
                    "chained_submit_llm_chat",
                    kwargs={
                        "indicators_code": indicators_code,
                        "trading_code": trading_code,
                    },
                    queue="alerts_default",
                    routing_key="alerts_default",
                    headers=create_task_headers(session_id),
                    task_id=str(uuid.uuid4()),
                    immutable=True,
                ),
                celery_app.signature(
                    "calculate_indicators",
                    kwargs={
                        "is_chained": True,
                    },
                    queue="alerts_default",
                    routing_key="alerts_default",
                    headers=create_task_headers(session_id),
                    task_id=str(uuid.uuid4()),
                    immutable=True,
                ),
            )
        )

    pipeline = chain(*tasks)

    soft_time_limit = 2 * 60
    pipeline.apply_async(soft_time_limit=soft_time_limit, time_limit=soft_time_limit + 20)

    return [task.id for task in tasks]
