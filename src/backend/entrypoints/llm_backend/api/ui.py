from fastapi import APIRouter

from market_alerts.containers import user_prompt_types
from market_alerts.domain.constants import (
    DIVIDEND_FIELDS,
    ECONOMIC_INDICATORS_STRUCTURE,
    FUNDAMENTALS_STRUCTURE
)
from market_alerts.domain.dtos import (
    data_periodicities_dtos,
    data_providers_dtos,
    data_time_ranges_dtos,
    datasets_dtos,
    optimization_samplers_dtos,
    optimization_target_funcs_dtos,
    trade_fill_prices_dtos,
)
from market_alerts.openai_utils import get_available_models

ui_router = APIRouter(prefix="/ui")


@ui_router.get("", tags=["UI"])
def get_ui_options():
    llm_models = [
        dict(label=model_name, query_param=backend_name, default_checked=default_checked)
        for model_name, (backend_name, default_checked) in get_available_models().items()
    ]

    return {
        "data_providers": [dp.__dict__ for dp in data_providers_dtos],
        "data_periodicities": [dp.__dict__ for dp in data_periodicities_dtos],
        "data_time_ranges": [dtr.__dict__ for dtr in data_time_ranges_dtos],
        "datasets": [ds.__dict__ for ds in datasets_dtos],
        "models": llm_models,
        "trade_fill_prices": [tfp.__dict__ for tfp in trade_fill_prices_dtos],
        "economic_indicators": ECONOMIC_INDICATORS_STRUCTURE,
        "fundamentals": FUNDAMENTALS_STRUCTURE,
        "dividend_fields": DIVIDEND_FIELDS,
        "user_prompt_types": user_prompt_types,
        "optimization_samplers": optimization_samplers_dtos,
        "optimization_target_funcs": optimization_target_funcs_dtos,
    }
