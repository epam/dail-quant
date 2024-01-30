from typing import Any, List, Optional

from pydantic import BaseModel, Field

from market_alerts.domain.dtos import (
    data_providers_dtos,
    datasets_dtos,
    optimization_samplers_dtos,
    optimization_target_funcs_dtos,
    trade_fill_prices_dtos,
)

# TODO: change defaults
from market_alerts.utils import string_to_ms

from .common import FlowStatusResponseModel


class DefaultSession(BaseModel):
    data_provider: str = next(filter(lambda dp: dp.default_checked, data_providers_dtos)).query_param
    datasets_keys: List[str] = [next(filter(lambda ds: ds.default_checked, datasets_dtos)).query_param]
    interval: int = string_to_ms("1 day")
    tradable_symbols_prompt: str = ""
    supplementary_symbols_prompt: str = ""
    economic_indicators: List[str] = []
    dividend_fields: List[str] = []
    time_period: int = string_to_ms("10 years")
    user_prompt_ids: List[int] = []
    generic_user_prompt: str = ""
    indicator_user_prompt: str = ""
    trading_user_prompt: str = ""
    indicators_dialogue: List[str] = []
    fx_rates: dict[str, Any] = dict()
    last_llm_context: str = ""
    actual_currency: str = "USD"
    bet_size: float = 0
    per_instrument_gross_limit: float = 0
    total_gross_limit: float = 0
    nop_limit: float = 0
    use_dividends_trading: bool = False
    fill_trade_price: str = next(filter(lambda tfp: tfp.default_checked, trade_fill_prices_dtos)).query_param
    execution_cost_bps: float = 0
    optimization_trials: int = 5
    optimization_train_size: float = 1.0
    optimization_params: list[dict[str, Any]] = []
    optimization_minimize: bool = True
    optimization_maximize: bool = True
    optimization_sampler: str = next(filter(lambda samp: samp.default_checked, optimization_samplers_dtos)).query_param
    optimization_target_func: str = next(filter(lambda tf: tf.default_checked, optimization_target_funcs_dtos)).query_param


class SessionResponseModel(BaseModel):
    session_id: str = Field(alias="sessionId")
    expires_in: int


class SessionInfoResponseModel(FlowStatusResponseModel):
    session_id: Optional[str] = Field(None, alias="sessionId")
    data_provider: Optional[str] = None
    datasets: Optional[List[str]] = None
    periodicity: Optional[int] = None
    tradable_symbols_prompt: Optional[str] = None
    supplementary_symbols_prompt: Optional[str] = None
    economic_indicators: Optional[List[str]] = None
    dividend_fields: Optional[List[str]] = None
    time_range: Optional[int] = None
    user_prompt_ids: List[int] = None
    generic_user_prompt: str = None
    indicator_user_prompt: str = None
    trading_user_prompt: str = None
    indicators_dialogue: Optional[List[str]] = None
    last_llm_context: str = None
    strategy_title: Optional[str] = None
    strategy_description: Optional[str] = None
    actual_currency: Optional[str] = None
    bet_size: Optional[float] = None
    per_instrument_gross_limit: Optional[float] = None
    total_gross_limit: Optional[float] = None
    nop_limit: Optional[float] = None
    account_for_dividends: Optional[bool] = None
    trade_fill_price: Optional[str] = None
    execution_cost_bps: Optional[float] = None
    optimization_trials: Optional[int] = None
    optimization_train_size: Optional[float] = None
    optimization_params: Optional[list[dict[str, Any]]] = None
    optimization_minimize: Optional[bool] = None
    optimization_maximize: Optional[bool] = None
    optimization_sampler: Optional[str] = None
    optimization_target_func: Optional[str] = None


class UpdateCodeRequestModel(BaseModel):
    indicators_code: str
    trading_code: str
