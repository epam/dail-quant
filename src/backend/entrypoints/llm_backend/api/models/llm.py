from typing import Dict, List

from fastapi import Query
from pydantic import BaseModel, Field

from .common import FlowStatusResponseModel


class SubmitLLMQueryRequestModel(BaseModel):
    llm_query: str = Field("")
    engine: str = Field("gpt-4")
    prompt_ids: List[int] = []


class LLMResponseModel(FlowStatusResponseModel):
    llm_response: str
    engine: str
    token_usage: Dict[str, int]
    request_timestamp: str
    execution_time: float


class BacktestingRequestModel(BaseModel):
    actual_currency: str = Query("USD", max_length=3)
    bet_size: float = Query(10000)
    per_instrument_gross_limit: float = Query(25000)
    total_gross_limit: float = Query(100000000)
    nop_limit: float = Query(100000000)
    account_for_dividends: bool = Query(False)
    trade_fill_price: str = Query("next_day_open")
    execution_cost_bps: float = Query(0)
