from typing import Any, List

from fastapi import Query
from pydantic import BaseModel, Field

from market_alerts.domain.dtos import data_providers_dtos, datasets_dtos
from market_alerts.utils import string_to_ms


class TickersFetchRequestModel(BaseModel):
    data_provider: str = Query(data_providers_dtos[0].query_param)
    datasets: List[str] = Field(Query([datasets_dtos[0].query_param]))
    periodicity: int = Query(string_to_ms("1 day"))
    tradable_symbols_prompt: str = Query("AAPL")
    supplementary_symbols_prompt: str = Query("")
    economic_indicators: List[str] = Field(Query([]))
    dividend_fields: List[str] = Field(Query([]))
    time_range: int = Query(string_to_ms("10 years"))


class TickersFetchInfoResponseModel(BaseModel):
    data_provider: str
    datasets: list[str]
    periodicity: int
    time_range: int
    fetched_symbols_meta: list[str]
    plots_meta: dict[str, Any]
    synth_formulas: dict[str, str]
    request_timestamp: str
    execution_time: float


class TickersLookupRequestModel(BaseModel):
    data_provider: str = Query(data_providers_dtos[0].query_param)
    tickers_query: str = Query("AAPL")
    output_size: int = Query(30, ge=30, le=120)
