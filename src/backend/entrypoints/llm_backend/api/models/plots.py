from fastapi import Query
from pydantic import BaseModel, Field


class DataRequestModel(BaseModel):
    symbols: list[str] = Field(Query(["AAPL"]))


class TradingDataRequestModel(DataRequestModel):
    is_indicators_needed: bool = Query(False)
