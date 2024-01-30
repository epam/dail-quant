from typing import Any, Dict, Optional

from pydantic import BaseModel, Field


class FlowStatusResponseModel(BaseModel):
    flow_status: Dict[str, Any]


class StrategyTitleModel(BaseModel):
    strategy_title: Optional[str] = Field("my_model", pattern=r"^[a-zA-Z0-9_\s-]+$")
