from .common import StrategyTitleModel


class SaveTradingStrategyRequestModel(StrategyTitleModel):
    ignore_exists: bool = False


class ShareTradingStrategyRequestModel(SaveTradingStrategyRequestModel):
    recipient: str
    make_public: bool = False
    description: str = ""
