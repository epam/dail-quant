from typing import Any, Dict

from market_alerts.domain.default_list import DefaultList

from .base import TradeFillPriceBase


class NextDayClose(TradeFillPriceBase):
    BACKEND_KEY = "next_day_close"
    FILL_PRICE_NAME = "Next day close"
    IS_DEFAULT = False

    def get_list_trade_fill_price_by_symbol(self, data_by_symbol: Dict[str, Any]) -> Dict[str, Any]:
        return {key: DefaultList(value["close"].tolist(), value["close"][0]) for key, value in data_by_symbol.items()}
