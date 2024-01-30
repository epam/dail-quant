from typing import Any, Dict

from market_alerts.domain.default_list import DefaultList

from .base import TradeFillPriceBase


class NextDayOpen(TradeFillPriceBase):
    BACKEND_KEY = "next_day_open"
    FILL_PRICE_NAME = "Next day open"
    IS_DEFAULT = True

    def get_list_trade_fill_price_by_symbol(self, data_by_symbol: Dict[str, Any]) -> Dict[str, Any]:
        return {key: DefaultList(value["open"].tolist(), value["open"][0]) for key, value in data_by_symbol.items()}
