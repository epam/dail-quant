from typing import Any, Dict

from market_alerts.domain.default_list import DefaultList

from .base import TradeFillPriceBase


class NextDayMid(TradeFillPriceBase):
    BACKEND_KEY = "next_day_mid"
    FILL_PRICE_NAME = "Next day mid"
    IS_DEFAULT = False

    def get_list_trade_fill_price_by_symbol(self, data_by_symbol: Dict[str, Any]) -> Dict[str, Any]:
        return {
            key: DefaultList(((value["open"] + value["close"]) / 2).tolist(), (value["open"][0] + value["close"][0]) / 2)
            for key, value in data_by_symbol.items()
        }
