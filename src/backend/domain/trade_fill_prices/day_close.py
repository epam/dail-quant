from typing import Any, Dict

from market_alerts.domain.default_list import DefaultList

from .base import TradeFillPriceBase


class DayClose(TradeFillPriceBase):
    BACKEND_KEY = "day_close"
    FILL_PRICE_NAME = "Day close"
    IS_DEFAULT = False

    def get_list_trade_fill_price_by_symbol(self, data_by_symbol: Dict[str, Any]) -> Dict[str, Any]:
        return {key: DefaultList(value["close"].tolist(), value["close"][0]) for key, value in data_by_symbol.items()}


#     def get_list_trade_fill_price_by_symbol(self, data_by_symbol: Dict[str, Any]) -> Dict[str, Any]:
#         return {
#             key: DefaultList(value["close"].shift(1).fillna(value["close"].tolist()[0]).tolist(), value["close"][0])
#             for key, value in data_by_symbol.items()
#         }
