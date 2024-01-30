from abc import ABC, abstractclassmethod
from typing import Any, Dict


class TradeFillPriceBase(ABC):
    BACKEND_KEY = ""
    FILL_PRICE_NAME = ""
    IS_DEFAULT = False

    @abstractclassmethod
    def get_list_trade_fill_price_by_symbol(self, data_by_symbol: Dict[str, Any]) -> Dict[str, Any]:
        pass
