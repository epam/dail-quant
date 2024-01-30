from typing import List, Optional, Union

from .base import TradeFillPriceBase


class TradeFillPriceContainer:
    def __init__(self, trade_fill_prices: List[TradeFillPriceBase] = None):
        if trade_fill_prices is None:
            self.trade_fill_prices = []
        else:
            self.trade_fill_prices = trade_fill_prices

    def get_trade_price_by_name(self, fill_price_name: str) -> Optional[TradeFillPriceBase]:
        for price in self.trade_fill_prices:
            if price.FILL_PRICE_NAME == fill_price_name:
                return price

    def get_trade_price_by_backend_key(self, fill_price_backend_key: str) -> Optional[TradeFillPriceBase]:
        for price in self.trade_fill_prices:
            if price.BACKEND_KEY == fill_price_backend_key:
                return price

    def get_default_price(self) -> Optional[TradeFillPriceBase]:
        for price in self.trade_fill_prices:
            if price.IS_DEFAULT:
                return price

    def get_price_index(self, backend_key: str) -> int:
        price = self.get_trade_price_by_backend_key(backend_key)

        try:
            return self.trade_fill_prices.index(price)
        except ValueError as e:
            if "None" in str(e):
                return self.trade_fill_prices.index(self.get_default_price())
            else:
                raise e

    def get_prices(self) -> List[TradeFillPriceBase]:
        return self.trade_fill_prices

    def get_price_backend_keys(self) -> List[str]:
        return [price.BACKEND_KEY for price in self.trade_fill_prices]

    def get_price_names(self) -> List[str]:
        return [price.FILL_PRICE_NAME for price in self.trade_fill_prices]
