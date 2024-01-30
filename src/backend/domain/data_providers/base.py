from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, List, Optional, Tuple

import pandas as pd


class BaseDataProvider(ABC):
    @abstractmethod
    def fetch_datasets(
        self,
        true_symbols: Dict[str, str],
        start_date: str,
        end_date: str,
        interval: str,
        datasets: List[str],
        primary_provider_id: str,
        true_economic_indicators: Dict[str, str],
        additional_dividends_fields: List[str],
        data_by_symbol: Dict[str, pd.DataFrame],
        meta_by_symbol: Dict[str, Any],
        progress_callback: Optional[Callable[[None], None]],
    ) -> Tuple[Dict[str, pd.DataFrame], Dict[str, Any], Optional[str]]:
        pass

    @abstractmethod
    def get_parallel_dividend_adjusted_prices(
        self, symbols: List[str], start_date: str, end_date: str, interval: str
    ) -> Dict[str, Any]:
        pass

    @abstractmethod
    def get_parallel_split_adjusted_prices(
        self, symbols: List[str], start_date: str, end_date: str, interval: str
    ) -> Dict[str, Any]:
        pass

    @abstractmethod
    def get_parallel_dividends(self, symbols: List[str], start_date: str, end_date: str) -> Dict[str, Any]:
        pass

    @abstractmethod
    def get_parallel_earnings(self, symbols: List[str], start_date: str, end_date: str) -> Dict[str, Any]:
        pass

    @abstractmethod
    def get_parallel_economic_indicators(
        self, true_economic_indicators: Dict[str, str], start_date: str, end_date: str
    ) -> Dict[str, Any]:
        pass

    @abstractmethod
    def get_split_adjusted_prices(self, symbol: str, start_date: str, end_date: str, interval: str) -> Dict[str, Any]:
        pass

    @abstractmethod
    def get_dividends(self, symbol: str, start_date: str, end_date: str) -> Dict[str, Any]:
        pass

    @abstractmethod
    def get_earnings(self, symbol: str, start_date: str, end_date: str) -> Dict[str, Any]:
        pass

    @abstractmethod
    def get_balance_sheets(self, symbols: List[str], true_symbols: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        pass

    @abstractmethod
    def get_income_statements(self, symbols: List[str], true_symbols: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        pass

    @abstractmethod
    def get_economic_indicators(self, symbol: str, start_date: str, end_date: str) -> Dict[str, Any]:
        pass

    @abstractmethod
    def search_ticker(self, tickers_query: str, output_limit: int = 30) -> List[Dict[str, str]]:
        pass
