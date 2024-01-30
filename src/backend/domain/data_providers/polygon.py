import json
from datetime import datetime
from string import Template
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import requests

from market_alerts.domain.exceptions import (
    DataNotFoundError,
    MethodIsNotImplementedError,
)

from .default import DefaultDataProvider


class PolygonDataProvider(DefaultDataProvider):
    PROVIDER_ID = "PI"
    PROVIDER_NAME = "PolygonIO"

    POLYGON_PRICES_URL_TEMPLATE = Template(
        f"https://api.polygon.io/v2/aggs/ticker/$symbol/range/$multiplier/$timespan/$start_date/$end_date?sort=asc&apiKey=$api_key"
    )

    POLYGON_SYMBOL_SEARCH_TEMPLATE = Template(
        f"https://api.polygon.io/v3/reference/tickers?search=$symbol&limit=$outputsize&apiKey=$polygon_api_key"
    )

    POLYGON_DIVIDENDS_URL_TEMPLATE = Template(
        f"https://api.polygon.io/v3/reference/dividends?ticker=$symbol&apiKey=$polygon_api_key"
    )

    POLYGON_SYMBOL_META_TEMPLATE = Template(f"https://api.polygon.io/v3/reference/tickers?ticker=$symbol&apiKey=$polygon_api_key")

    def __init__(self, polygon_api_key: str):
        super().__init__()

        self.api_key = polygon_api_key

    def get_split_adjusted_prices(
        self, symbol: str, start_date: str, end_date: str, interval: str
    ) -> Tuple[Dict[str, Any], Optional[str]]:
        error_message = None
        multiplier, timespan = self._convert_periodicity_to_polygon_multiplier_timespan(interval)

        start_date = start_date.split()[0]
        end_date = end_date.split()[0]

        url = self.POLYGON_PRICES_URL_TEMPLATE.substitute(
            api_key=self.api_key,
            symbol=symbol,
            start_date=start_date,
            end_date=end_date,
            multiplier=multiplier,
            timespan=timespan,
        )

        data_prices = json.loads(requests.get(url).content.decode())

        if data_prices.get("code") == 400:
            raise DataNotFoundError(data_prices["message"])

        if data_prices:
            next_url = data_prices.get("next_url", None)

        converted_prices = self._convert_prices_to_twelvedata_format(data_prices, None)

        while next_url is not None:
            data_prices = json.loads(requests.get(next_url + f"&apiKey={self.api_key}").content.decode())

            if data_prices.get("code") == 400:
                raise DataNotFoundError(data_prices["message"])

            if data_prices:
                next_url = data_prices.get("next_url", None)
                if next_url is not None:
                    next_url = next_url + f"&apiKey={self.api_key}"
            else:
                next_url = None

            converted_prices = self._convert_prices_to_twelvedata_format(data_prices, converted_prices)

        return converted_prices, error_message

    def get_dividends(self, symbol: str, start_date: str, end_date: str) -> Tuple[Dict[str, Any], Optional[str]]:
        error_message = None
        start_date = start_date.split()[0]
        end_date = end_date.split()[0]

        url = self.POLYGON_DIVIDENDS_URL_TEMPLATE.substitute(polygon_api_key=self.api_key, symbol=symbol)

        json_dividends = requests.get(url).content.decode()
        data_dividends = json.loads(json_dividends)

        if data_dividends.get("code") == 400:
            raise DataNotFoundError(data_dividends["message"])

        while not self._check_dividends_are_full(start_date, data_dividends):
            next_url = data_dividends.get("next_url", None)

            if next_url is None:
                break

            next_divs = json.loads(requests.get(next_url + f"&apiKey={self.api_key}").content.decode())

            data_dividends["results"].extend(next_divs["results"])
            data_dividends["next_url"] = next_divs.get("next_url", None)

        trimmed_payments = self._trim_dividends_if_too_much_data(start_date, end_date, data_dividends)
        data_dividends["results"] = trimmed_payments

        return self._convert_dividends_to_twelvedata_format(data_dividends), error_message

    @staticmethod
    def _check_dividends_are_full(start_date: str, data_dividends: Dict[str, Any]) -> bool:
        if not data_dividends:
            return True

        start_datetime = datetime.strptime(start_date, "%Y-%m-%d")

        for payment in data_dividends["results"]:
            payment_datetime_ex = datetime.strptime(payment["ex_dividend_date"], "%Y-%m-%d")

            if payment_datetime_ex <= start_datetime:
                return True

        return False

    @staticmethod
    def _trim_dividends_if_too_much_data(start_date: str, end_date: str, data_dividends: Dict[str, Any]) -> Dict[str, Any]:
        start_datetime = datetime.strptime(start_date, "%Y-%m-%d")
        end_datetime = datetime.strptime(end_date, "%Y-%m-%d")

        result_trim = []

        for payment in data_dividends["results"]:
            payment_datetime_ex = datetime.strptime(payment["ex_dividend_date"], "%Y-%m-%d")

            if payment_datetime_ex >= start_datetime and payment_datetime_ex <= end_datetime:
                result_trim.append(payment)

        return result_trim

    @staticmethod
    def _convert_dividends_to_twelvedata_format(data_dividends: Dict[str, Any]):
        converted_result = {
            "meta": {
                "symbol": None,
                "name": "",
                "currency": None,
                "exchange": "",
                "mic_code": "",
                "exchange_timezone": "",
            },
            "dividends": [],
        }

        for payment in data_dividends["results"]:
            if converted_result["meta"]["symbol"] is None:
                converted_result["meta"]["symbol"] = payment["ticker"]
            if converted_result["meta"]["currency"] is None:
                converted_result["meta"]["currency"] = payment["currency"]

            converted_payment = {
                "ex_date": payment["ex_dividend_date"],
                "amount": payment["cash_amount"],
            }
            converted_result["dividends"].append(converted_payment)

        return converted_result

    @staticmethod
    def _convert_periodicity_to_polygon_multiplier_timespan(periodicity: str) -> Tuple[int, str]:
        multiplier = ""
        timespan = ""

        for char in periodicity:
            if char.isdigit():
                multiplier += char
            else:
                timespan += char

        if "h" in timespan:
            timespan = "hour"

        if "min" in timespan:
            timespan = "minute"

        return int(multiplier), timespan

    def _convert_prices_to_twelvedata_format(
        self, polygon_timeseries: Dict[str, Any], format_container: Dict[str, Any]
    ) -> Dict[str, Any]:
        if format_container is not None:
            result = format_container
        else:
            meta = self._get_ticker_meta(polygon_timeseries["ticker"])
            result = {
                "meta": meta,
                "values": [],
            }

        for val in polygon_timeseries.get("results", []):
            timestamp = datetime.fromtimestamp(val["t"] / 1000).strftime("%Y-%m-%d %H:%M:%S")

            result["values"].append(
                {
                    "datetime": timestamp.split(" ")[0],
                    "open": val["o"],
                    "high": val["h"],
                    "low": val["l"],
                    "close": val["c"],
                    "volume": val["v"],
                }
            )

        return result

    def search_ticker(self, tickers_query: str, output_limit: int = 1000) -> List[Dict[str, str]]:
        url = self.POLYGON_SYMBOL_SEARCH_TEMPLATE.substitute(
            symbol=tickers_query, outputsize=output_limit, polygon_api_key=self.api_key
        )

        data = json.loads(requests.get(url).content.decode())
        symbols = self._convert_ticker_info_to_twelvedate_format(data["results"])

        return symbols

    @staticmethod
    def _convert_ticker_info_to_twelvedate_format(ticker_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        converted_data = []

        for ticker in ticker_data:
            converted_ticker = {
                "symbol": ticker["ticker"],
                "instrument_name": ticker.get("name", ""),
                "exchange": ticker.get("primary_exchange", ""),
                "mic_code": "",
                "exchange_timezone": "",
                "instrument_type": ticker.get("type", ""),
                "country": ticker.get("locale", ""),
                "currency": ticker.get("currency_name", ""),
            }

            converted_data.append(converted_ticker)

        return converted_data

    def _get_ticker_meta(self, symbol: str) -> Dict[str, Any]:
        url = self.POLYGON_SYMBOL_META_TEMPLATE.substitute(polygon_api_key=self.api_key, symbol=symbol)

        data = json.loads(requests.get(url).content.decode())

        if len(data["results"]) > 0:
            meta = {
                "symbol": symbol,
                "interval": "",
                "currency": "USD",
                "exchange_timezone": data["results"][0].get("locale", ""),
                "exchange": data["results"][0].get("primary_exchange", ""),
                "mic_code": "",
                "type": data["results"][0].get("name", ""),
            }
        else:
            meta = {
                "symbol": symbol,
                "interval": "",
                "currency": "USD",
                "exchange_timezone": "",
                "exchange": "",
                "mic_code": "",
                "type": "",
            }

        return meta

    def get_earnings(self, symbol: str, start_date: str, end_date: str) -> Dict[str, Any]:
        return {}

    def get_balance_sheets(self, symbols: List[str], true_symbols: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        return {}, {}

    def get_income_statements(self, symbols: List[str], true_symbols: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        return {}, {}

    def get_economic_indicators(self, symbol: str, start_date: str, end_date: str) -> Dict[str, Any]:
        NotImplementedError("Method get_economic_indicators is not implemented for polygon io data provider.")
