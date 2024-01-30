import json
from string import Template
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import requests

from market_alerts.domain.exceptions import DataNotFoundError

from .default import DefaultDataProvider


# Twelvedata error codes:
# 400 - Bad Request - There is an error with one or multiple parameters.
# 401 - Unauthorize - Your API key is wrong or not valid.
# 403 - Forbidden - Your API key is valid but has no permissions to make request available on the upper plans.
# 404 - Not Found - The specified data can not be found.
# 414 - Parameter Too Long - The parameter which accepts multiple values is out of range.
# 429 - Too Many Requests - You've reached your API request limits.
# 500 - Internal Server Error - There is an error on the server-side. Try again later.
class TwelveDataProvider(DefaultDataProvider):
    PROVIDER_ID = "TW"
    PROVIDER_NAME = "TwelveData"
    DEFAULT_CHECKED = True

    TWELVE_PRICES_URL_TEMPLATE = Template(
        f"https://api.twelvedata.com/time_series?symbol=$symbol&interval=$interval&start_date=$start_date&end_date=$end_date&order=ASC&apikey=$api_key"
    )

    TWELVE_DIVIDENDS_URL_TEMPLATE = Template(
        f"https://api.twelvedata.com/dividends?symbol=$symbol&start_date=$start_date&end_date=$end_date&order=ASC&apikey=$api_key"
    )

    TWELVE_EARNINGS_URL_TEMPLATE = Template(
        f"https://api.twelvedata.com/earnings?symbol=$symbol&start_date=$start_date&end_date=$end_date&order=ASC&apikey=$api_key"
    )

    TWELVE_BALANCE_SHEET_URL_TEMPLATE = Template(f"https://api.twelvedata.com/balance_sheet?symbol=$symbol&apikey=$api_key")

    TWELVE_INCOME_STATEMENT_URL_TEMPLATE = Template(f"https://api.twelvedata.com/income_statement?symbol=$symbol&apikey=$api_key")

    TWELVE_SYMBOL_SEARCH_URL_TEMPLATE = Template(
        f"https://api.twelvedata.com/symbol_search?symbol=$symbol&outputsize=$outputsize"
    )

    def __init__(self, twelve_api_key: str):
        super().__init__()

        self.api_key = twelve_api_key

    def get_split_adjusted_prices(
        self, symbol: str, start_date: str, end_date: str, interval: str
    ) -> Tuple[Dict[str, Any], Optional[str]]:
        url = self.TWELVE_PRICES_URL_TEMPLATE.substitute(
            api_key=self.api_key,
            symbol=symbol,
            start_date=start_date,
            end_date=end_date,
            interval=interval,
        )
        data_prices = json.loads(requests.get(url).content.decode())
        error_message = None

        if data_prices.get("code", 200) != 200:
            error_message = f"Error on fetching split adjusted prices for symbol {symbol}, for following time range: {start_date} - {end_date} with message: {data_prices['message']}"
            if data_prices["code"] == 429:
                return None, error_message
            elif data_prices["code"] == 400:
                return {
                    "meta": {
                        "symbol": symbol,
                        "interval": interval,
                        "currency": "",
                        "exchange_timezone": "",
                        "exchange": "",
                        "mic_code": "",
                        "type": "",
                    },
                    "values": [],
                }, None
            else:
                raise DataNotFoundError(error_message)

        last_datetime = data_prices["values"][0]["datetime"]
        desired_date = start_date.split()[0]

        if data_prices["meta"]["type"] not in ["Physical Currency", "Digital Currency"]:
            data_prices["values"] = self._drop_saturday_prices(data_prices)

        if last_datetime == end_date:
            return data_prices, error_message

        if last_datetime != desired_date:
            try:
                additional_timeseries, error_message = self.get_split_adjusted_prices(
                    symbol, desired_date, last_datetime, interval
                )
            except DataNotFoundError:
                return data_prices, error_message

            if additional_timeseries is not None:
                additional_timeseries["values"].extend(data_prices["values"])
                data_prices["values"] = additional_timeseries["values"]

        return data_prices, error_message

    @staticmethod
    def _drop_saturday_prices(data_prices: Dict[str, Any]) -> Dict[str, Any]:
        prices_df = pd.DataFrame.from_dict(data_prices["values"])

        prices_df["temp_datetime"] = pd.to_datetime(prices_df["datetime"])

        prices_df = prices_df[prices_df["temp_datetime"].apply(lambda x: x.weekday() != 5)]
        prices_df.drop(columns="temp_datetime", inplace=True)

        return list(prices_df.transpose().to_dict().values())

    def get_dividends(self, symbol: str, start_date: str, end_date: str) -> Tuple[Dict[str, Any], Optional[str]]:
        url_dividends = self.TWELVE_DIVIDENDS_URL_TEMPLATE.substitute(
            api_key=self.api_key,
            symbol=symbol,
            start_date=start_date,
            end_date=end_date,
        )
        error_message = None
        content = requests.get(url_dividends).content.decode()
        data_dividends = json.loads(content)

        if data_dividends.get("code", 200) != 200:
            if data_dividends["code"] in [429, 400]:
                error_message = data_dividends["message"]
                return None, error_message
            else:
                raise DataNotFoundError(
                    f"Error on fetching dividends for symbol {symbol}, for following time range: {start_date} - {end_date} with message: {data_dividends.get('message', 'Empty message')}, with code {data_dividends['code']}"
                )

        return data_dividends, error_message

    def get_earnings(self, symbol: str, start_date: str, end_date: str) -> Dict[str, Any]:
        url_earnings = self.TWELVE_EARNINGS_URL_TEMPLATE.substitute(
            api_key=self.api_key,
            symbol=symbol,
            start_date=start_date,
            end_date=end_date,
        )
        data_earnings = json.loads(requests.get(url_earnings).content.decode())

        if data_earnings.get("code") == 400:
            raise DataNotFoundError(data_earnings["message"])

        return data_earnings

    def get_balance_sheets(self, symbols: List[str], true_symbols: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        sheet_by_symbol = {}
        currency_by_symbol = {}

        tmp_true_symbols = {}
        for key, value in true_symbols.items():
            tmp_true_symbols[key] = value.split("@")[0]

        true_symbols = tmp_true_symbols

        for symbol in symbols:
            url = self.TWELVE_BALANCE_SHEET_URL_TEMPLATE.substitute(api_key=self.api_key, symbol=true_symbols[symbol])

            resp = json.loads(requests.get(url).content.decode())

            if resp.get("code") == 404:
                raise DataNotFoundError(resp["message"])

            balance_sheets = resp.get("balance_sheet", None)
            if balance_sheets is None:
                continue
            currency = resp["meta"]["currency"]

            symbol_key = true_symbols[symbol]
            symbol_key = symbol_key.replace(":", "\:")

            currency_by_symbol[symbol_key] = currency
            sheet_by_symbol[symbol_key] = self._get_sheet(balance_sheets)

        return sheet_by_symbol, currency_by_symbol

    def get_income_statements(self, symbols: List[str], true_symbols: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        sheet_by_symbol = {}
        currency_by_symbol = {}

        tmp_true_symbols = {}
        for key, value in true_symbols.items():
            tmp_true_symbols[key] = value.split("@")[0]

        true_symbols = tmp_true_symbols
        for symbol in symbols:
            url = self.TWELVE_INCOME_STATEMENT_URL_TEMPLATE.substitute(api_key=self.api_key, symbol=true_symbols[symbol])

            data = json.loads(requests.get(url).content.decode())

            income_statements = data.get("income_statement", None)
            if income_statements is None:
                continue
            currency = data["meta"]["currency"]

            symbol_key = true_symbols[symbol]
            symbol_key = symbol_key.replace(":", "\:")

            currency_by_symbol[symbol_key] = currency
            sheet_by_symbol[symbol_key] = self._get_sheet(income_statements)

        return sheet_by_symbol, currency_by_symbol

    def _get_sheet(self, sheets):
        f_sheets = []
        for sheet in sheets:
            tmp_sheet = self._build_aggrid_dict(sheet)
            f_sheets.append(tmp_sheet)

        values_sheet = self._convert_fiscal_date(f_sheets)

        return values_sheet

    @staticmethod
    def _convert_fiscal_date(sheets):
        values_sheet = []
        if len(sheets) > 0:
            for key in sheets[0].keys():
                if key == "Fiscal date":
                    continue

                key_values = {"sheetHierarchy": key}
                for sheet in sheets:
                    key_values[sheet["Fiscal date"]] = sheet[key]

                values_sheet.append(key_values)

        return values_sheet

    def _build_aggrid_dict(self, data: Dict, parent_group="") -> Dict[str, Any]:
        result = {}
        for key, value in data.items():
            if isinstance(value, dict):
                subdict = self._build_aggrid_dict(value, key)

                if parent_group != "":
                    new_subdict = {}
                    for old_key, old_value in subdict.items():
                        new_subdict[parent_group.capitalize().replace("_", " ") + "/" + old_key] = old_value
                    subdict = new_subdict

                result.update(subdict)
            else:
                new_key = key.capitalize().replace("_", " ")
                if parent_group != "":
                    new_key = parent_group.capitalize().replace("_", " ") + "/" + new_key

                result[new_key] = value

        return result

    def search_ticker(self, tickers_query: str, output_limit: int = 120) -> List[Dict[str, str]]:
        url = self.TWELVE_SYMBOL_SEARCH_URL_TEMPLATE.substitute(symbol=tickers_query, outputsize=output_limit)

        data = json.loads(requests.get(url).content.decode())
        symbols = data["data"]

        return symbols

    def get_economic_indicators(self, symbol: str, start_date: str, end_date: str) -> Dict[str, Any]:
        NotImplementedError("Method get_economic_indicators is not implemented for twelve data provider.")
