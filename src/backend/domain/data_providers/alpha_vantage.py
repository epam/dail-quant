import json
from datetime import datetime, timedelta
from string import Template
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import requests

from market_alerts.domain.exceptions import (
    DataNotFoundError,
    MethodIsNotImplementedError,
)

from .constants import ECONOMIC_INDICATORS, INDICATORS_PERIODICITY
from .default import DefaultDataProvider


class AlphaVantageDataProvider(DefaultDataProvider):
    PROVIDER_ID = "AV"
    PROVIDER_NAME = "AlphaVantage"

    ALPHAVANTAGE_TIMESERIES_URL_TEMPLATE = Template(
        f"https://www.alphavantage.co/query?function=TIME_SERIES_DAILY_ADJUSTED&outputsize=full&symbol=$symbol&apikey=$api_key"
    )

    ALPHAVANTAGE_FX_DAILY_URL_TEMPLATE = Template(
        "https://www.alphavantage.co/query?function=FX_DAILY&from_symbol=$from_symbol&to_symbol=$to_symbol&outputsize=full&apikey=$api_key"
    )

    ALPHAVANTAGE_EARNINGS_URL_TEMPLATE = Template(
        "https://www.alphavantage.co/query?function=EARNINGS&symbol=$symbol&apikey=$api_key"
    )

    ALPHAVANTAGE_TICKER_SEARCH_URL_TEMPLATE = Template(
        "https://www.alphavantage.co/query?function=SYMBOL_SEARCH&keywords=$symbol&apikey=$api_key"
    )

    def __init__(self, alpha_vantage_api_key: str):
        super().__init__()

        self.api_key = alpha_vantage_api_key

    def get_split_adjusted_prices(
        self, symbol: str, start_date: str, end_date: str, interval: str
    ) -> Tuple[Dict[str, Any], Optional[str]]:
        error_message = None
        if "/" in symbol:
            from_symbol, to_symbol = symbol.split("/")
            fx_url = self.ALPHAVANTAGE_FX_DAILY_URL_TEMPLATE.substitute(
                from_symbol=from_symbol, to_symbol=to_symbol, api_key=self.api_key
            )

            result = self.get_alphavantage_timeseries(symbol, fx_url, start_date, end_date, True)
        else:
            timeseries_url = self.ALPHAVANTAGE_TIMESERIES_URL_TEMPLATE.substitute(symbol=symbol, api_key=self.api_key)

            result = self.get_alphavantage_timeseries(symbol, timeseries_url, start_date, end_date, False)

            split_adjusted_prices, _ = self.transform_split_adjusted(result)

            result["values"] = split_adjusted_prices

        return result, error_message

    def get_alphavantage_timeseries(
        self, symbol: str, timeseries_url: str, start_date: str, end_date: str, is_fx: bool
    ) -> Dict[str, Any]:
        start_date = start_date.split()[0]
        end_date = end_date.split()[0]

        result = json.loads(requests.get(timeseries_url).content.decode())

        if "Error Message" in result:
            raise DataNotFoundError(
                f"Data provider failed on fetch for following symbol\: {symbol} \n With following message\: {result['Error Message']}"
            )

        if not result:
            raise DataNotFoundError(f"Data provider didn't returned any data for following symbol\: {symbol}")

        if is_fx:
            data = result["Time Series FX (Daily)"]
            symbol = "".join([result["Meta Data"]["2. From Symbol"], "/", result["Meta Data"]["3. To Symbol"]])
            timezone = result["Meta Data"]["6. Time Zone"]
        else:
            data = result["Time Series (Daily)"]
            symbol = result["Meta Data"]["2. Symbol"]
            timezone = result["Meta Data"]["5. Time Zone"]

        start_date_entry, end_date_entry = self.get_time_border_data(data, start_date, end_date)

        start_date_entry = self.convert_timeseries_entry(start_date_entry[1], start_date_entry[0])
        end_date_entry = self.convert_timeseries_entry(end_date_entry[1], end_date_entry[0])

        result = self.convert_raw_timeseries(symbol, timezone, data)

        result["values"].reverse()

        start_idx = result["values"].index(start_date_entry)
        end_idx = result["values"].index(end_date_entry) + 1
        result["values"] = result["values"][start_idx:end_idx]

        return result

    def get_economic_indicators(self, symbol: str, start_date: str, end_date: str) -> Dict[str, Any]:
        start_date = start_date.split()[0]
        end_date = end_date.split()[0]

        indicator, periodicity = symbol.split(".")

        if indicator not in ECONOMIC_INDICATORS.keys():
            return None

        av_periodicity = INDICATORS_PERIODICITY[periodicity]["av_key"]

        fetch_template: Template = ECONOMIC_INDICATORS[indicator]["av_fetch_url"]
        fetch_url = fetch_template.substitute(periodicity=av_periodicity, api_key=self.api_key)

        result = json.loads(requests.get(fetch_url).content.decode())
        if "Error Message" in result:
            raise DataNotFoundError(
                f"Data provider failed on getting economic indicators on following indicator\: {symbol} \n With following message\: {result['Error Message']}"
            )

        if not result:
            raise DataNotFoundError(f"Data provider didn't returned any data for following economic indicator\: {symbol}")

        result = self.convert_economic_indicators(result)
        tmp = {i["date"]: i for i in result["values"]}

        start_date_entry, end_date_entry = self.get_time_border_data(tmp, start_date, end_date)
        result["values"].reverse()

        start_idx = result["values"].index(start_date_entry[1])
        end_idx = result["values"].index(end_date_entry[1]) + 1

        result["values"] = result["values"][start_idx:end_idx]

        return result

    def convert_economic_indicators(self, indicator_values: Dict[str, Any]) -> Dict[str, Any]:
        result = {"values": []}

        for entry in indicator_values["data"]:
            result["values"].append(
                {
                    "date": entry["date"],
                    "close": float(entry["value"]) if entry["value"] != "." else None,
                }
            )

        return result

    def convert_raw_timeseries(self, symbol: str, timezone: str, raw_values: Dict[str, Any]) -> Dict[str, Any]:
        converted_data = {
            "meta": {
                "symbol": symbol,
                "interval": "",
                "currency": "USD",
                "exchange_timezone": timezone,
                "exchange": "",
                "mic_code": "",
                "type": "",
            },
            "values": [],
        }

        for date, data in raw_values.items():
            converted_data["values"].append(self.convert_timeseries_entry(data, date))

        return converted_data

    def get_parallel_dividend_adjusted_prices(
        self, symbols: List[str], start_date: str, end_date: str, interval: str
    ) -> Tuple[Dict[str, Any], Optional[str]]:
        symbol_to_timeseries, error_message = self.get_parallel_split_adjusted_prices(
            symbols, start_date=start_date, end_date=end_date, interval=interval
        )

        symbol_to_dividends, error_message = self.get_parallel_dividends(symbols, start_date=start_date, end_date=end_date)

        for symbol in symbols:
            prev_div_adj_factor = 1
            payment_split_adj = 0
            price_split_adj = 1

            dividents = {payment["ex_date"]: payment["amount"] for payment in symbol_to_dividends[symbol]["dividends"]}

            # Make timeseries list Desc
            symbol_to_timeseries[symbol]["values"] = symbol_to_timeseries[symbol]["values"][::-1]

            # Remove duplicates
            symbol_to_timeseries[symbol]["values"] = list(
                {dictionary["datetime"]: dictionary for dictionary in symbol_to_timeseries[symbol]["values"]}.values()
            )

            for price in symbol_to_timeseries[symbol]["values"]:
                curr_div_adj_factor = prev_div_adj_factor * (1 / (1 + payment_split_adj / price_split_adj))

                prev_div_adj_factor = curr_div_adj_factor
                payment_split_adj = dividents.get(price["datetime"], 0)
                price_split_adj = price["close"]

                price["open"] = round(float(price["open"]) * curr_div_adj_factor, 2)
                price["high"] = round(float(price["high"]) * curr_div_adj_factor, 2)
                price["low"] = round(float(price["low"]) * curr_div_adj_factor, 2)
                price["close"] = round(float(price["close"]) * curr_div_adj_factor, 2)
                price["volume"] = price["volume"] / curr_div_adj_factor if price.get("volume", None) is not None else 0

            # Make timeseries list Asc
            symbol_to_timeseries[symbol]["values"] = symbol_to_timeseries[symbol]["values"][::-1]

        return symbol_to_timeseries, error_message

    def get_dividends(self, symbol: str, start_date: str, end_date: str) -> Tuple[Dict[str, Any], Optional[str]]:
        is_fx = False
        error_message = None
        if "/" in symbol:
            is_fx = True
            return {
                "meta": {
                    "symbol": symbol,
                    "interval": "",
                    "currency": "USD",
                    "exchange_timezone": "",
                    "exchange": "",
                    "mic_code": "",
                    "type": "",
                },
                "dividends": [],
            }

        timeseries_url = self.ALPHAVANTAGE_TIMESERIES_URL_TEMPLATE.substitute(symbol=symbol, api_key=self.api_key)

        result = self.get_alphavantage_timeseries(symbol, timeseries_url, start_date, end_date, is_fx)
        if "Error Message" in result:
            raise DataNotFoundError(
                f"Data provider failed on fetch for following symbol\: {symbol} \n With following message\: {result['Error Message']}"
            )

        if not result:
            raise DataNotFoundError(f"Data provider didn't returned any data for following symbol\: {symbol}")

        _, split_adjusted_dividends = self.transform_split_adjusted(result)

        result = {"meta": result["meta"], "dividends": [div for div in split_adjusted_dividends if div["amount"] != 0.0]}

        return result, error_message

    def get_earnings(self, symbol: str, start_date: str, end_date: str) -> Dict[str, Any]:
        start_date = start_date.split()[0]
        end_date = end_date.split()[0]

        earnings_url = self.ALPHAVANTAGE_EARNINGS_URL_TEMPLATE.substitute(symbol=symbol, api_key=self.api_key)

        result = json.loads(requests.get(earnings_url).content.decode())

        if "Error Message" in result:
            raise DataNotFoundError(
                f"Data provider failed on fetch for following symbol\: {symbol}\nWith following message\: {result['Error Message']}"
            )

        if not result:
            return {
                "meta": {
                    "symbol": symbol,
                    "name": "",
                    "currency": "USD",
                    "exchange": "",
                    "mic_code": "",
                    "exchange_timezone": "",
                },
                "earnings": [],
            }

        quarterly = result["quarterlyEarnings"]
        tmp_quarterly = {item["reportedDate"]: item for item in quarterly}

        start_date_entry, end_date_entry = self.get_time_border_data(tmp_quarterly, start_date, end_date)

        quarterly.reverse()
        start_index = result["quarterlyEarnings"].index(start_date_entry[1])
        end_index = result["quarterlyEarnings"].index(end_date_entry[1]) + 1
        quarterly = quarterly[start_index:end_index]

        converted_earnings = self.convert_earnings(symbol, quarterly)

        return converted_earnings

    def convert_earnings(self, symbol: str, quarterly: List[Dict[str, str]]) -> Dict[str, Any]:
        result = {
            "meta": {"symbol": symbol, "name": "", "currency": "USD", "exchange": "", "mic_code": "", "exchange_timezone": ""},
            "earnings": [],
        }

        for earning in quarterly:
            entry = {
                "date": earning["reportedDate"],
                "time": "",
                "eps_estimate": earning["estimatedEPS"] if earning["estimatedEPS"] != "None" else None,
                "eps_actual": earning["reportedEPS"] if earning["reportedEPS"] != "None" else None,
                "difference": earning["surprise"] if earning["surprise"] != "None" else None,
                "surprise_prc": earning["surprisePercentage"] if earning["surprisePercentage"] != "None" else None,
            }

            result["earnings"].append(entry)

        return result

    def get_time_border_data(self, timeseries: Dict[str, Any], start_date: str, end_date: str) -> Tuple[Tuple[Any], Tuple[Any]]:
        date_format = "%Y-%m-%d"
        last_date = list(timeseries.keys())[-1]
        first_date = list(timeseries.keys())[0]

        if datetime.strptime(start_date, date_format) < datetime.strptime(last_date, date_format):
            start_date = last_date
        else:
            while timeseries.get(start_date, None) is None:
                start = datetime.strptime(start_date, date_format)
                start += timedelta(days=1)
                start_date = start.strftime(date_format)

        if datetime.strptime(end_date, date_format) > datetime.strptime(first_date, date_format):
            end_date = first_date
        else:
            while timeseries.get(end_date, None) is None:
                end = datetime.strptime(end_date, date_format)
                end -= timedelta(days=1)
                start_date = start.strftime(date_format)

        return (start_date, timeseries[start_date]), (end_date, timeseries[end_date])

    def convert_timeseries_entry(self, entry, date):
        conv_entry = {
            "datetime": date,
            "open": float(entry["1. open"]),
            "high": float(entry["2. high"]),
            "low": float(entry["3. low"]),
            "close": float(entry["4. close"]),
        }

        if entry.get("6. volume", None) is not None:
            conv_entry["volume"] = float(entry["6. volume"])

        if entry.get("7. dividend amount", None) is not None:
            conv_entry["dividends_amount"] = float(entry["7. dividend amount"])

        if entry.get("8. split coefficient", None) is not None:
            conv_entry["split_coef"] = float(entry["8. split coefficient"])

        return conv_entry

    def transform_split_adjusted(self, timeseries: Dict[str, Any]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        not_adjsuted_series = timeseries["values"]
        not_adjsuted_series.reverse()

        split_adj_factor = 1
        split_coef = 1

        adjusted_series = []
        adjusted_dividends = []

        for entry in not_adjsuted_series:
            split_adj_factor = split_adj_factor / split_coef
            split_coef = entry["split_coef"]

            adjusted_series.append(
                {
                    "datetime": entry["datetime"],
                    "open": entry["open"] * split_adj_factor,
                    "high": entry["high"] * split_adj_factor,
                    "low": entry["low"] * split_adj_factor,
                    "close": entry["close"] * split_adj_factor,
                    "volume": entry["volume"] / split_adj_factor,
                }
            )

            adjusted_dividends.append(
                {
                    "ex_date": entry["datetime"],
                    "amount": entry["dividends_amount"] * split_adj_factor,
                }
            )

        adjusted_series.reverse()
        adjusted_dividends.reverse()

        return adjusted_series, adjusted_dividends

    def _add_dividend_adjustment_factor_field(
        self,
        provider_id: str,
        true_symbols: Dict[str, str],
        symbol_to_dividends: Dict[str, Any],
        data_by_symbol: Dict[str, pd.DataFrame],
        dividend_adjusted: bool = False,
    ) -> Dict[str, pd.DataFrame]:
        if dividend_adjusted:
            for symbol in true_symbols.keys():
                if provider_id is not None:
                    prov_symbol = symbol + "_" + provider_id
                else:
                    prov_symbol = symbol
                data_by_symbol[prov_symbol]["div_adj_factor"] = 1

            return data_by_symbol

        for symbol, true_symbol in true_symbols.items():
            symbol_with_provider = symbol + "_" + provider_id if provider_id is not None else symbol

            prev_div_adj_factor = 1
            payment_split_adj = 0
            price_split_adj = 1

            data_dict = data_by_symbol[symbol_with_provider].to_dict(orient="index")
            timeseries = [{"datetime": str(key).split()[0], **value} for key, value in data_dict.items()]

            dividents = {payment["ex_date"]: payment["amount"] for payment in symbol_to_dividends[true_symbol]["dividends"]}

            # Make timeseries list Desc
            timeseries = timeseries[::-1]

            # Remove duplicates
            timeseries = list({dictionary["datetime"]: dictionary for dictionary in timeseries}.values())

            for price in timeseries:
                curr_div_adj_factor = prev_div_adj_factor * (1 / (1 + payment_split_adj / price_split_adj))

                price["div_adj_factor"] = curr_div_adj_factor

                prev_div_adj_factor = curr_div_adj_factor
                payment_split_adj = dividents.get(price["datetime"], 0)
                price_split_adj = price["close"]

            # Make timeseries list Asc
            timeseries = timeseries[::-1]

            df = pd.DataFrame(timeseries).set_index("datetime")
            df.index = pd.to_datetime(df.index)

            data_by_symbol[symbol_with_provider] = df

        return data_by_symbol

    def search_ticker(self, tickers_query: str, output_limit: int = None) -> List[Dict[str, str]]:
        url = self.ALPHAVANTAGE_TICKER_SEARCH_URL_TEMPLATE.substitute(symbol=tickers_query, api_key=self.api_key)

        data = json.loads(requests.get(url).content.decode())
        symbols = data["bestMatches"]

        return self._convert_symbol_search_data_to_twelve_format(symbols)

    @staticmethod
    def _convert_symbol_search_data_to_twelve_format(symbols_data: List[Dict[str, str]]) -> List[Dict[str, str]]:
        result = []

        for symbol in symbols_data:
            ticker, exchange = (
                symbol["1. symbol"].split(".") if len(symbol["1. symbol"].split(".")) > 1 else (symbol["1. symbol"], "")
            )
            result.append(
                {
                    "symbol": ticker,
                    "instrument_name": symbol["2. name"],
                    "exchange": exchange,
                    "mic_code": "",
                    "exchange_timezone": symbol["7. timezone"],
                    "instrument_type": symbol["3. type"],
                    "country": symbol["4. region"],
                    "currency": symbol["8. currency"],
                }
            )

        return result

    def get_balance_sheets(self, symbols: List[str], true_symbols: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        return {}, {}

    def get_income_statements(self, symbols: List[str], true_symbols: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        return {}, {}
