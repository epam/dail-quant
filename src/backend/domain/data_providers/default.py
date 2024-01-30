import os
from abc import ABC
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from functools import partial
from typing import Any, Callable, Dict, List, Optional, Tuple

import pandas as pd
from dateutil.relativedelta import relativedelta

from .base import BaseDataProvider
from .constants import AdditionalDividendFields, Datasets


class DefaultDataProvider(BaseDataProvider, ABC):
    PROVIDER_ID = ""
    PROVIDER_NAME = ""
    DEFAULT_CHECKED = False

    def __init__(self):
        self.data_subsets = {
            "Prices": {
                Datasets.SplitAdjustedPrices: partial(self._fetch_prices, dividend_adjusted=False),
                Datasets.SplitAndDividendAdjustedPrices: partial(self._fetch_prices, dividend_adjusted=True),
            },
            "Earnings": {
                Datasets.Earnings: self._fetch_earnings,
            },
            "Dividends": {
                Datasets.Dividends: self._add_dividends_fields,
            },
            "EconomicIndicators": {
                Datasets.EconomicIndicators: self._fetch_economic_indicators,
            },
        }
        self.progress_callback = None

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
        self.progress_callback = progress_callback

        for subset_name, subset_fetch_method in self.data_subsets["Prices"].items():
            if subset_name in datasets:
                data_by_symbol, meta_by_symbol, error_message = subset_fetch_method(
                    true_symbols, start_date, end_date, interval, primary_provider_id, meta_by_symbol, data_by_symbol
                )

                if error_message is not None:
                    return data_by_symbol, meta_by_symbol, error_message

        for subset_name, subset_fetch_method in self.data_subsets["Dividends"].items():
            if subset_name in datasets:
                data_by_symbol, error_message = subset_fetch_method(
                    additional_dividends_fields,
                    true_symbols,
                    start_date,
                    end_date,
                    primary_provider_id,
                    data_by_symbol,
                    dividend_adjusted=Datasets.SplitAndDividendAdjustedPrices in datasets,
                )

                if error_message is not None:
                    return data_by_symbol, meta_by_symbol, error_message

        for subset_name, subset_get_method in self.data_subsets["Earnings"].items():
            if subset_name in datasets:
                data_by_symbol = subset_get_method(
                    true_symbols, start_date, end_date, primary_provider_id, data_by_symbol, meta_by_symbol
                )

        for subset_name, subset_get_method in self.data_subsets["EconomicIndicators"].items():
            if subset_name in datasets:
                data_by_symbol = subset_get_method(
                    true_economic_indicators, start_date, end_date, primary_provider_id, data_by_symbol
                )

        return data_by_symbol, meta_by_symbol, error_message

    def get_parallel_dividend_adjusted_prices(
        self, symbols: List[str], start_date: str, end_date: str, interval: str
    ) -> Tuple[Dict[str, Any], Optional[str]]:
        symbol_to_timeseries, prices_error_message = self.get_parallel_split_adjusted_prices(
            symbols, start_date=start_date, end_date=end_date, interval=interval
        )

        symbol_to_dividends, dividends_error_message = self.get_parallel_dividends(
            symbols, start_date=start_date, end_date=end_date
        )

        for symbol in symbols:
            if symbol not in symbol_to_timeseries or symbol not in symbol_to_dividends:
                continue

            adjustment_coeff = 1
            prev_payment = 0
            dividents = {payment["ex_date"]: payment["amount"] for payment in symbol_to_dividends[symbol]["dividends"]}

            # Make timeseries list Desc
            symbol_to_timeseries[symbol]["values"] = symbol_to_timeseries[symbol]["values"][::-1]

            # Remove duplicates
            symbol_to_timeseries[symbol]["values"] = list(
                {dictionary["datetime"]: dictionary for dictionary in symbol_to_timeseries[symbol]["values"]}.values()
            )

            for price in symbol_to_timeseries[symbol]["values"]:
                adjustment_coeff = (1 - prev_payment / float(price["close"])) * adjustment_coeff

                price["open"] = round(float(price["open"]) * adjustment_coeff, 2)
                price["high"] = round(float(price["high"]) * adjustment_coeff, 2)
                price["low"] = round(float(price["low"]) * adjustment_coeff, 2)
                price["close"] = round(float(price["close"]) * adjustment_coeff, 2)

                prev_payment = dividents.get(price["datetime"], 0)

            # Make timeseries list Asc
            symbol_to_timeseries[symbol]["values"] = symbol_to_timeseries[symbol]["values"][::-1]

        return symbol_to_timeseries, prices_error_message if prices_error_message is not None else dividends_error_message

    def get_parallel_split_adjusted_prices(
        self, symbols: List[str], start_date: str, end_date: str, interval: str
    ) -> Tuple[Dict[str, Any], Optional[str]]:
        symbol_to_split_adjusted_prices = {}

        max_workers = os.cpu_count() * 2 - 1
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_symbol = {
                executor.submit(self.get_split_adjusted_prices, symbol, start_date, end_date, interval): symbol
                for symbol in symbols
            }
            if self.progress_callback is not None:
                for future in future_to_symbol:
                    future.add_done_callback(self.progress_callback)

            for future in as_completed(future_to_symbol):
                symbol = future_to_symbol[future]
                symbol_data, error_message = future.result()

                if symbol_data is not None:
                    symbol_to_split_adjusted_prices[symbol] = symbol_data

                if error_message is not None:
                    for future in future_to_symbol.keys():
                        future.cancel()
                    executor.shutdown(wait=False)
                    break

        return symbol_to_split_adjusted_prices, error_message

    def get_parallel_economic_indicators(self, symbols: List[str], start_date: str, end_date: str) -> Dict[str, Any]:
        symbol_to_economic_indicators = {}

        max_workers = os.cpu_count() * 2 - 1
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_symbol = {
                executor.submit(self.get_economic_indicators, symbol, start_date, end_date): symbol for symbol in symbols
            }
            if self.progress_callback is not None:
                for future in future_to_symbol:
                    future.add_done_callback(self.progress_callback)

            for future in as_completed(future_to_symbol):
                symbol = future_to_symbol[future]
                symbol_data = future.result()

                symbol_to_economic_indicators[symbol] = symbol_data

        return symbol_to_economic_indicators

    def _fetch_earnings(
        self,
        true_symbols: Dict[str, str],
        start_date: str,
        end_date: str,
        primary_provider_id: str,
        data_by_symbol: Dict[str, pd.DataFrame],
        meta_by_symbol: Dict[str, Any],
    ) -> Tuple[Dict[str, pd.DataFrame], Dict[str, Any]]:
        symbols = true_symbols.keys()

        symbol_to_earnings = self.get_parallel_earnings(true_symbols.values(), start_date, end_date)

        for symbol in symbols:
            data_earnings = symbol_to_earnings[true_symbols[symbol]]

            if self.PROVIDER_ID != primary_provider_id:
                symbol = symbol + "_" + self.PROVIDER_ID

            df = data_by_symbol.get(symbol, pd.DataFrame())

            values_earnings = data_earnings.get("earnings", None)

            if meta_by_symbol.get("meta", None) is None:
                meta_by_symbol["meta"] = data_earnings.get("meta", None)
            if values_earnings:
                df_earnings = pd.DataFrame(values_earnings).set_index("date")["eps_actual"].apply(pd.to_numeric)
                df_earnings = df_earnings[~df_earnings.index.duplicated(keep="first")]
                df_earnings.index = pd.to_datetime(df_earnings.index)
                idxes_union = df.index.union(df_earnings.index)
                idxes_union = idxes_union.sort_values()
                df = df.reindex(idxes_union).fillna(method="ffill")
                df_earnings = df_earnings.reindex(idxes_union).fillna(0)
                df_earnings = (df_earnings).dropna().cumsum()
                df["earnings"] = df_earnings

            data_by_symbol[symbol] = df

        return data_by_symbol, meta_by_symbol

    def _fetch_economic_indicators(
        self,
        true_economic_indicators: Dict[str, str],
        start_date: str,
        end_date: str,
        primary_provider_id: str,
        data_by_symbol: Dict[str, pd.DataFrame],
    ) -> Dict[str, pd.DataFrame]:
        indicator_to_data = self.get_parallel_economic_indicators(true_economic_indicators.values(), start_date, end_date)

        for indicator, true_indicator in true_economic_indicators.items():
            if indicator_to_data[true_indicator] is None:
                continue

            data_economic_inds = indicator_to_data[true_indicator]

            df = data_by_symbol.get(indicator, pd.DataFrame())

            values_economic_inds = data_economic_inds.get("values", None)

            if values_economic_inds:
                df_economic_inds = pd.DataFrame(values_economic_inds).set_index("date")["close"].apply(pd.to_numeric)
                df_economic_inds = df_economic_inds[~df_economic_inds.index.duplicated(keep="first")]
                df_economic_inds.dropna(inplace=True)
                df_economic_inds.index = pd.to_datetime(df_economic_inds.index)
                df["close"] = df_economic_inds
                df["value"] = df_economic_inds

            if self.PROVIDER_ID != primary_provider_id:
                indicator += "_" + self.PROVIDER_ID

            data_by_symbol[indicator] = df

        return data_by_symbol

    def _fetch_prices(
        self,
        true_symbols: Dict[str, str],
        start_date: str,
        end_date: str,
        interval: str,
        primary_provider_id: str,
        meta_by_symbol: Dict[str, Any],
        data_by_symbol: Dict[str, pd.DataFrame],
        dividend_adjusted: bool = False,
    ) -> Tuple[Dict[str, pd.DataFrame], Dict[str, Any], Optional[str]]:
        error_message = None
        if not true_symbols:
            return data_by_symbol, meta_by_symbol, error_message

        not_fetched_symbols = []
        not_complete_symbols = []

        # Add provider id to a symbol key and "_" if first char of symbol is digit
        true_symbols = {
            key + "_" + self.PROVIDER_ID if self.PROVIDER_ID != primary_provider_id else key: value
            for key, value in true_symbols.items()
        }
        for key, value in true_symbols.items():
            if key[0].isdigit():
                true_symbols["_" + key] = value
                del true_symbols[key]

        for symbol, true_symbol in true_symbols.items():
            # Not fetched at all yet.
            if symbol not in data_by_symbol.keys():
                not_fetched_symbols.append(symbol)
                continue

            values_df = data_by_symbol[symbol].copy(deep=True)
            is_complete, is_redundant = self.check_prices_completeness_redundancy(values_df, start_date)

            # There is lack of data that needs to be fetched in addition to data we already have.
            if not is_complete:
                not_complete_symbols.append(true_symbol)
                continue

            # Have redundant data that need to be cutted off. No need to perforn additional fetch.
            if is_redundant:
                # Redundant data cut off.
                data_by_symbol = self._cut_prices_by_dates(symbol, start_date, end_date, data_by_symbol)

            if self.progress_callback is not None:
                if not dividend_adjusted:
                    self.progress_callback(None)
                else:
                    self.progress_callback(None)
                    self.progress_callback(None)

        if not_complete_symbols:
            symbols_to_refetch, data_by_symbol, error_message = self._fetch_additional_prices(
                true_symbols, not_complete_symbols, interval, start_date, dividend_adjusted, data_by_symbol
            )
            not_fetched_symbols.extend(symbols_to_refetch)

        if not_fetched_symbols:
            # Fetch full data for symbols that wasn't already fetched.
            data_by_symbol, meta_by_symbol, error_message = self._fetch_full_prices(
                true_symbols,
                not_fetched_symbols,
                interval,
                start_date,
                end_date,
                dividend_adjusted,
                meta_by_symbol,
                data_by_symbol,
            )

        return data_by_symbol, meta_by_symbol, error_message

    @staticmethod
    def _cut_prices_by_dates(
        symbol: str, start_date: str, end_date: str, data_by_symbol: Dict[str, pd.DataFrame]
    ) -> Dict[str, pd.DataFrame]:
        start_border = pd.Timestamp(start_date.split()[0])
        end_border = pd.Timestamp(end_date.split()[0])

        df = data_by_symbol[symbol]

        data_by_symbol[symbol] = df.loc[start_border:end_border]

        return data_by_symbol

    def _fetch_additional_prices(
        self,
        true_symbols: Dict[str, str],
        not_complete_symbols: List[str],
        interval: str,
        start_date: str,
        dividend_adjusted: bool,
        data_by_symbol: Dict[str, pd.DataFrame],
    ) -> Tuple[List[str], Dict[str, pd.DataFrame], Optional[str]]:
        # Reverse true symbols to map true symbol for fetch on symbol for data_by_symbol
        reversed_true_symbols = {value: key for key, value in true_symbols.items()}

        not_fetched_symbols = []
        start_dates = [data_by_symbol[reversed_true_symbols[symbol]].iloc[0].name for symbol in not_complete_symbols]
        # Taking latest start date from all symbols and add 2 days to it so we have couple of prices overlapping
        # to check if they are the same in data_by_symbol we allready have and data we fetched additionaly
        start_datetime_with_overlap = max(start_dates) + pd.DateOffset(days=2)
        start_date_with_overlap = str(start_datetime_with_overlap).split()[0]

        if dividend_adjusted:
            symbol_to_additional_data, error_message = self.get_parallel_dividend_adjusted_prices(
                not_complete_symbols,
                start_date,
                start_date_with_overlap,
                interval,
            )
        else:
            symbol_to_additional_data, error_message = self.get_parallel_split_adjusted_prices(
                not_complete_symbols,
                start_date,
                start_date_with_overlap,
                interval,
            )

        for symbol, additional_data in symbol_to_additional_data.items():
            if not additional_data["values"]:
                continue

            if symbol[0].isdigit():
                symbol = "_" + symbol

            # Take first date from data_by_symbol to compare to newly fetched data
            check_datetime = data_by_symbol[reversed_true_symbols[symbol]].iloc[0].name
            additional_data_df = self._convert_provider_prices_to_df(additional_data)

            # Rows to compare
            original_row = data_by_symbol[reversed_true_symbols[symbol]].loc[[check_datetime]]

            try:
                additional_row = additional_data_df.loc[[check_datetime]]
            except KeyError:
                continue

            common_cols = [col for col in list(original_row.columns) if col in additional_row.columns]
            original_row = original_row[common_cols]

            for column in original_row.columns:
                additional_row[column] = additional_row[column].astype(original_row[column].dtype)

            # If they are the same adding fetched data, else need to refetch for that symbol
            if original_row.equals(additional_row):
                full_timeseries = pd.concat([additional_data_df, data_by_symbol[reversed_true_symbols[symbol]]])
                full_timeseries = full_timeseries.loc[~full_timeseries.index.duplicated(keep="first")]

                data_by_symbol[reversed_true_symbols[symbol]] = full_timeseries
            else:
                not_fetched_symbols.append(reversed_true_symbols[symbol])

        return not_fetched_symbols, data_by_symbol, error_message

    def _fetch_full_prices(
        self,
        true_symbols: Dict[str, str],
        not_fetched_symbols: List[str],
        interval: str,
        start_date: str,
        end_date: str,
        dividend_adjusted: bool,
        meta_by_symbol: Dict[str, Any],
        data_by_symbol: Dict[str, pd.DataFrame],
    ) -> Tuple[Dict[str, pd.DataFrame], Dict[str, Any], Optional[str]]:
        if dividend_adjusted:
            symbol_to_fully_fetched_data, error_message = self.get_parallel_dividend_adjusted_prices(
                [true_symbols[s] for s in not_fetched_symbols],
                start_date,
                end_date,
                interval,
            )
        else:
            symbol_to_fully_fetched_data, error_message = self.get_parallel_split_adjusted_prices(
                [true_symbols[s] for s in not_fetched_symbols],
                start_date,
                end_date,
                interval,
            )

        for symbol in not_fetched_symbols:
            if true_symbols[symbol] not in symbol_to_fully_fetched_data:
                continue

            data_prices = symbol_to_fully_fetched_data[true_symbols[symbol]]
            meta = data_prices.get("meta", None)

            df = self._convert_provider_prices_to_df(data_prices)

            data_by_symbol[symbol] = df
            meta_by_symbol[symbol] = meta

        return data_by_symbol, meta_by_symbol, error_message

    @staticmethod
    def _convert_provider_prices_to_df(data_prices: Dict[str, Any]) -> pd.DataFrame:
        values = data_prices.get("values", None)

        if len(values) > 0:
            df = pd.DataFrame(values).set_index("datetime").apply(pd.to_numeric)
        else:
            df = pd.DataFrame(values)

        df = df[~df.index.duplicated(keep="first")]
        df.index = pd.to_datetime(df.index)
        if not "volume" in df:
            df["volume"] = 1.0

        return df

    def _add_dividends_fields(
        self,
        additional_dividends_fields: List[str],
        true_symbols: Dict[str, str],
        start_date: str,
        end_date: str,
        primary_provider_id: str,
        data_by_symbol: Dict[str, pd.DataFrame],
        dividend_adjusted: bool = False,
    ) -> Tuple[Dict[str, pd.DataFrame], Optional[str]]:
        if not additional_dividends_fields:
            return data_by_symbol

        provider_id = self.PROVIDER_ID if self.PROVIDER_ID != primary_provider_id else None

        start_date = str(pd.to_datetime(start_date) - pd.DateOffset(years=2)).split()[0]
        end_date = str(pd.to_datetime(end_date) + pd.DateOffset(years=2)).split()[0]

        symbol_to_dividends, error_message = self.get_parallel_dividends(
            true_symbols.values(), start_date=start_date, end_date=end_date
        )

        if AdditionalDividendFields.DividendAmount in additional_dividends_fields:
            data_by_symbol = self._add_dividend_amount_field(provider_id, true_symbols, symbol_to_dividends, data_by_symbol)

        if AdditionalDividendFields.DividendAdjustmentFactor in additional_dividends_fields:
            data_by_symbol = self._add_dividend_adjustment_factor_field(
                provider_id, true_symbols, symbol_to_dividends, data_by_symbol, dividend_adjusted
            )

        add_trailing_yield = AdditionalDividendFields.Trailing12MonthDividendYield in additional_dividends_fields
        add_forward_yield = AdditionalDividendFields.ForwardDividendYield in additional_dividends_fields

        if add_trailing_yield or add_forward_yield:
            data_by_symbol = self._add_dividend_yield_fields(
                provider_id, true_symbols, symbol_to_dividends, data_by_symbol, add_forward_yield, add_trailing_yield
            )

        return data_by_symbol, error_message

    def get_parallel_dividends(self, symbols: List[str], start_date: str, end_date: str) -> Tuple[Dict[str, Any], Optional[str]]:
        symbol_to_dividends = {}
        error_message = None

        max_workers = os.cpu_count() * 2 - 1
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_symbol = {executor.submit(self.get_dividends, symbol, start_date, end_date): symbol for symbol in symbols}

            if self.progress_callback is not None:
                for future in future_to_symbol:
                    future.add_done_callback(self.progress_callback)

            for future in as_completed(future_to_symbol):
                symbol = future_to_symbol[future]
                symbol_data, error_message = future.result()

                if symbol_data is not None:
                    symbol_to_dividends[symbol] = symbol_data
                if error_message is not None:
                    for future in future_to_symbol.keys():
                        future.cancel()
                    executor.shutdown(wait=False)
                    break

        return symbol_to_dividends, error_message

    @staticmethod
    def _add_dividend_amount_field(
        provider_id: str,
        true_symbols: Dict[str, str],
        symbol_to_dividends: Dict[str, Any],
        data_by_symbol: Dict[str, pd.DataFrame],
    ) -> Dict[str, pd.DataFrame]:
        for symbol, true_symbol in true_symbols.items():
            symbol_with_provider = symbol + "_" + provider_id if provider_id is not None else symbol

            if not true_symbol in symbol_to_dividends or not symbol_with_provider in data_by_symbol:
                continue

            dividend_values = {payment["ex_date"]: payment["amount"] for payment in symbol_to_dividends[true_symbol]["dividends"]}

            data_dict = data_by_symbol[symbol_with_provider].to_dict(orient="index")
            timeseries = [{"datetime": str(key).split()[0], **value} for key, value in data_dict.items()]

            for entry in timeseries:
                div_amt = dividend_values.get(entry["datetime"], 0.0)
                entry["div_amt"] = div_amt

            timeseries_df = pd.DataFrame(timeseries).set_index("datetime")
            timeseries_df.index = pd.to_datetime(timeseries_df.index)

            data_by_symbol[symbol_with_provider] = timeseries_df

        return data_by_symbol

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
                symbol_with_provider = symbol + "_" + provider_id if provider_id is not None else symbol
                if symbol_with_provider not in data_by_symbol:
                    continue
                data_by_symbol[symbol_with_provider]["div_adj_factor"] = 1

            return data_by_symbol

        for symbol, true_symbol in true_symbols.items():
            symbol_with_provider = symbol + "_" + provider_id if provider_id is not None else symbol

            if symbol_with_provider not in data_by_symbol or true_symbol not in symbol_to_dividends:
                continue

            data_dict = data_by_symbol[symbol_with_provider].to_dict(orient="index")
            timeseries = [{"datetime": str(key).split()[0], **value} for key, value in data_dict.items()]

            adj_coeff = 1
            prev_payment = 0
            dividents = {payment["ex_date"]: payment["amount"] for payment in symbol_to_dividends[true_symbol]["dividends"]}

            # Make timeseries list Desc
            timeseries = timeseries[::-1]

            # Remove duplicates
            timeseries = list({dictionary["datetime"]: dictionary for dictionary in timeseries}.values())

            for price in timeseries:
                adj_coeff = (1 - prev_payment / float(price["close"])) * adj_coeff

                price["div_adj_factor"] = adj_coeff

                prev_payment = dividents.get(price["datetime"], 0)

            # Make timeseries list Asc
            timeseries = timeseries[::-1]

            df = pd.DataFrame(timeseries).set_index("datetime")
            df.index = pd.to_datetime(df.index)

            data_by_symbol[symbol_with_provider] = df

        return data_by_symbol

    def _add_dividend_yield_fields(
        self,
        provider_id: str,
        true_symbols: Dict[str, str],
        symbol_to_dividends: Dict[str, Any],
        data_by_symbol: Dict[str, pd.DataFrame],
        forward_div_yield: bool = False,
        trail_12mo_div_yield: bool = False,
    ) -> Dict[str, pd.DataFrame]:
        if not forward_div_yield and not trail_12mo_div_yield:
            return data_by_symbol

        eps_days = 10

        for symbol, true_symbol in true_symbols.items():
            symbol_with_provider = symbol + "_" + provider_id if provider_id is not None else symbol

            if symbol_with_provider not in data_by_symbol or true_symbol not in symbol_to_dividends:
                continue

            dividend_values = {
                entry["ex_date"]: {
                    "amount": entry["amount"],
                    "date": entry["ex_date"],
                    "datetime": datetime.strptime(entry["ex_date"], "%Y-%m-%d"),
                }
                for entry in symbol_to_dividends[true_symbol]["dividends"]
            }

            dividend_values = self._add_expected_dividend_frequency(dividend_values)

            prices = data_by_symbol[symbol_with_provider].to_dict(orient="index")
            prices = self._add_string_date_to_prices(prices)

            div_sums_by_date = {entry["datetime"]: None for entry in prices}

            div_sum = 0

            div_yield = self._calculate_initial_yield(
                dividend_values, datetimes=[i["datetime"] for i in prices], eps_days=eps_days
            )

            for payment in div_yield:
                div_sum += payment["amount"]

            div_yield_sum = div_sum

            for entry in prices:
                curr_price_date = datetime.strptime(entry["datetime"], "%Y-%m-%d")
                eps1_start, yield_start, eps2_start = self._get_dividend_yield_dates_with_epsilon(curr_price_date, eps_days)

                div_payment = dividend_values.get(entry["datetime"], None)

                div_sum += div_payment["amount"] if div_payment is not None else 0
                div_sums_by_date[entry["datetime"]] = div_sum

                out_div_payment = None
                first_yield_payment = div_yield[0] if div_yield else None

                if first_yield_payment and (first_yield_payment["datetime"] < eps2_start):
                    out_div_payment = div_yield.pop(0)

                if div_payment is not None:
                    div_yield.append(div_payment)
                    if (
                        first_yield_payment
                        and first_yield_payment["datetime"] >= eps2_start
                        and first_yield_payment["datetime"] < yield_start
                        and div_payment["datetime"] > eps1_start
                    ):
                        out_div_payment = div_yield.pop(0)

                latest_div_expected_frequency = div_yield[-1]["expected_frequency"] if div_yield else 0

                if len(div_yield) > latest_div_expected_frequency:
                    out_div_payment = div_yield.pop(0)

                if out_div_payment is not None and out_div_payment["date"] in div_sums_by_date.keys():
                    out_div_sum = div_sums_by_date[out_div_payment["date"]]
                    div_yield_sum = div_sum - out_div_sum

                entry = self._add_dividend_yields_into_prices(
                    entry, div_yield_sum, div_yield, latest_div_expected_frequency, trail_12mo_div_yield, forward_div_yield
                )

            data_by_symbol = self._insert_prices_into_data_by_symbol(prices, data_by_symbol, symbol_with_provider)

        return data_by_symbol

    def _add_expected_dividend_frequency(self, dividends: Dict[str, Any]):
        if dividends:
            dividend_datetimes = [i["datetime"] for i in dividends.values()]

            if len(dividend_datetimes) == 1:
                dividends[list(dividends.keys())[0]]["expected_frequency"] = 1

                return dividends

            payment_after_last_dividend_date = dividend_datetimes[0] + (dividend_datetimes[0] - dividend_datetimes[1])
            payment_before_first_dividend_date = dividend_datetimes[-1] - (dividend_datetimes[-2] - dividend_datetimes[-1])

            for index, (_, value) in enumerate(dividends.items()):
                if index == 0:
                    day_difference_next_payment = payment_after_last_dividend_date - value["datetime"]
                    day_difference_previous_payment = value["datetime"] - dividend_datetimes[index + 1]
                elif index == len(dividend_datetimes) - 1:
                    day_difference_next_payment = dividend_datetimes[index - 1] - value["datetime"]
                    day_difference_previous_payment = value["datetime"] - payment_before_first_dividend_date
                else:
                    day_difference_previous_payment = value["datetime"] - dividend_datetimes[index + 1]
                    day_difference_next_payment = dividend_datetimes[index - 1] - value["datetime"]

                expected_frequency = self._get_expected_frequency(
                    day_difference_next_payment=day_difference_next_payment.days,
                    day_difference_previous_payment=day_difference_previous_payment.days,
                )

                value["expected_frequency"] = expected_frequency

        return dividends

    @staticmethod
    def _get_expected_frequency(day_difference_previous_payment: int, day_difference_next_payment: int):
        if day_difference_previous_payment < 400:
            accurate_frequency = 365 / day_difference_previous_payment
        else:
            accurate_frequency = 365 / day_difference_next_payment

        if accurate_frequency > 7:
            return 12
        elif accurate_frequency > 2.5:
            return 4
        elif accurate_frequency > 1.5:
            return 2

        return 1

    @staticmethod
    def _add_string_date_to_prices(prices: Dict[str, Any]) -> Dict[str, Any]:
        return [{"datetime": str(key).split()[0], **value} for key, value in prices.items()]

    def _calculate_initial_yield(self, dividend_values: Dict[str, Any], datetimes: List[str], eps_days: int = 10) -> List[Any]:
        div_yield = []
        first_eps_payments = []
        last_eps_payments = []

        first_price_date = datetime.strptime(datetimes[0], "%Y-%m-%d") - relativedelta(days=1)
        first_yield_end = first_price_date
        first_eps1_start, first_yield_start, first_eps2_start = self._get_dividend_yield_dates_with_epsilon(
            first_price_date, eps_days
        )

        for div_payment in list(dividend_values.values())[::-1]:
            if div_payment["datetime"] >= first_eps1_start and div_payment["datetime"] <= first_yield_end:
                first_eps_payments.append(div_payment)
            elif div_payment["datetime"] >= first_yield_start and div_payment["datetime"] <= first_eps1_start:
                div_yield.append(div_payment)
            elif (
                len(first_eps_payments) == 0
                and div_payment["datetime"] >= first_eps2_start
                and div_payment["datetime"] <= first_yield_start
            ):
                last_eps_payments.append(div_payment)

        div_yield = first_eps_payments + div_yield + last_eps_payments
        return div_yield

    @staticmethod
    def _get_dividend_yield_dates_with_epsilon(curr_price_date: datetime, eps_days: int) -> Tuple[datetime, datetime, datetime]:
        eps1_start = curr_price_date - relativedelta(days=eps_days)
        yield_start = curr_price_date - relativedelta(years=1) + relativedelta(days=eps_days)
        eps2_start = yield_start - relativedelta(days=eps_days * 2)
        return eps1_start, yield_start, eps2_start

    @staticmethod
    def _add_dividend_yields_into_prices(
        price_entry: Dict[str, Any],
        div_yield_sum: float,
        div_yield: List[Dict[str, Any]],
        latest_div_expected_frequency: int,
        trail_12mo_div_yield: bool,
        forward_div_yield: bool,
    ) -> Dict[str, Any]:
        if trail_12mo_div_yield:
            price_entry["trail_12mo_div_yield"] = div_yield_sum / price_entry["close"]
        if forward_div_yield:
            latest_div_amount = div_yield[-1]["amount"] if div_yield else 0
            price_entry["forward_div_yield"] = latest_div_amount * latest_div_expected_frequency / price_entry["close"]

        return price_entry

    @staticmethod
    def _insert_prices_into_data_by_symbol(
        prices: Dict[str, Any], data_by_symbol: Dict[str, pd.DataFrame], symbol_with_provider: str
    ) -> Dict[str, pd.DataFrame]:
        prices_df = pd.DataFrame(prices).set_index("datetime")
        prices_df.index = pd.to_datetime(prices_df.index)
        data_by_symbol[symbol_with_provider] = prices_df

        return data_by_symbol

    def get_parallel_earnings(self, symbols: List[str], start_date: str, end_date: str) -> Dict[str, Any]:
        symbol_to_earnings = {}

        max_workers = os.cpu_count() * 2 - 1
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_symbol = {executor.submit(self.get_earnings, symbol, start_date, end_date): symbol for symbol in symbols}

            for future in as_completed(future_to_symbol):
                symbol = future_to_symbol[future]
                symbol_data = future.result()

                symbol_to_earnings[symbol] = symbol_data

        return symbol_to_earnings

    @staticmethod
    def check_prices_completeness_redundancy(prices: pd.DataFrame, start_date: str) -> Tuple[bool, bool]:
        """
        Checks if data complete, and if it is, the function checks if timeseries have redundant data.

        Parameters:
            data (dict) - Data dict to check.\n
            start_date (str) - Desired data start date.

        Returns:
            is_complete (bool) - Value that defines if data shoud be fetched additionaly to data we already have.\n
            is_redundant (bool) - Value that defines if data dict have redundant data that shoud be cutted out.
        """
        is_complete = False
        is_redundant = False

        desired_start_datetime = pd.to_datetime(start_date.split()[0])
        series_start_datetime = pd.to_datetime(prices.iloc[0].name)

        while desired_start_datetime.day_name() in ["Saturday", "Sunday"]:
            desired_start_datetime += pd.DateOffset(1)

        if desired_start_datetime >= series_start_datetime:
            is_complete = True
            if desired_start_datetime > series_start_datetime:
                is_redundant = True

        return is_complete, is_redundant
