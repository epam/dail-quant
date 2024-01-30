import re
from copy import deepcopy
from functools import partial
from typing import Any, Callable, Dict, List, Optional, Tuple

import pandas as pd

from market_alerts.domain.exceptions import ProviderAPILimitsError


def unify_indexes(data_by_symbol: Dict[str, Any], time_line: Optional[Any] = None) -> Dict[str, Any]:
    if time_line is not None:
        for key in data_by_symbol:
            data_by_symbol[key] = data_by_symbol[key].reindex(time_line).fillna(method="ffill")
        return data_by_symbol
    else:
        idxes_union = pd.Index([])
        for key in data_by_symbol:
            idxes_union = idxes_union.union(data_by_symbol[key].index)
        idxes_union = idxes_union.sort_values()
        for key in data_by_symbol:
            data_by_symbol[key] = data_by_symbol[key].reindex(idxes_union).fillna(method="ffill")
        return data_by_symbol


def get_sparse_dividends(
    time_line,
    symbol: str,
    true_symbols: List[str],
    div_start_date: str,
    div_end_date: str,
    provider,
) -> Dict[str, Any]:
    data_dividends, error_message = provider.get_dividends(
        symbol=true_symbols[symbol],
        start_date=div_start_date,
        end_date=div_end_date,
    )

    if data_dividends is None and error_message is not None:
        raise ProviderAPILimitsError(error_message)

    values_dividends = data_dividends.get("dividends", None)
    if values_dividends:
        df_dividends = pd.DataFrame(values_dividends).set_index("ex_date").apply(pd.to_numeric)["amount"]
        df_dividends = df_dividends[~df_dividends.index.duplicated(keep="first")]
        df_dividends.index = pd.to_datetime(df_dividends.index)
        df_dividends = df_dividends.sort_index().dropna().cumsum()
        df_dividends = unify_indexes({symbol: df_dividends}, time_line)[symbol]
        df_dividends = df_dividends.fillna(0.0) - df_dividends.shift(1).fillna(0.0)
    else:
        df_dividends = pd.Series(index=time_line)
        df_dividends[:] = 0.0
    return df_dividends


def get_fx_rate(time_line, fx_rate_symbol: str, start_date: str, end_date: str, interval: str, provider) -> pd.DataFrame:
    data_prices, error_message = provider.get_split_adjusted_prices(
        symbol=fx_rate_symbol,
        start_date=start_date,
        end_date=end_date,
        interval=interval,
    )

    if data_prices is None and error_message is not None:
        raise ProviderAPILimitsError(error_message)
    
    values = data_prices.get("values", None)

    df = pd.DataFrame(values).set_index("datetime").apply(pd.to_numeric)
    df = df[~df.index.duplicated(keep="first")]
    df.index = pd.to_datetime(df.index)

    df = df.reindex(time_line).fillna(method="ffill")
    return df


def get_globals() -> Dict[str, Any]:
    import builtins

    import numpy as np
    import pandas as pd
    import sklearn as sk
    import ta

    return {
        "ta": ta,
        "pd": pd,
        "pandas": pd,
        "np": np,
        "numpy": np,
        "sk": sk,
        "sklearn": sk,
        "locals": builtins.locals,
    }


def redirect_print(print_log, *messages):
    print_log.append(" ".join([str(message) for message in messages]))


def get_print_redirect() -> Tuple[Callable, List[str]]:
    print_log = []

    bound_redirect_print = partial(redirect_print, print_log)

    return bound_redirect_print, print_log


def parse_tickers(query: str, primary_provider_id: str) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
    symbols, supp_symbols, synth_formulas, supp_synth_formulas = [], [], {}, {}
    available_operations = ["*", "/", "+", "-"]
    special_symbols = [":"]
    special_symbols_not_to_remove = ["@", "."]
    quote_special_symbols = ["/"]
    quotes = ['"', "'", "‘", "’", "`"]
    quote_counter = 0

    record_symbol = ""

    for item in [i.strip() for i in query.split(",")]:
        is_supp = check_if_supplementary(item)

        if is_supp:
            item = remove_supp_parentheses(item)

        item_true_symbols = []
        item_symbols = {}
        item_supp_symbols = {}
        is_synth = False

        if "=" in item:
            synth_name, synth_formula = item.split("=")
        else:
            synth_name, synth_formula = None, item

        for char in synth_formula:
            if char in quotes:
                if quote_counter == 0:
                    quote_counter = 1
                    continue
                else:
                    quote_counter = 0
                    item_true_symbols.append(record_symbol)
                    record_symbol = ""
                    continue

            if quote_counter == 1:
                record_symbol += char
                continue

            if (
                char.isalpha()
                or char.isdigit()
                or char in special_symbols
                or char in special_symbols_not_to_remove
                or quote_counter == 1
                and char in quote_special_symbols
            ):
                record_symbol += char
                continue

            if char in available_operations and quote_counter == 0:
                is_synth = True

            if char in available_operations and record_symbol:
                if check_digit_ticker(record_symbol):
                    item_true_symbols.append(record_symbol)
                record_symbol = ""

        if record_symbol != "":
            item_true_symbols.append(record_symbol)
            record_symbol = ""

        for quote in quotes:
            synth_formula = synth_formula.replace(quote, "")

        for true_symbol in item_true_symbols:
            # if "@" not in true_symbol:
            #     true_symbol_with_provider = f"{true_symbol}@{primary_provider_id}"
            #     # Find all tickers without @ symbol and replace them with ticker@primary_provider_id
            #     synth_formula = re.compile(f"{true_symbol}(?!@)").sub(true_symbol_with_provider, synth_formula)
            #     true_symbol = true_symbol_with_provider

            tmp = true_symbol

            for symbol in special_symbols:
                tmp = tmp.replace(symbol, "")
            for symbol in quote_special_symbols:
                tmp = tmp.replace(symbol, "")
            if tmp[0].isdigit():
                tmp = "_" + tmp

            tmp = tmp.replace("@", "_")
            synth_formula = synth_formula.replace(true_symbol, tmp)

            if is_supp:
                item_supp_symbols[true_symbol] = tmp
            else:
                item_symbols[true_symbol] = tmp

        if is_synth:
            if synth_name is None:
                synth_name = "".join(item_symbols.values())

            if is_supp:
                supp_synth_formulas[synth_name.strip()] = synth_formula.strip()
            else:
                synth_formulas[synth_name.strip()] = synth_formula.strip()

        symbols.extend(item_symbols.keys())
        supp_symbols.extend(item_supp_symbols.keys())

    symbols = list(dict.fromkeys(symbols))
    supp_symbols = list(dict.fromkeys(supp_symbols))
    provider_to_ticker = get_symbol_providers(symbols, primary_provider_id)
    provider_to_supp_ticker = get_symbol_providers(supp_symbols, primary_provider_id)

    all_symbols = {
        "tradable": symbols,
        "supplementary": supp_symbols,
    }

    all_synth_formulas = {
        "tradable": synth_formulas,
        "supplementary": supp_synth_formulas,
    }

    all_provider_to_ticker = {
        "tradable": provider_to_ticker,
        "supplementary": provider_to_supp_ticker,
    }

    return all_symbols, all_synth_formulas, all_provider_to_ticker


def check_if_supplementary(query: str) -> bool:
    query = query.strip()
    match = re.match(r"^SUPP\(.*\)$", query)
    return bool(match)


def remove_supp_parentheses(string):
    return re.sub(r"(^SUPP\()|(\)$)", "", string)


def get_symbol_providers(symbols: List[str], primary_provider_id: str) -> Dict[str, str]:
    provider_to_symbol = {}

    for symbol in symbols:
        if "@" not in symbol:
            if primary_provider_id in provider_to_symbol:
                provider_to_symbol[primary_provider_id]["true_symbols"].append(symbol)
            else:
                provider_to_symbol[primary_provider_id] = {"true_symbols": [symbol]}

            continue

        ticker, provider = symbol.split("@")
        if provider in provider_to_symbol:
            provider_to_symbol[provider]["true_symbols"].append(ticker)
        else:
            provider_to_symbol[provider] = {"true_symbols": [ticker]}

    return provider_to_symbol


def check_digit_ticker(record_symbol: str) -> bool:
    for char in record_symbol:
        if char.isalpha():
            return True
    return False


def merge_provider_to_ticker(tradable: Dict[str, Any], supplementary: Dict[str, Any]) -> Dict[str, Any]:
    result = deepcopy(tradable)
    all_keys = dict.fromkeys(list(tradable.keys()) + list(supplementary.keys()))

    for key in all_keys:
        if key not in supplementary.keys():
            continue

        if key not in result:
            result[key] = deepcopy(supplementary[key])
            continue

        result[key]["true_symbols"] = list(dict.fromkeys(result[key]["true_symbols"] + supplementary[key]["true_symbols"]))

    return result


def insert_economic_indicators(indicators_to_provider: Dict[str, Any], all_symbols_to_provider: Dict[str, Any]) -> Dict[str, Any]:
    result = deepcopy(all_symbols_to_provider)

    for provider_id, inds in indicators_to_provider.items():
        if provider_id in result:
            result[provider_id]["economic_indicators"] = inds["true_symbols"]
        else:
            result[provider_id] = {"true_symbols": [], "economic_indicators": inds["true_symbols"]}

    return result


def get_default_calendar(start_date: str, end_date: str) -> pd.DatetimeIndex:
    defualt_calendar = pd.date_range(start_date, end_date, freq="D").round("D")
    defualt_calendar = defualt_calendar[defualt_calendar.dayofweek < 5]
    defualt_calendar = defualt_calendar[(~defualt_calendar.month.isin([1])) | (~defualt_calendar.day.isin([1]))]
    defualt_calendar = defualt_calendar[(~defualt_calendar.month.isin([12])) | (~defualt_calendar.day.isin([25]))]
    return defualt_calendar
