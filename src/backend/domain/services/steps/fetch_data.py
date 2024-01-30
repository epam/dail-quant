from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, Tuple

import pandas as pd
from pandas import DatetimeIndex

from market_alerts.containers import data_periodicities, data_providers, data_timeranges
from market_alerts.domain.constants import LIST_SYMBOLS
from market_alerts.domain.services.steps.utils import (
    get_default_calendar,
    get_symbol_providers,
    insert_economic_indicators,
    merge_provider_to_ticker,
    parse_tickers,
    unify_indexes,
)


def symbol_step(
    session_dict: Dict[str, Any], progress_callback: Optional[Callable[[None], None]] = None
) -> Tuple[str, List[str], Dict[str, Any], Optional[str]]:
    ##################
    session_dict["sparse_symbols"] = []
    ##################
    primary_provider_id = session_dict["data_provider"]

    session_dict["true_economic_indicators"] = get_symbol_providers(
        session_dict.get("economic_indicators", []), primary_provider_id
    )

    all_symbols, all_synth_formulas, all_provider_to_ticker = parse_tickers(
        session_dict["tradable_symbols_prompt"], primary_provider_id
    )

    true_tradable_symbols = all_symbols["tradable"]
    synth_formulas_to_trade = all_synth_formulas["tradable"]
    session_dict["synth_formulas_to_trade"] = tuple(synth_formulas_to_trade)

    provider_to_tradable_symbols = all_provider_to_ticker["tradable"]

    true_supplementary_symbols = all_symbols["supplementary"]
    synth_formulas_not_to_trade = all_synth_formulas["supplementary"]
    session_dict["synth_formulas_not_to_trade"] = tuple(synth_formulas_not_to_trade)

    provider_to_supplementary_symbols = all_provider_to_ticker["supplementary"]

    all_provider_to_symbols = merge_provider_to_ticker(provider_to_tradable_symbols, provider_to_supplementary_symbols)
    all_provider_to_symbols = insert_economic_indicators(session_dict["true_economic_indicators"], all_provider_to_symbols)

    true_symbols: List = true_tradable_symbols + true_supplementary_symbols

    for provider_id, entry in provider_to_tradable_symbols.items():
        provider_to_tradable_symbols[provider_id]["symbols"] = [true_symbol_to_symbol(i) for i in entry["true_symbols"]]

    synth_formulas = synth_formulas_to_trade | synth_formulas_not_to_trade
    session_dict["fetched_data"] = True

    session_dict["symbols"] = [true_symbol_to_symbol(i) for i in true_symbols]
    session_dict["tradable_symbols"] = [true_symbol_to_symbol(i) for i in true_tradable_symbols]
    session_dict["supplementary_symbols"] = [true_symbol_to_symbol(i) for i in true_supplementary_symbols]
    session_dict["true_symbols"] = dict(zip(session_dict["symbols"], true_symbols))
    session_dict["provider_to_tradable_symbols"] = provider_to_tradable_symbols
    session_dict["provider_to_supplementary_symbols"] = provider_to_supplementary_symbols

    data_timerange = data_timeranges[session_dict["time_period"]]

    request_end_date = datetime.now()
    request_start_date = request_end_date - data_timerange["value"]
    session_dict["end_date"] = request_end_date.strftime("%Y-%m-%d %H:%M:%S")
    session_dict["start_date"] = request_start_date.strftime("%Y-%m-%d %H:%M:%S")

    data_periodicity = data_periodicities[session_dict["interval"]]["value"]

    (
        session_dict["data_by_symbol"],
        meta_by_symbol,
        session_dict["dividends_currency_match"],
        session_dict["time_line"],
        error_message,
    ) = get_data(
        session_dict["start_date"],
        session_dict["end_date"],
        data_periodicity,
        session_dict["datasets_keys"],
        all_provider_to_symbols,
        primary_provider_id,
        session_dict.get("data_by_symbol", {}),
        session_dict.get("meta", {}),
        session_dict["tradable_symbols"],
        session_dict["dividend_fields"],
        progress_callback,
    )

    session_dict["meta"] = meta_by_symbol
    session_dict["economic_indicator_symbols"] = [
        key for key in session_dict["data_by_symbol"] if "value" in session_dict["data_by_symbol"][key].columns
    ]

    session_dict["meta_by_symbol"] = {key: get_meta_data(meta_by_symbol[key], LIST_SYMBOLS) for key in meta_by_symbol}

    code_to_get_symbols = "\n".join(
        ["%s = session_dict['data_by_symbol']['%s']" % (key, key) for key in session_dict["data_by_symbol"]]
    )
    code_to_get_synthetics = "\n".join(f"{synth_name} = {synth_formula}" for synth_name, synth_formula in synth_formulas.items())

    code_to_exec = """
import ta
import pandas as pd
%s
%s
""" % (
        code_to_get_symbols,
        code_to_get_synthetics,
    )
    exec(code_to_exec)

    data_by_synth = {}
    for synth_name in synth_formulas:
        data_by_synth[synth_name] = locals()[synth_name]

    session_dict["synths"] = list(synth_formulas)
    session_dict["data_by_synth"] = data_by_synth
    session_dict["synth_formulas"] = synth_formulas

    fetched_symbols_meta = []
    for meta in meta_by_symbol.values():
        if meta is None:
            continue
        symbol = meta["symbol"]
        exchange = meta.get("exchange", "")
        name = ""
        for item in LIST_SYMBOLS:
            if item["symbol"] == symbol and item.get("exchange", "") == exchange:
                name = item.get("name", "")

        fetched_symbols_meta.append(f"{symbol}&#x3A;{exchange} | {name}")

    request_timestamp = request_end_date.strftime("%d/%m/%Y %H:%M:%S")

    session_dict.setdefault("symbol_to_currency", {})

    for s in session_dict["symbols"]:
        if s not in session_dict["true_symbols"]:
            continue
        if "/" in session_dict["true_symbols"][s]:
            session_dict["symbol_to_currency"][s] = session_dict["true_symbols"][s]
        else:
            session_dict["symbol_to_currency"][s] = session_dict["meta"][s]["currency"]

    for provider_id, entry in all_provider_to_symbols.items():
        ind_symbols = ["_".join([true_symbol_to_symbol(i), provider_id]) for i in entry.get("economic_indicators", [])]
        session_dict["symbols"].extend(ind_symbols)

    return request_timestamp, fetched_symbols_meta, synth_formulas, error_message


def get_data(
    start_date: str,
    end_date: str,
    interval: str,
    datasets: List[str],
    provider_to_symbols: Dict[str, Any],
    primary_provider_id: str,
    data_by_symbol: Dict[str, Any],
    meta_by_symbol: Dict[str, Any],
    tradable_symbols: List[str],
    dividend_fields: List[str],
    progress_callback: Optional[Callable[[None], None]],
) -> Tuple[Dict[str, Any], Dict[str, Any], bool, DatetimeIndex, Optional[str]]:
    dividends_currency_match = True

    all_provider_symbols = []
    for provider_code, symbols in provider_to_symbols.items():
        all_provider_symbols.extend(
            [
                true_symbol_to_symbol(symbol) + "_" + provider_code
                if provider_code != primary_provider_id
                else true_symbol_to_symbol(symbol)
                for symbol in symbols["true_symbols"]
            ]
        )

    meta_by_symbol = {symbol: meta_by_symbol[symbol] for symbol in all_provider_symbols if symbol in meta_by_symbol}
    data_by_symbol = {symbol: data_by_symbol[symbol] for symbol in all_provider_symbols if symbol in data_by_symbol}

    for provider_id, all_symbols in provider_to_symbols.items():
        provider = data_providers[provider_id]

        symbols = [true_symbol_to_symbol(i) for i in all_symbols["true_symbols"]]
        true_symbols = dict(zip(symbols, all_symbols["true_symbols"]))

        indicators = [true_symbol_to_symbol(i) for i in all_symbols.get("economic_indicators", [])]
        true_indicators = dict(zip(indicators, all_symbols.get("economic_indicators", [])))

        data_by_symbol, meta_by_symbol, error_message = provider.fetch_datasets(
            true_symbols,
            start_date,
            end_date,
            interval,
            datasets,
            primary_provider_id,
            true_indicators,
            dividend_fields,
            data_by_symbol,
            meta_by_symbol,
            progress_callback,
        )

        if error_message is not None:
            break

    data_by_symbol_trad, data_by_symbol_non_trad = {
        key: value for key, value in data_by_symbol.items() if key in tradable_symbols
    }, {key: value for key, value in data_by_symbol.items() if key not in tradable_symbols}
    if tradable_symbols:
        data_by_symbol_trad = unify_indexes(data_by_symbol_trad)
        for s in data_by_symbol_trad:
            data_by_symbol_trad[s].index = pd.DatetimeIndex(data_by_symbol_trad[s].index)
            data_by_symbol_trad[s] = data_by_symbol_trad[s][
                (~data_by_symbol_trad[s].index.month.isin([1])) | (~data_by_symbol_trad[s].index.day.isin([1]))
            ]
            data_by_symbol_trad[s] = data_by_symbol_trad[s][
                (~data_by_symbol_trad[s].index.month.isin([12])) | (~data_by_symbol_trad[s].index.day.isin([25]))
            ]
        time_line = data_by_symbol_trad[list(data_by_symbol_trad.keys())[0]].index
    else:
        time_line = get_default_calendar(start_date, end_date)
    data_by_symbol_non_trad = unify_indexes(data_by_symbol_non_trad, time_line)
    data_by_symbol = data_by_symbol_trad | data_by_symbol_non_trad

    return data_by_symbol, meta_by_symbol, dividends_currency_match, time_line, error_message


def true_symbol_to_symbol(symbol: str) -> str:
    na_letters = ["-", ".", "/", "&", " ", ":", '"']

    for l in na_letters:
        symbol = symbol.replace(l, "")

    if symbol[0].isdigit():
        symbol = "_" + symbol

    symbol = symbol.replace("@", "_")
    return symbol


def get_meta_data(meta: Dict[str, Any], list_symbols: List[Dict[str, Any]]) -> str:
    if meta is None:
        return ""

    if "exchange" in meta:
        for ls_meta in list_symbols:
            if meta["symbol"] == ls_meta["symbol"] and "exchange" in ls_meta and meta["exchange"] == ls_meta["exchange"]:
                break
    else:
        for ls_meta in list_symbols:
            if meta["symbol"] == ls_meta["symbol"]:
                break
    return " ".join(["%s: %s" % (k, v) for k, v in ls_meta.items() if k != "symbol"])
