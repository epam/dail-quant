import inspect
import types
from typing import Any, Dict, Generator

import numpy as np
import pandas as pd

from market_alerts.containers import (
    data_periodicities,
    data_providers,
    trade_fill_prices,
)
from market_alerts.domain.constants import DATA_PROVIDER_FX_RATES
from market_alerts.domain.default_list import DefaultList
from market_alerts.domain.services.steps.risk_limit import *
from market_alerts.domain.services.steps.utils import (
    get_fx_rate,
    get_globals,
    get_print_redirect,
    get_sparse_dividends,
    unify_indexes,
)
from market_alerts.infrastructure.services.code import (
    compile_code,
    exec_code,
    get_code_sections,
)


def get_trade_stats(
    trading_stats_by_symbol,
    tradable_symbols,
    list_trade_fill_price_by_symbol,
    fx_rates,
    execution_cost_bps,
    apply_dividends,
    cum_dividends_by_symbol,
    cum_acct_ccy_dividends_by_symbol,
):
    entry_price = {key: {} for key in tradable_symbols}
    exit_price = {key: {} for key in tradable_symbols}
    current_position = {key: [] for key in tradable_symbols}
    for symbol in tradable_symbols:
        symbols_time_line = trading_stats_by_symbol[symbol].index
        array_buy_alert = trading_stats_by_symbol[symbol]["buy_alert"].values
        array_sell_alert = trading_stats_by_symbol[symbol]["sell_alert"].values
        array_quantity = trading_stats_by_symbol[symbol]["quantity"].values
        if apply_dividends:
            if symbol in cum_dividends_by_symbol:
                cur_symbol_cum_dividends = cum_dividends_by_symbol[symbol].values
                cur_symbol_cum_acct_ccy_dividends = cum_acct_ccy_dividends_by_symbol[symbol].values
            else:
                cur_symbol_cum_dividends = None
                cur_symbol_cum_acct_ccy_dividends = None

        for idx, (t, buy_alert, sell_alert, quantity, price, fx_rate) in enumerate(
            zip(
                symbols_time_line,
                array_buy_alert,
                array_sell_alert,
                array_quantity,
                list_trade_fill_price_by_symbol[symbol],
                fx_rates[symbol],
            )
        ):
            if buy_alert or sell_alert:
                if apply_dividends:
                    if cur_symbol_cum_dividends is not None:
                        cum_dividend = cur_symbol_cum_dividends[max(idx - 1, 0)]
                        acct_ccy_cum_dividend = cur_symbol_cum_acct_ccy_dividends[max(idx - 1, 0)]
                    else:
                        cum_dividend = 0.0
                        acct_ccy_cum_dividend = 0.0

                if execution_cost_bps is not None:
                    price = (
                        (price + abs(price) * execution_cost_bps / 10000)
                        if buy_alert
                        else ((price - abs(price) * execution_cost_bps / 10000) if sell_alert else 0.0)
                    )
                qty = quantity
                if not current_position[symbol] or np.sign(current_position[symbol][0]["size"]) == np.sign(qty):
                    current_position[symbol].append({"size": qty, "price": price, "date": t})
                    entry_price[symbol][t] = {"size": qty, "price": price, "acct_ccy_price": price / fx_rate}
                    if apply_dividends:
                        entry_price[symbol][t]["cum_dividend"] = cum_dividend
                        entry_price[symbol][t]["acct_ccy_cum_dividend"] = acct_ccy_cum_dividend
                    exit_price[symbol][t] = []
                else:
                    while abs(qty) > 0 and current_position[symbol]:
                        if abs(qty) >= abs(current_position[symbol][0]["size"]):
                            exit_price[symbol][current_position[symbol][0]["date"]].append(
                                {
                                    "size": current_position[symbol][0]["size"],
                                    "price": price,
                                    "date": t,
                                    "acct_ccy_price": price / fx_rate,
                                }
                            )
                            if apply_dividends:
                                exit_price[symbol][current_position[symbol][0]["date"]][-1]["dividends"] = (
                                    cum_dividend - entry_price[symbol][current_position[symbol][0]["date"]]["cum_dividend"]
                                ) * current_position[symbol][0]["size"]
                                exit_price[symbol][current_position[symbol][0]["date"]][-1]["acct_ccy_dividends"] = (
                                    acct_ccy_cum_dividend
                                    - entry_price[symbol][current_position[symbol][0]["date"]]["acct_ccy_cum_dividend"]
                                ) * current_position[symbol][0]["size"]
                            qty += current_position[symbol][0]["size"]
                            del current_position[symbol][0]
                        else:
                            exit_price[symbol][current_position[symbol][0]["date"]].append(
                                {"size": -qty, "price": price, "date": t, "acct_ccy_price": price / fx_rate}
                            )
                            if apply_dividends:
                                exit_price[symbol][current_position[symbol][0]["date"]][-1]["dividends"] = (
                                    cum_dividend - entry_price[symbol][current_position[symbol][0]["date"]]["cum_dividend"]
                                ) * (-qty)
                                exit_price[symbol][current_position[symbol][0]["date"]][-1]["acct_ccy_dividends"] = (
                                    acct_ccy_cum_dividend
                                    - entry_price[symbol][current_position[symbol][0]["date"]]["acct_ccy_cum_dividend"]
                                ) * (-qty)
                            current_position[symbol][0]["size"] += qty
                            qty = 0
                    if abs(qty) > 0:
                        current_position[symbol].append({"size": qty, "price": price, "date": t})
                        entry_price[symbol][t] = {"size": qty, "price": price, "acct_ccy_price": price / fx_rate}
                        if apply_dividends:
                            entry_price[symbol][t]["cum_dividend"] = cum_dividend
                            entry_price[symbol][t]["acct_ccy_cum_dividend"] = acct_ccy_cum_dividend
                        exit_price[symbol][t] = []
        while current_position[symbol]:
            exit_price[symbol][current_position[symbol][0]["date"]].append(
                {"size": current_position[symbol][0]["size"], "price": price, "date": t, "acct_ccy_price": price / fx_rate}
            )
            if apply_dividends:
                exit_price[symbol][current_position[symbol][0]["date"]][-1]["dividends"] = (
                    cum_dividend - entry_price[symbol][current_position[symbol][0]["date"]]["cum_dividend"]
                ) * current_position[symbol][0]["size"]
                exit_price[symbol][current_position[symbol][0]["date"]][-1]["acct_ccy_dividends"] = (
                    acct_ccy_cum_dividend - entry_price[symbol][current_position[symbol][0]["date"]]["acct_ccy_cum_dividend"]
                ) * current_position[symbol][0]["size"]
            del current_position[symbol][0]

    trades_by_symbol = dict()
    for symbol in tradable_symbols:
        trades = []
        for entry_t, exits in exit_price[symbol].items():
            for exit in exits:
                size = abs(exit["size"])
                type_ = exit["size"] > 0
                entry_p = entry_price[symbol][entry_t]["price"]
                entry_acct_ccy_p = entry_price[symbol][entry_t]["acct_ccy_price"]
                exit_p = exit["price"]
                exit_acct_ccy_p = exit["acct_ccy_price"]
                exit_t = exit["date"]
                net_profit = (exit_p - entry_p) * exit["size"]
                net_profit_fund = (exit_acct_ccy_p - entry_acct_ccy_p) * exit["size"]
                if apply_dividends:
                    net_profit += exit["dividends"]
                    net_profit_fund += exit["acct_ccy_dividends"]
                is_it_win = net_profit_fund > 0
                time_in_trade = (exit_t - entry_t).days
                trades.append(
                    [
                        type_,
                        size,
                        entry_t,
                        entry_p,
                        entry_acct_ccy_p,
                        exit_t,
                        exit_p,
                        exit_acct_ccy_p,
                        net_profit,
                        is_it_win,
                        net_profit_fund,
                        time_in_trade,
                    ]
                )
        trades_by_symbol[symbol] = pd.DataFrame(
            trades,
            columns=[
                "long",
                "size",
                "entry time",
                "entry price",
                "entry price acct ccy",
                "exit time",
                "exit price",
                "exit price acct ccy",
                "net profit",
                "win",
                "net profit acct ccy",
                "days in trade",
            ],
        )

    def make_trade_stats(current_trades):
        trade_stats = dict()
        trade_stats["# Trades"] = current_trades.shape[0]
        trade_stats["Winning Trades"] = current_trades["win"].sum()
        trade_stats["Losing Trades"] = (~current_trades["win"]).sum()
        trade_stats["Win Ratio"] = current_trades["win"].mean()
        trade_stats["Avg. Trade"] = round(current_trades["net profit acct ccy"].mean(), 2)
        if current_trades["win"].any():
            trade_stats["Avg. Winning Trade"] = round(current_trades["net profit acct ccy"][current_trades["win"]].mean(), 2)
        else:
            trade_stats["Avg. Winning Trade"] = 0.0
        if (~current_trades["win"]).any():
            trade_stats["Avg. Losing Trade"] = round(current_trades["net profit acct ccy"][~current_trades["win"]].mean(), 2)
        else:
            trade_stats["Avg. Losing Trade"] = 0.0

        if abs(trade_stats["Avg. Losing Trade"]) >= 1e-20:
            trade_stats["Payoff Ratio"] = trade_stats["Avg. Winning Trade"] / abs(trade_stats["Avg. Losing Trade"])
        else:
            trade_stats["Payoff Ratio"] = float("nan")
        if current_trades["size"].sum() >= 1e-20:
            trade_stats["Profit Margin (Bps)"] = round(
                current_trades["net profit acct ccy"].sum() / current_trades["size"].sum(), 2
            )
        else:
            trade_stats["Profit Margin (Bps)"] = float("nan")

        sum_entry_price_acct_ccy_mul_size = (current_trades["entry price acct ccy"] * current_trades["size"]).sum()
        if abs(sum_entry_price_acct_ccy_mul_size) >= 1e-20:
            trade_stats["Profit Margin (Bps)"] = round(
                current_trades["net profit acct ccy"].sum() / sum_entry_price_acct_ccy_mul_size * 10000, 2
            )
        else:
            trade_stats["Profit Margin (Bps)"] = float("nan")

        trade_stats["Avg. Holding Days"] = round(current_trades["days in trade"].mean(), 2)
        return trade_stats

    trade_stats_by_symbol = dict()
    for symbol in tradable_symbols:
        current_trades = trades_by_symbol[symbol]
        trade_stats_by_symbol[symbol] = make_trade_stats(current_trades)

    long_trade_stats_by_symbol = dict()
    for symbol in tradable_symbols:
        current_trades = trades_by_symbol[symbol].loc[trades_by_symbol[symbol]["long"]]
        long_trade_stats_by_symbol[symbol] = make_trade_stats(current_trades)

    short_trade_stats_by_symbol = dict()
    for symbol in tradable_symbols:
        current_trades = trades_by_symbol[symbol].loc[~trades_by_symbol[symbol]["long"]]
        short_trade_stats_by_symbol[symbol] = make_trade_stats(current_trades)

    list_trades = []
    for symbol in tradable_symbols:
        trades = trades_by_symbol[symbol].copy()
        if trades.shape[0] > 0:
            trades["symbol"] = symbol
            list_trades.append(trades)
    if list_trades:
        all_trades = pd.concat(list_trades).sort_values("exit time").reset_index(drop=True)
    else:
        all_trades = pd.DataFrame(
            [],
            columns=[
                "long",
                "size",
                "entry time",
                "entry price",
                "entry price acct ccy",
                "exit time",
                "exit price",
                "exit price acct ccy",
                "net profit",
                "win",
                "net profit acct ccy",
                "days in trade",
            ],
        )
    global_trade_stats = make_trade_stats(all_trades)
    long_global_trade_stats = make_trade_stats(all_trades.loc[all_trades["long"]])
    short_global_trade_stats = make_trade_stats(all_trades.loc[~all_trades["long"]])

    return (
        trades_by_symbol,
        trade_stats_by_symbol,
        long_trade_stats_by_symbol,
        short_trade_stats_by_symbol,
        all_trades,
        global_trade_stats,
        long_global_trade_stats,
        short_global_trade_stats,
    )


def get_actual_currency_fx_rates(session_dict, actual_currency: str) -> dict[str, Any]:
    time_line = session_dict["time_line"]
    provider = data_providers[session_dict["data_provider"]]
    fx_rate_template = DATA_PROVIDER_FX_RATES[session_dict["data_provider"]]
    symbol_to_currency = session_dict["symbol_to_currency"]

    for s in set(symbol_to_currency.values()):
        if "/" not in s:
            fx_rate_symbol = fx_rate_template % (actual_currency, s)
        else:
            left_curr, right_curr = s.split("/")
            fx_rate_symbol = fx_rate_template % (left_curr, right_curr)
        if not fx_rate_symbol in session_dict["fx_rates"]:
            if s == actual_currency:
                session_dict["fx_rates"][fx_rate_symbol] = pd.DataFrame(
                    data=1.0, columns=["open", "high", "low", "close"], index=time_line
                )
            else:
                session_dict["fx_rates"][fx_rate_symbol] = get_fx_rate(
                    time_line,
                    fx_rate_symbol,
                    session_dict["start_date"],
                    session_dict["end_date"],
                    data_periodicities[session_dict["interval"]]["value"],
                    provider,
                )
    return unify_indexes(session_dict["fx_rates"], time_line)


def get_sparse_dividends_for_each_tradable_symbol(session_dict) -> dict[str, Any]:
    time_line = session_dict["time_line"]
    provider = data_providers[session_dict["data_provider"]]

    dividends_by_symbol = session_dict.setdefault("dividends_by_symbol", dict())

    new_time_period = session_dict["time_period"]
    dividends_time_period = session_dict.setdefault("dividends_time_period", new_time_period)

    for symbol in session_dict["tradable_symbols"]:
        if symbol not in dividends_by_symbol or new_time_period > dividends_time_period:
            dividends_by_symbol[symbol] = get_sparse_dividends(
                time_line=time_line,
                provider=provider,
                symbol=symbol,
                div_end_date=session_dict["end_date"],
                div_start_date=session_dict["start_date"],
                true_symbols=session_dict["true_symbols"],
            )
            session_dict["dividends_time_period"] = new_time_period
    return dividends_by_symbol


def get_global_strategy_stats(
    trading_stats_by_symbol: dict[str, pd.DataFrame],
    strategy_stats: pd.DataFrame,
    tradable_symbols: list[str],
    time_line: pd.DatetimeIndex,
) -> dict[str, Any]:
    pos_cost = sum(
        [
            (trading_stats_by_symbol[s]["acct_ccy_cost"] * (trading_stats_by_symbol[s]["acct_ccy_cost"] > 0)).fillna(0)
            for s in tradable_symbols
        ]
    )
    neg_cost = sum(
        [
            (trading_stats_by_symbol[s]["acct_ccy_cost"].abs() * (trading_stats_by_symbol[s]["acct_ccy_cost"] < 0)).fillna(0)
            for s in tradable_symbols
        ]
    )
    aprox_n_trading_days = 365 * time_line.shape[0] / (time_line[-1] - time_line[0]).days
    n_trading_days_per_year = 250 if abs(250 - aprox_n_trading_days) < abs(365 - aprox_n_trading_days) else 365

    global_strategy_stats = dict()
    global_strategy_stats["Gross Profit/Loss"] = round(strategy_stats["acct_ccy_pnl"][-1], 2)
    global_strategy_stats["Turnover BOT"] = round(neg_cost.sum(), 2)
    global_strategy_stats["Turnover SOLD"] = round(pos_cost.sum(), 2)
    drawdowns = [[0, 0, 0, 0]]
    for i, stg_pnl in enumerate(strategy_stats["acct_ccy_pnl"].values):
        if stg_pnl > drawdowns[-1][0]:
            drawdowns.append([stg_pnl, stg_pnl, i, i])
        else:
            drawdowns[-1][3] += 1
            drawdowns[-1][1] = drawdowns[-1][1] if drawdowns[-1][1] < stg_pnl else stg_pnl
    max_drawdown = max(drawdowns, key=lambda x: x[0] - x[1])
    max_drawdown_dur = max(drawdowns, key=lambda x: x[3] - x[2])
    global_strategy_stats["Max DD"] = round(max_drawdown[0] - max_drawdown[1], 2)
    if abs(global_strategy_stats["Max DD"]) < 1e-20:
        global_strategy_stats["Return/DD Ratio"] = float("nan")
    else:
        global_strategy_stats["Return/DD Ratio"] = global_strategy_stats["Gross Profit/Loss"] / global_strategy_stats["Max DD"]
    global_strategy_stats["Max DD duration"] = max_drawdown_dur[3] - max_drawdown_dur[2]
    global_strategy_stats["Annual Return"] = round(strategy_stats["acct_ccy_day_pnl"].mean() * n_trading_days_per_year, 2)
    global_strategy_stats["Annualized Volatility"] = strategy_stats["acct_ccy_day_pnl"].std() * np.sqrt(n_trading_days_per_year)
    if abs(global_strategy_stats["Annualized Volatility"]) < 1e-20:
        global_strategy_stats["Information Ratio"] = float("nan")
    else:
        global_strategy_stats["Information Ratio"] = (
            global_strategy_stats["Annual Return"] / global_strategy_stats["Annualized Volatility"]
        )
    return global_strategy_stats


def trading_step(
    session_dict: Dict[str, Any],
    trading_handler=None,
    risk_rules=None,
    apply_dividends=False,
    reverse_signal=False,
    start_idx=0,
    end_idx=None,
    is_trades_stats_needed=True,
) -> Generator[int, None, None]:
    if trading_handler is not None:
        trading_code = inspect.getsource(trading_handler)
        llm_response = f"""```python
```
```python
{trading_code}
```"""
    else:
        llm_response = session_dict["indicators_dialogue"][-1]

    _, trading_code = get_code_sections(llm_response)

    session_dict["trading_code"] = trading_code

    actual_currency = session_dict["actual_currency"] if session_dict["actual_currency"] else "USD"
    time_line = session_dict["time_line"]

    fx_rates = get_actual_currency_fx_rates(session_dict, actual_currency)

    if apply_dividends:
        dividends_by_symbol = get_sparse_dividends_for_each_tradable_symbol(session_dict)

    tradable_symbols = session_dict["tradable_symbols"]
    bet_size = session_dict["bet_size"]
    total_gross_limit = session_dict["total_gross_limit"]
    nop_limit = session_dict["nop_limit"]
    per_instrument_gross_limit = session_dict["per_instrument_gross_limit"]

    fx_rate_template = DATA_PROVIDER_FX_RATES[session_dict["data_provider"]]
    symbol_to_currency = session_dict["symbol_to_currency"]
    fx_rate_symbol_by_symbol = {
        s: fx_rate_template % (actual_currency, symbol_to_currency[s])
        if "/" not in symbol_to_currency[s]
        else fx_rate_template % (symbol_to_currency[s].split("/")[0], symbol_to_currency[s].split("/")[1])
        for s in tradable_symbols
    }
    trading_stats_by_symbol = {s: dict() for s in tradable_symbols}

    risk_processor = None
    if risk_rules is not None:
        risk_processor = RiskRuleProcessor(
            betSize=bet_size, grossLimit=total_gross_limit, nopLimit=nop_limit, instrumentGrossLimit=per_instrument_gross_limit
        )
        risk_processor.addRiskRules(risk_rules)
        strategy_pnl_stats = {}

        strategy_pnl_stats["acct_ccy_pnl"] = DefaultList([], 0.0)

    ################################################################################################

    apply_execution_cost = session_dict.get("execution_cost_bps", 0.0) != 0.0

    ################################################################################################

    default_value_by_property = {
        "quantity": 0.0,
        "quote_ccy_cost": 0.0,
        "acct_ccy_cost": 0.0,
        "quote_ccy_total_cost": 0.0,
        "acct_ccy_total_cost": 0.0,
        "total_size": 0.0,
        "quote_ccy_value": 0.0,
        "acct_ccy_value": 0.0,
        "quote_ccy_pnl": 0.0,
        "quote_ccy_day_pnl": 0.0,
        "acct_ccy_pnl": 0.0,
        "acct_ccy_day_pnl": 0.0,
        "buy_alert": False,
        "sell_alert": False,
    }
    if end_idx is None:
        end_idx = len(time_line)

    if session_dict["fill_trade_price"] == "day_close":
        check_len = end_idx
        check_range = range(start_idx, end_idx)
        trade_range = range(start_idx, end_idx)
    else:
        check_len = end_idx - 1
        check_range = range(start_idx, end_idx - 1)
        trade_range = range(start_idx + 1, end_idx)

    if apply_dividends:
        default_value_by_property["cum_dividends"] = 0.0
        default_value_by_property["acct_ccy_cum_dividends"] = 0.0

    for s in tradable_symbols:
        for p, v in default_value_by_property.items():
            #             trading_stats_by_symbol[s][p] = DefaultList([v], v)
            trading_stats_by_symbol[s][p] = np.ones(check_len + 1, dtype=type(v)) * v

    strategy_stats = dict()

    default_value_by_strategy_property = {
        "acct_ccy_total_cost": 0.0,
        "acct_ccy_value": 0.0,
        "gross_value": 0.0,
        "acct_ccy_pnl": 0.0,
        "acct_ccy_day_pnl": 0.0,
    }
    for p, v in default_value_by_strategy_property.items():
        #         strategy_stats[p] = DefaultList([v], v)
        strategy_stats[p] = np.ones(check_len + 1, dtype=type(v)) * v

    array_close_price_by_symbol = {key: session_dict["data_by_symbol"][key]["close"].to_numpy() for key in tradable_symbols}
    array_open_price_by_symbol = {key: session_dict["data_by_symbol"][key]["open"].to_numpy() for key in tradable_symbols}
    array_fx_rates = {key: value["close"].to_numpy() for key, value in fx_rates.items()}

    if apply_dividends:
        array_dividends_by_symbol = {key: value.to_numpy() for key, value in dividends_by_symbol.items()}

    ################################################################################################
    ##                                                                                            ##
    ##                                need to rewrite this                                        ##
    ##                                                                                            ##
    ################################################################################################

    list_trade_fill_price_by_symbol = trade_fill_prices.get_trade_price_by_backend_key(
        session_dict["fill_trade_price"]
    ).get_list_trade_fill_price_by_symbol({key: session_dict["data_by_symbol"][key] for key in tradable_symbols})

    ################################################################################################

    comp_trading_code = compile_code(trading_code)

    def calculate_properties(apply_execution_cost: bool = False):
        total_size = quantity + trading_stats_by_symbol[s]["total_size"][check_idx]
        quote_ccy_value = total_size * array_close_price_by_symbol[s][idx]
        acct_ccy_value = quote_ccy_value / array_fx_rates[fx_rate_symbol_by_symbol[s]][idx]
        buy_alert = quantity > 1e-10
        sell_alert = quantity < -1e-10
        if apply_execution_cost:
            base_price = list_trade_fill_price_by_symbol[s][idx]
            execution_cost = (
                (base_price + abs(base_price) * session_dict["execution_cost_bps"] / 10000)
                if buy_alert
                else ((base_price - abs(base_price) * session_dict["execution_cost_bps"] / 10000) if sell_alert else 0.0)
            )
            quote_ccy_cost = -quantity * execution_cost
        else:
            quote_ccy_cost = -quantity * list_trade_fill_price_by_symbol[s][idx]
        acct_ccy_cost = quote_ccy_cost / array_fx_rates[fx_rate_symbol_by_symbol[s]][idx]
        quote_ccy_total_cost = quote_ccy_cost + trading_stats_by_symbol[s]["quote_ccy_total_cost"][check_idx]
        acct_ccy_total_cost = acct_ccy_cost + trading_stats_by_symbol[s]["acct_ccy_total_cost"][check_idx]
        quote_ccy_pnl = quote_ccy_total_cost + quote_ccy_value
        if apply_dividends:
            cur_dividends = total_size * array_dividends_by_symbol[s][idx] if s in array_dividends_by_symbol else 0.0
            cum_dividends = trading_stats_by_symbol[s]["cum_dividends"][check_idx] + cur_dividends
            quote_ccy_pnl += cum_dividends
        quote_ccy_day_pnl = quote_ccy_pnl - trading_stats_by_symbol[s]["quote_ccy_pnl"][check_idx]
        acct_ccy_pnl = acct_ccy_total_cost + acct_ccy_value
        if apply_dividends:
            acct_ccy_cum_dividends = (
                trading_stats_by_symbol[s]["acct_ccy_cum_dividends"][check_idx]
                + cur_dividends / array_fx_rates[fx_rate_symbol_by_symbol[s]][idx]
            )
            acct_ccy_pnl += acct_ccy_cum_dividends
        acct_ccy_day_pnl = acct_ccy_pnl - trading_stats_by_symbol[s]["acct_ccy_pnl"][check_idx]
        properties = {
            "quote_ccy_cost": quote_ccy_cost,
            "acct_ccy_cost": acct_ccy_cost,
            "quote_ccy_total_cost": quote_ccy_total_cost,
            "acct_ccy_total_cost": acct_ccy_total_cost,
            "total_size": total_size,
            "quote_ccy_value": quote_ccy_value,
            "acct_ccy_value": acct_ccy_value,
            "quote_ccy_pnl": quote_ccy_pnl,
            "quote_ccy_day_pnl": quote_ccy_day_pnl,
            "acct_ccy_pnl": acct_ccy_pnl,
            "acct_ccy_day_pnl": acct_ccy_day_pnl,
            "buy_alert": buy_alert,
            "sell_alert": sell_alert,
        }
        if apply_dividends:
            properties["cum_dividends"] = cum_dividends
            properties["acct_ccy_cum_dividends"] = acct_ccy_cum_dividends
        return properties

    def calculate_strategy_properties():
        properties = {
            "acct_ccy_total_cost": sum(
                [trading_stats_by_symbol[s]["acct_ccy_total_cost"][check_idx + 1] for s in tradable_symbols]
            ),
            "acct_ccy_value": sum([trading_stats_by_symbol[s]["acct_ccy_value"][check_idx + 1] for s in tradable_symbols]),
            "gross_value": sum([np.abs(trading_stats_by_symbol[s]["acct_ccy_value"][check_idx + 1]) for s in tradable_symbols]),
            "acct_ccy_pnl": sum([trading_stats_by_symbol[s]["acct_ccy_pnl"][check_idx + 1] for s in tradable_symbols]),
            "acct_ccy_day_pnl": sum([trading_stats_by_symbol[s]["acct_ccy_day_pnl"][check_idx + 1] for s in tradable_symbols]),
        }
        return properties

    cur_sum_value = 0.0
    pos_value = 0.0
    neg_value = 0.0

    lclsglbls = session_dict.get("lclsglbls", get_globals())

    if lclsglbls.get("data_by_symbol", None) is None:
        for key in session_dict["data_by_symbol"]:
            lclsglbls[key] = session_dict["data_by_symbol"][key]
        for key in session_dict["data_by_synth"]:
            lclsglbls[key] = session_dict["data_by_synth"][key]
        lclsglbls["data_by_symbol"] = session_dict["data_by_symbol"]

    if lclsglbls.get("supplementary_symbols", None) is None:
        lclsglbls["supplementary_symbols"] = session_dict["supplementary_symbols"]

    if lclsglbls.get("tradable_symbols", None) is None:
        lclsglbls["tradable_symbols"] = session_dict["tradable_symbols"]

    #     trading_info = {
    #         key: {k: DefaultList([], trading_stats_by_symbol[key][k][0]) for k in trading_stats_by_symbol[key]}
    #         for key in trading_stats_by_symbol
    #     }

    fx_rates_info = dict()
    for key in tradable_symbols:
        fx_rates_info[key] = fx_rates[fx_rate_symbol_by_symbol[key]]["close"]

    lclsglbls["fx_rates"] = fx_rates_info
    lclsglbls["time_line"] = time_line
    lclsglbls["trading_info"] = trading_stats_by_symbol
    lclsglbls["strategy_info"] = strategy_stats
    lclsglbls["bet_size"] = bet_size  # do not remove; required for backward compatibility of old models

    all_iterations_num = end_idx - 1

    prev_percent, current_percent = 0, 0

    if trading_handler is None:
        lclsglbls["print"], session_dict["trading_code_log"] = get_print_redirect()

    for check_idx, idx in zip(check_range, trade_range):
        current_percent = min(idx * 100 // all_iterations_num, 99)

        if current_percent != prev_percent:
            yield current_percent

        # set limits according to risk rule settings
        if risk_processor is not None:
            #             bet_size = risk_processor.BetSize
            total_gross_limit = risk_processor.GrossLimit
            nop_limit = risk_processor.NOPLimit
        #             per_instrument_gross_limit_by_symbol
        #         for key in trading_stats_by_symbol:
        #             for k in trading_stats_by_symbol[key]:
        #                 trading_info[key][k].append(trading_stats_by_symbol[key][k][-1])

        lclsglbls["order"] = {s: {"size": 0, "unit_of_measure": "base units"} for s in tradable_symbols}
        lclsglbls["idx"] = check_idx
        lclsglbls["bet_size"] = bet_size

        if risk_processor is not None:
            strategy_pnl_stats["acct_ccy_pnl"].append(
                sum([trading_stats_by_symbol[s]["acct_ccy_pnl"][-1] for s in tradable_symbols])
            )
            risk_processor.applyRiskRules(trading_stats_by_symbol, strategy_pnl_stats, time_line[idx], time_line)

        # set limits according to risk rules
        if risk_processor is not None:
            nop_limit = risk_processor.NOPLimit
            total_gross_limit = risk_processor.GrossLimit
            lclsglbls["default_bet_size_by_symbol"] = {
                s: min(risk_processor.getSymbolDetails(s).BetSize, risk_processor.BetSize) for s in tradable_symbols
            }
            per_instrument_gross_limit_by_symbol = {
                s: min(risk_processor.getSymbolDetails(s).InstrumentGrossLimit, risk_processor.InstrumentGrossLimit)
                for s in tradable_symbols
            }
        else:
            lclsglbls["default_bet_size_by_symbol"] = {s: bet_size for s in tradable_symbols}
            per_instrument_gross_limit_by_symbol = {s: per_instrument_gross_limit for s in tradable_symbols}

        if trading_handler is not None:
            handler_func_with_updated_globals = types.FunctionType(trading_handler.__code__, lclsglbls, trading_handler.__name__)
            lclsglbls.update(handler_func_with_updated_globals())
        else:
            exec_code(trading_code, lclsglbls, lclsglbls, comp_trading_code)
            del lclsglbls["__builtins__"]

        prev_percent = current_percent

        for s in tradable_symbols:
            force_close_positon = False
            suspend_trading = risk_processor is not None and risk_processor.isTradingSuspended(
                s
            )  # do not trade is suspended by risk rule
            # close position by stop loss
            if (suspend_trading and trading_stats_by_symbol[s]["acct_ccy_value"][-1] != 0) or idx == end_idx - 1:
                lclsglbls["order"][s] = {
                    "size": -trading_stats_by_symbol[s]["total_size"][check_idx],
                    "unit_of_measure": "base units",
                }
                force_close_positon = True
            if pd.isna(lclsglbls["order"][s]["size"]):
                lclsglbls["order"][s]["size"] = 0
            if lclsglbls["order"][s]["unit_of_measure"] == "base units":
                quantity = np.round(lclsglbls["order"][s]["size"])
            else:
                quantity = np.round(
                    (lclsglbls["order"][s]["size"])
                    / (array_close_price_by_symbol[s][check_idx] / array_fx_rates[fx_rate_symbol_by_symbol[s]][check_idx])
                )
            if reverse_signal:
                quantity = -quantity
            properties = calculate_properties(apply_execution_cost)
            for key, value in properties.items():
                if np.isnan(value):
                    quantity = 0.0
                    properties = {k: default_value_by_property[k] for k, v in properties.items()}
                    break

            is_abs_total_size_decreases = np.abs(properties["total_size"]) < np.abs(
                trading_stats_by_symbol[s]["total_size"][check_idx]
            )
            cur_sum_value = cur_sum_value - np.abs(trading_stats_by_symbol[s]["acct_ccy_value"][check_idx])
            pos_value = pos_value - trading_stats_by_symbol[s]["acct_ccy_value"][check_idx] * (
                trading_stats_by_symbol[s]["acct_ccy_value"][check_idx] > 0
            )
            neg_value = neg_value - np.abs(trading_stats_by_symbol[s]["acct_ccy_value"][check_idx]) * (
                trading_stats_by_symbol[s]["acct_ccy_value"][check_idx] < 0
            )
            NOP_pos_value = pos_value + (properties["acct_ccy_value"] if properties["acct_ccy_value"] > 0 else 0)
            NOP_neg_value = neg_value + (-properties["acct_ccy_value"] if properties["acct_ccy_value"] < 0 else 0)
            NOP_value = np.maximum(NOP_pos_value, NOP_neg_value)

            is_total_gross_limit_met = (cur_sum_value + np.abs(properties["acct_ccy_value"])) > total_gross_limit
            is_nop_limit_met = NOP_value > nop_limit
            #             is_pos_limit_met = NOP_pos_value > pos_limit
            #             is_neg_limit_met = NOP_neg_value > neg_limit
            is_per_instrument_gross_limit_met = np.abs(properties["acct_ccy_value"]) > per_instrument_gross_limit_by_symbol[s]
            if (suspend_trading and not force_close_positon) or (
                not is_abs_total_size_decreases
                and (
                    is_total_gross_limit_met
                    or is_nop_limit_met
                    or is_per_instrument_gross_limit_met
                    #                     or is_pos_limit_met
                    #                     or is_neg_limit_met
                )
            ):
                quantity = 0.0
                properties = calculate_properties(apply_execution_cost)
            for key, value in properties.items():
                if np.isnan(value):
                    quantity = 0.0
                    properties = {k: default_value_by_property[k] for k, v in properties.items()}
                    break

            trading_stats_by_symbol[s]["quantity"][check_idx + 1] = quantity
            for key, value in properties.items():
                trading_stats_by_symbol[s][key][check_idx + 1] = value

            cur_sum_value = cur_sum_value + np.abs(trading_stats_by_symbol[s]["acct_ccy_value"][check_idx + 1])
            pos_value = pos_value + trading_stats_by_symbol[s]["acct_ccy_value"][check_idx + 1] * (
                trading_stats_by_symbol[s]["acct_ccy_value"][check_idx + 1] > 0
            )
            neg_value = neg_value + np.abs(trading_stats_by_symbol[s]["acct_ccy_value"][check_idx + 1]) * (
                trading_stats_by_symbol[s]["acct_ccy_value"][check_idx + 1] < 0
            )
        strategy_properties = calculate_strategy_properties()
        for key, value in strategy_properties.items():
            strategy_stats[key][check_idx + 1] = value

    if session_dict["fill_trade_price"] == "day_close":
        trading_stats_by_symbol = {
            s: {key: trading_stats_by_symbol[s][key][1:] for key in trading_stats_by_symbol[s]} for s in trading_stats_by_symbol
        }
        strategy_stats = {key: strategy_stats[key][1:] for key in strategy_stats}
    trading_stats_by_symbol = {
        s: pd.DataFrame(trading_stats_by_symbol[s], index=time_line[:end_idx]).iloc[start_idx:] for s in trading_stats_by_symbol
    }
    session_dict["long_alert"] = {s: trading_stats_by_symbol[s]["buy_alert"] for s in trading_stats_by_symbol}
    session_dict["short_alert"] = {s: trading_stats_by_symbol[s]["sell_alert"] for s in trading_stats_by_symbol}
    strategy_stats = pd.DataFrame(strategy_stats, index=time_line[:end_idx]).iloc[start_idx:]
    pos_value = sum(
        [
            (trading_stats_by_symbol[s]["acct_ccy_value"] * (trading_stats_by_symbol[s]["acct_ccy_value"] > 0)).fillna(0)
            for s in tradable_symbols
        ]
    )
    neg_value = sum(
        [
            (trading_stats_by_symbol[s]["acct_ccy_value"].abs() * (trading_stats_by_symbol[s]["acct_ccy_value"] < 0)).fillna(0)
            for s in tradable_symbols
        ]
    )
    strategy_stats["NOP_value"] = np.maximum(pos_value, neg_value)
    global_strategy_stats = get_global_strategy_stats(trading_stats_by_symbol, strategy_stats, tradable_symbols, time_line)
    session_dict["trading_stats_by_symbol"] = trading_stats_by_symbol
    session_dict["strategy_stats"] = strategy_stats
    session_dict["global_strategy_stats"] = global_strategy_stats
    if is_trades_stats_needed:
        if apply_dividends:
            cum_dividends_by_symbol = {}
            cum_acct_ccy_dividends_by_symbol = {}

            for key, value in dividends_by_symbol.items():
                if key not in tradable_symbols:
                    continue
                else:
                    cum_dividends_by_symbol[key] = value[start_idx:end_idx].cumsum()
                    cum_acct_ccy_dividends_by_symbol[key] = (
                        (value / fx_rates[fx_rate_symbol_by_symbol[key]]["close"]).fillna(0)[start_idx:end_idx].cumsum()
                    )

        (
            trades_by_symbol,
            trade_stats_by_symbol,
            long_trade_stats_by_symbol,
            short_trade_stats_by_symbol,
            all_trades,
            global_trade_stats,
            long_global_trade_stats,
            short_global_trade_stats,
        ) = get_trade_stats(
            trading_stats_by_symbol,
            tradable_symbols,
            {key: value[start_idx:end_idx] for key, value in list_trade_fill_price_by_symbol.items()},
            {key: value[start_idx:end_idx] for key, value in fx_rates_info.items()},
            session_dict["execution_cost_bps"] if apply_execution_cost else None,
            apply_dividends,
            cum_dividends_by_symbol if apply_dividends else None,
            cum_acct_ccy_dividends_by_symbol if apply_dividends else None,
        )

        yield 100

        session_dict["trades_by_symbol"] = trades_by_symbol
        session_dict["trade_stats_by_symbol"] = trade_stats_by_symbol
        session_dict["long_trade_stats_by_symbol"] = long_trade_stats_by_symbol
        session_dict["short_trade_stats_by_symbol"] = short_trade_stats_by_symbol
        session_dict["all_trades"] = all_trades
        session_dict["global_stats_by_symbol"] = global_trade_stats
        session_dict["long_global_trade_stats"] = long_global_trade_stats
        session_dict["short_global_trade_stats"] = short_global_trade_stats
    else:
        yield 100


def get_combined_trading_statistics(
    session_dict,
    list_trading_stats_by_symbol,
    list_strategy_stats,
    start_idx=0,
    end_idx=None,
    apply_dividends=False,
):
    """
    Returns a dictionary of combined statistics
    Parameters:
        session_dict: session, which contains all the fields we need, like execution_cost_bps, actual_currency, fx_rates, etc.
        list_trading_stats_by_symbol: python list of trading_stats_by_symbol from session_dicts which statistics we want to combine(
                                      [s["trading_stats_by_symbol"] for s in many_session_dicts]).
        list_strategy_stats: python list of strategy_stats from session_dicts which statistics we want to combine.
        apply_dividends: same as apply_dividends of these sessions
    """

    time_line = session_dict["time_line"]
    tradable_symbols = session_dict["tradable_symbols"]
    actual_currency = session_dict["actual_currency"] if session_dict["actual_currency"] else "USD"
    fx_rates = get_actual_currency_fx_rates(session_dict, actual_currency)
    symbol_to_currency = session_dict["symbol_to_currency"]
    fx_rate_template = DATA_PROVIDER_FX_RATES[session_dict["data_provider"]]
    fx_rate_symbol_by_symbol = {
        s: fx_rate_template % (actual_currency, symbol_to_currency[s])
        if "/" not in symbol_to_currency[s]
        else fx_rate_template % (symbol_to_currency[s].split("/")[0], symbol_to_currency[s].split("/")[1])
        for s in tradable_symbols
    }
    fx_rates_info = dict()
    for key in tradable_symbols:
        fx_rates_info[key] = fx_rates[fx_rate_symbol_by_symbol[key]]["close"]
    if apply_dividends:
        dividends_by_symbol = get_sparse_dividends_for_each_tradable_symbol(session_dict)
    apply_execution_cost = session_dict.get("execution_cost_bps", 0.0) != 0.0
    if end_idx is None:
        end_idx = len(time_line)

    combined_trading_stats_by_symbol = {
        symbol: list_trading_stats_by_symbol[0][symbol].copy() for symbol in list_trading_stats_by_symbol[0]
    }
    for symbol in combined_trading_stats_by_symbol:
        for col in combined_trading_stats_by_symbol[symbol].columns:
            combined_trading_stats_by_symbol[symbol][col].values[:] = 0
    for t_stats_by_symbol in list_trading_stats_by_symbol:
        for symbol in combined_trading_stats_by_symbol:
            combined_trading_stats_by_symbol[symbol] += t_stats_by_symbol[symbol]
    trading_stats_by_symbol = combined_trading_stats_by_symbol

    combined_strategy_stats = list_strategy_stats[0].copy()
    for col in combined_strategy_stats.columns:
        combined_strategy_stats[col].values[:] = 0
    for s_stats in list_strategy_stats:
        combined_strategy_stats += s_stats
    strategy_stats = combined_strategy_stats

    pos_value = sum(
        [
            (trading_stats_by_symbol[s]["acct_ccy_value"] * (trading_stats_by_symbol[s]["acct_ccy_value"] > 0)).fillna(0)
            for s in tradable_symbols
        ]
    )
    neg_value = sum(
        [
            (trading_stats_by_symbol[s]["acct_ccy_value"].abs() * (trading_stats_by_symbol[s]["acct_ccy_value"] < 0)).fillna(0)
            for s in tradable_symbols
        ]
    )
    strategy_stats["NOP_value"] = np.maximum(pos_value, neg_value)
    long_alert = {s: trading_stats_by_symbol[s]["buy_alert"] for s in trading_stats_by_symbol}
    short_alert = {s: trading_stats_by_symbol[s]["sell_alert"] for s in trading_stats_by_symbol}

    global_strategy_stats = get_global_strategy_stats(trading_stats_by_symbol, strategy_stats, tradable_symbols, time_line)

    list_trade_fill_price_by_symbol = trade_fill_prices.get_trade_price_by_backend_key(
        session_dict["fill_trade_price"]
    ).get_list_trade_fill_price_by_symbol({key: session_dict["data_by_symbol"][key] for key in tradable_symbols})

    if apply_dividends:
        cum_dividends_by_symbol = {}
        cum_acct_ccy_dividends_by_symbol = {}

        for key, value in dividends_by_symbol.items():
            if key not in tradable_symbols:
                continue
            else:
                cum_dividends_by_symbol[key] = value[start_idx:end_idx].cumsum()
                cum_acct_ccy_dividends_by_symbol[key] = (
                    (value / fx_rates[fx_rate_symbol_by_symbol[key]]["close"]).fillna(0)[start_idx:end_idx].cumsum()
                )

    (
        trades_by_symbol,
        trade_stats_by_symbol,
        long_trade_stats_by_symbol,
        short_trade_stats_by_symbol,
        all_trades,
        global_trade_stats,
        long_global_trade_stats,
        short_global_trade_stats,
    ) = get_trade_stats(
        trading_stats_by_symbol,
        tradable_symbols,
        {key: value[start_idx:end_idx] for key, value in list_trade_fill_price_by_symbol.items()},
        {key: value[start_idx:end_idx] for key, value in fx_rates_info.items()},
        session_dict["execution_cost_bps"] if apply_execution_cost else None,
        apply_dividends,
        cum_dividends_by_symbol if apply_dividends else None,
        cum_acct_ccy_dividends_by_symbol if apply_dividends else None,
    )
    return {
        "long_alert": long_alert,
        "short_alert": short_alert,
        "trading_stats_by_symbol": trading_stats_by_symbol,
        "strategy_stats": strategy_stats,
        "global_strategy_stats": global_strategy_stats,
        "trades_by_symbol": trades_by_symbol,
        "trade_stats_by_symbol": trade_stats_by_symbol,
        "long_trade_stats_by_symbol": long_trade_stats_by_symbol,
        "short_trade_stats_by_symbol": short_trade_stats_by_symbol,
        "all_trades": all_trades,
        "global_stats_by_symbol": global_trade_stats,
        "long_global_trade_stats": long_global_trade_stats,
        "short_global_trade_stats": short_global_trade_stats,
    }
