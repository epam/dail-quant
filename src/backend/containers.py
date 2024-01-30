import warnings
from enum import Enum
from typing import Dict

from dateutil.relativedelta import relativedelta
from optuna.exceptions import ExperimentalWarning
from optuna.samplers import CmaEsSampler, QMCSampler, RandomSampler, TPESampler

from market_alerts.config import (
    ALERTS_BACKEND_SERVICE_LIMITS_ENABLED,
    ALERTS_BACKEND_SERVICE_URL,
    ALPHAVANTAGE_API_KEY,
    POLYGON_API_KEY,
    TWELVE_API_KEY,
)
from market_alerts.domain.data_providers import (
    AlphaVantageDataProvider,
    DefaultDataProvider,
    PolygonDataProvider,
    TwelveDataProvider,
)
from market_alerts.domain.trade_fill_prices import (
    DayClose,
    NextDayClose,
    NextDayMid,
    NextDayOpen,
    TradeFillPriceContainer,
)
from market_alerts.infrastructure.services.proxy import AlertsBackendProxy
from market_alerts.utils import string_to_ms


class UserPromptTypes(str, Enum):
    Generic = "generic_user_prompt"
    IndicatorBlockOnly = "indicator_user_prompt"
    TradingBlockOnly = "trading_user_prompt"


data_providers: Dict[str, DefaultDataProvider] = {
    "TW": TwelveDataProvider(twelve_api_key=TWELVE_API_KEY),
    "PI": PolygonDataProvider(polygon_api_key=POLYGON_API_KEY),
    "AV": AlphaVantageDataProvider(alpha_vantage_api_key=ALPHAVANTAGE_API_KEY),
}

# TODO: when we are done with supporting streamlit, I would suggest to use integer values of milliseconds here as query params
data_periodicities = {
    string_to_ms("1 day"): {
        "label": "1 day",
        "value": "1day",
        "default_checked": True,
    },
}

data_timeranges = {
    string_to_ms("1 day"): {
        "label": "1 day",
        "value": relativedelta(days=1),
    },
    string_to_ms("1 week"): {
        "label": "1 week",
        "value": relativedelta(weeks=1),
    },
    string_to_ms("1 month"): {
        "label": "1 month",
        "value": relativedelta(months=1),
    },
    string_to_ms("3 months"): {
        "label": "3 months",
        "value": relativedelta(months=3),
    },
    string_to_ms("6 months"): {
        "label": "6 months",
        "value": relativedelta(months=6),
    },
    string_to_ms("1 year"): {
        "label": "1 year",
        "value": relativedelta(years=1),
    },
    string_to_ms("5 years"): {
        "label": "5 years",
        "value": relativedelta(years=5),
    },
    string_to_ms("10 years"): {
        "label": "10 years",
        "value": relativedelta(years=10),
        "default_checked": True,
    },
    string_to_ms("20 years"): {
        "label": "20 years",
        "value": relativedelta(years=20),
    },
    string_to_ms("30 years"): {
        "label": "30 years",
        "value": relativedelta(years=30),
    },
}

trade_fill_prices = TradeFillPriceContainer([DayClose(), NextDayOpen(), NextDayClose(), NextDayMid()])

with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=ExperimentalWarning)
    qmc_sampler = {
        "label": "QMC Sampler",
        "value": QMCSampler,
        "is_random": True,
    }

optimization_samplers = {
    "tpe": {
        "label": "TPE Sampler",
        "value": TPESampler,
        "default_checked": True,
    },
    "cma": {
        "label": "CmaEs Sampler",
        "value": CmaEsSampler,
    },
    "rand": {
        "label": "Random Sampler",
        "value": RandomSampler,
        "is_random": True,
    },
    "qmc": qmc_sampler,
}

optimization_target_funcs = {
    "gpl": {
        "label": "Gross Profit/Loss",
        "value": lambda s: s["global_strategy_stats"]["Gross Profit/Loss"],
        "default_checked": True,
        "is_trades_stats_needed": False,
    },
    "turn_bot": {
        "label": "Turnover BOT",
        "value": lambda s: s["global_strategy_stats"]["Turnover BOT"],
        "is_trades_stats_needed": False,
    },
    "turn_sold": {
        "label": "Turnover SOLD",
        "value": lambda s: s["global_strategy_stats"]["Turnover SOLD"],
        "is_trades_stats_needed": False,
    },
    "max_dd": {
        "label": "Max DD",
        "value": lambda s: s["global_strategy_stats"]["Max DD"],
        "is_trades_stats_needed": False,
    },
    "ret_dd_rat": {
        "label": "Return/DD Ratio",
        "value": lambda s: s["global_strategy_stats"]["Return/DD Ratio"],
        "is_trades_stats_needed": False,
    },
    "max_dd_dur": {
        "label": "Max DD duration",
        "value": lambda s: s["global_strategy_stats"]["Max DD duration"],
        "is_trades_stats_needed": False,
    },
    "ann_ret": {
        "label": "Annual Return",
        "value": lambda s: s["global_strategy_stats"]["Annual Return"],
        "is_trades_stats_needed": False,
    },
    "ann_volat": {
        "label": "Annualized Volatility",
        "value": lambda s: s["global_strategy_stats"]["Annualized Volatility"],
        "is_trades_stats_needed": False,
    },
    "inf_rat": {
        "label": "Information Ratio",
        "value": lambda s: s["global_strategy_stats"]["Information Ratio"],
        "is_trades_stats_needed": False,
    },
    "num_trades": {
        "label": "# Trades",
        "value": lambda s: s["global_stats_by_symbol"]["# Trades"],
        "is_trades_stats_needed": True,
    },
    "win_trades": {
        "label": "Winning Trades",
        "value": lambda s: s["global_stats_by_symbol"]["Winning Trades"],
        "is_trades_stats_needed": True,
    },
    "los_trades": {
        "label": "Losing Trades",
        "value": lambda s: s["global_stats_by_symbol"]["Losing Trades"],
        "is_trades_stats_needed": True,
    },
    "win_rat": {
        "label": "Win Ratio",
        "value": lambda s: s["global_stats_by_symbol"]["Win Ratio"],
        "is_trades_stats_needed": True,
    },
    "avg_trade": {
        "label": "Avg. Trade",
        "value": lambda s: s["global_stats_by_symbol"]["Avg. Trade"],
        "is_trades_stats_needed": True,
    },
    "avg_win_trade": {
        "label": "Avg. Winning Trade",
        "value": lambda s: s["global_stats_by_symbol"]["Avg. Winning Trade"],
        "is_trades_stats_needed": True,
    },
    "avg_los_trade": {
        "label": "Avg. Losing Trade",
        "value": lambda s: s["global_stats_by_symbol"]["Avg. Losing Trade"],
        "is_trades_stats_needed": True,
    },
    "pf_rat": {
        "label": "Payoff Ratio",
        "value": lambda s: s["global_stats_by_symbol"]["Payoff Ratio"],
        "is_trades_stats_needed": True,
    },
    "prof_marg": {
        "label": "Profit Margin (Bps)",
        "value": lambda s: s["global_stats_by_symbol"]["Profit Margin (Bps)"],
        "is_trades_stats_needed": True,
    },
    "avg_hold_d": {
        "label": "Avg. Holding Days",
        "value": lambda s: s["global_stats_by_symbol"]["Avg. Holding Days"],
        "is_trades_stats_needed": True,
    },
}

alerts_backend_proxy_singleton = AlertsBackendProxy.get_instance(
    ALERTS_BACKEND_SERVICE_URL, ALERTS_BACKEND_SERVICE_LIMITS_ENABLED
)

user_prompt_types = [
    {"label": "Generic", "query_param": UserPromptTypes.Generic},
    {"label": "Indicator block", "query_param": UserPromptTypes.IndicatorBlockOnly},
    {"label": "Trading block", "query_param": UserPromptTypes.TradingBlockOnly},
]
