import json
from collections import defaultdict
from typing import Any, Dict, List

import numpy as np
import optuna
import pandas as pd
from fastapi import APIRouter, Depends
from plotly.utils import PlotlyJSONEncoder

from market_alerts.containers import data_providers
from market_alerts.domain.services import (
    build_trade_stats_data,
    visualize_optim_plot_param_importances,
    visualize_optim_plot_slice,
    visualize_strategy_stats,
    visualize_trading_nodes,
    visualize_tree_data_nodes,
)
from market_alerts.entrypoints.llm_backend.domain.exceptions import (
    BacktestingNotPerformedError,
    DataNotFetchedError,
    DatasetNotRequestedError,
    IndicatorsNotGeneratedError,
    OptimizationNotPerformedError,
)
from market_alerts.entrypoints.llm_backend.infrastructure.session import Session, Steps

from ..containers import get_optimization_storage
from .models.plots import DataRequestModel, TradingDataRequestModel
from .utils import get_session

plots_router = APIRouter(prefix="/plots")


@plots_router.get("/prices", response_model=dict[str, list[dict[str, Any]]], tags=["Plots"])
def get_prices_plots_data(request_model: DataRequestModel = Depends(), session: Session = Depends(get_session)):
    if not session.flow_status.is_step_done(Steps.FETCH_TICKERS):
        raise DataNotFetchedError

    plots_meta = session.flow_status.step_fetch["plots_meta"]["symbols"]

    for symbol in request_model.symbols:
        if symbol not in plots_meta:
            raise DataNotFetchedError(f"Data for '{symbol}' wasn't fetched")

    data: dict[str, pd.DataFrame] = {
        **session["data_by_symbol"],
        **session.get("data_by_synth", {}),
    }

    response = defaultdict(list)

    for symbol in request_model.symbols:
        dividends_lines: list[str] = [line["name"] for chart in plots_meta[symbol]["charts"] for line in chart]
        dividends_lines.remove("close")
        keys_to_add = ["open", "high", "low", "close"] + dividends_lines

        for idx, row in data[symbol].iterrows():
            info = {key: (row[key] if not np.isnan(row[key]) else None) for key in keys_to_add if key in row}
            info["date"] = idx.strftime("%Y-%m-%d")
            response[symbol].append(info)

    return response


@plots_router.get("/balance_sheets", response_model=Dict[str, Any], tags=["Plots"])
def get_balance_sheets(session: Session = Depends(get_session)):
    if not session.flow_status.is_step_done(Steps.FETCH_TICKERS):
        raise DataNotFetchedError
    if "bal" not in session["datasets_keys"]:
        raise DatasetNotRequestedError

    primary_data_provider_id = session["data_provider"]
    provider = data_providers[primary_data_provider_id]
    balance_sheets, balance_currencies = provider.get_balance_sheets(session["symbols"], session["true_symbols"])

    return {"balance_sheets": balance_sheets, "balance_currencies": balance_currencies}


@plots_router.get("/income_statements", response_model=Dict[str, Any], tags=["Plots"])
def get_income_statements(session: Session = Depends(get_session)):
    if not session.flow_status.is_step_done(Steps.FETCH_TICKERS):
        raise DataNotFetchedError
    if "inc" not in session["datasets_keys"]:
        raise DatasetNotRequestedError

    primary_data_provider_id = session["data_provider"]
    provider = data_providers[primary_data_provider_id]
    income_statements, income_currencies = provider.get_income_statements(session["symbols"], session["true_symbols"])

    return {"income_statements": income_statements, "income_currencies": income_currencies}


@plots_router.get("/indicators", response_model=List[Dict[str, Any]], tags=["Plots"])
def get_indicators_plots(session: Session = Depends(get_session)):
    if not session.flow_status.is_step_done(Steps.CALCULATE_INDICATORS):
        raise IndicatorsNotGeneratedError

    indicators_plots = visualize_tree_data_nodes(
        {
            **session["data_by_symbol"],
            **session.get("data_by_synth", {}),
            **session.get("data_by_indicator", {}),
        },
        session["roots"],
        session["main_roots"],
        "Indicators",
        session["interval"],
    )

    plots_json = [json.loads(json.dumps(f, cls=PlotlyJSONEncoder)) for f in indicators_plots]

    return plots_json


@plots_router.get("/indicators/v2", response_model=dict[str, list[dict[str, Any]]], tags=["Plots"])
def get_indicators_plots_data(request_model: DataRequestModel = Depends(), session: Session = Depends(get_session)):
    if not session.flow_status.is_step_done(Steps.CALCULATE_INDICATORS):
        raise IndicatorsNotGeneratedError

    plots_meta = session.flow_status.step_indicators["plots_meta"]

    for symbol in request_model.symbols:
        if symbol not in plots_meta:
            raise DataNotFetchedError(f"Indicator '{symbol}' wasn't calculated")

    data: dict[str, pd.DataFrame | pd.Series] = {
        **session["data_by_symbol"],
        **session.get("data_by_synth", {}),
        **session.get("data_by_indicator", {}),
    }

    response = defaultdict(list)

    for symbol in request_model.symbols:
        symbol_data = data[symbol]

        if isinstance(symbol_data, pd.DataFrame):
            indicators_lines: list[str] = [line["name"] for chart in plots_meta[symbol]["charts"] for line in chart]
            indicators_lines.remove("close")
            ohlc_lines = ["open", "high", "low", "close"]

            for idx, row in symbol_data.iterrows():
                info = {
                    ohlc_line: (row[ohlc_line] if not np.isnan(row[ohlc_line]) else None)
                    for ohlc_line in ohlc_lines
                    if ohlc_line in row
                }
                for indicator_line in indicators_lines:
                    info[indicator_line] = data[indicator_line][idx] if not np.isnan(data[indicator_line][idx]) else None

                info["date"] = idx.strftime("%Y-%m-%d")

                response[symbol].append(info)
        elif isinstance(symbol_data, pd.Series):
            continue
        else:
            raise RuntimeError("symbol data was neither pd.DataFrame, nor pd.Series")

    return response


@plots_router.get("/trading_symbols", response_model=List[Dict[str, Any]], tags=["Plots"])
def get_trading_symbols_plots(session: Session = Depends(get_session)):
    if not session.flow_status.is_step_done(Steps.PERFORM_BACKTESTING):
        raise BacktestingNotPerformedError

    trading_symbols_plots = visualize_trading_nodes(
        session["data_by_symbol"],
        session["trading_stats_by_symbol"],
        session["long_alert"],
        session["short_alert"],
        "",
        1,
        session["interval"],
    )

    plots_json = [json.loads(json.dumps(f, cls=PlotlyJSONEncoder)) for f in trading_symbols_plots]

    return plots_json


@plots_router.get("/trading_symbols/v2", response_model=dict[str, list[dict[str, Any]]], tags=["Plots"])
def get_trading_symbols_plots(request_model: TradingDataRequestModel = Depends(), session: Session = Depends(get_session)):
    if not session.flow_status.is_step_done(Steps.PERFORM_BACKTESTING):
        raise BacktestingNotPerformedError

    for symbol in request_model.symbols:
        if symbol not in session.flow_status.step_fetch["plots_meta"]["symbols"]:
            raise DataNotFetchedError(f"Data for '{symbol}' wasn't fetched")

    # data: dict[str, pd.DataFrame | pd.Series] = {
    #     **session["data_by_symbol"],
    #     **session["trading_stats_by_symbol"],
    #     **session["long_alert"],
    #     **session["short_alert"],
    #     # TODO: need synth or not?
    #     # **session.get("data_by_synth", {}),
    # }

    response = defaultdict(list)

    for symbol in request_model.symbols:
        symbol_data, symbol_trading_stats, symbol_buy_alerts, symbol_sell_alerts = (
            session["data_by_symbol"][symbol],
            session["trading_stats_by_symbol"][symbol],
            session["long_alert"][symbol],
            session["short_alert"][symbol],
        )

        symbol_indicators = []
        if request_model.is_indicators_needed:
            indicators_charts = session.flow_status.step_indicators.get("plots_meta", {}).get(symbol, {"charts": []})["charts"]
            for chart in indicators_charts:
                for line in chart:
                    if line["type"] == "indicator":
                        symbol_indicators.append(line["name"])

        trading_charts_lines_mappings = {"pnl": "acct_ccy_pnl", "market_value": "acct_ccy_value"}
        ohlc_lines = ["open", "high", "low", "close"]

        for idx, row in symbol_data.iterrows():
            info = {
                ohlc_line: (row[ohlc_line] if not np.isnan(row[ohlc_line]) else None)
                for ohlc_line in ohlc_lines
                if ohlc_line in row
            }
            for frontend_key, df_key in trading_charts_lines_mappings.items():
                trading_stat = symbol_trading_stats[df_key][idx]
                info[frontend_key] = trading_stat if not np.isnan(trading_stat) else None

            info["buy_alert"] = bool(symbol_buy_alerts[idx])
            info["sell_alert"] = bool(symbol_sell_alerts[idx])

            if data_by_indicator := session.get("data_by_indicator", {}):
                for ind in symbol_indicators:
                    indicator_value = data_by_indicator[ind][idx]
                    info[ind] = indicator_value if not np.isnan(indicator_value) else None

            info["date"] = idx.strftime("%Y-%m-%d")

            response[symbol].append(info)

    return response


@plots_router.get("/trading_strategies", response_model=List[Dict[str, Any]], tags=["Plots"])
def get_trading_strategies_plots(session: Session = Depends(get_session)):
    if not session.flow_status.is_step_done(Steps.PERFORM_BACKTESTING):
        raise BacktestingNotPerformedError

    trading_symbols_plots = visualize_strategy_stats(
        session["strategy_stats"],
        "",
        1,
        session["interval"],
    )

    plots_json = [json.loads(json.dumps(f, cls=PlotlyJSONEncoder)) for f in trading_symbols_plots]

    return plots_json


@plots_router.get("/trading_strategy_stats", response_model=Dict[str, Any], tags=["Plots"])
def get_trading_strategy_stats_plots(session: Session = Depends(get_session)):
    if not session.flow_status.is_step_done(Steps.PERFORM_BACKTESTING):
        raise BacktestingNotPerformedError

    data = pd.DataFrame(session["global_strategy_stats"], index=[0])

    # double_precision=10 fixes JSON serialization issues
    json_data = data.to_json(orient="records")

    data_dict = json.loads(json_data)

    return data_dict[0]


@plots_router.get("/trades_stats", tags=["Plots"])
def get_trades_stats_plots(session: Session = Depends(get_session)):
    if not session.flow_status.is_step_done(Steps.PERFORM_BACKTESTING):
        raise BacktestingNotPerformedError

    all_trades = build_trade_stats_data(session["trade_stats_by_symbol"], session["global_stats_by_symbol"], "All trades")
    short_trades = build_trade_stats_data(
        session["short_trade_stats_by_symbol"], session["short_global_trade_stats"], "Short trades"
    )
    long_trades = build_trade_stats_data(session["long_trade_stats_by_symbol"], session["long_global_trade_stats"], "Long trades")

    data = pd.concat([all_trades, short_trades, long_trades])

    return data.to_json(orient="split")


@plots_router.get("/optimization_slice", tags=["Plots"])
def get_optimization_slice_plot(session: Session = Depends(get_session)):
    if not session.flow_status.is_step_done(Steps.OPTIMIZE):
        raise OptimizationNotPerformedError

    studies_names = session.get("optimization_studies_names", [])

    plots = []
    params = None

    for study_name in studies_names:
        study = optuna.load_study(study_name=study_name, storage=get_optimization_storage(pool_size=3, max_overflow=2))
        plots.append(visualize_optim_plot_slice(study, params, mode=2))

    plots_json = [json.loads(json.dumps(f, cls=PlotlyJSONEncoder)) for f in plots]

    return plots_json


@plots_router.get("/optimization_param_importances", tags=["Plots"])
def get_optimization_param_importances_plot(session: Session = Depends(get_session)):
    if not session.flow_status.is_step_done(Steps.OPTIMIZE):
        raise OptimizationNotPerformedError

    studies_names = session.get("optimization_studies_names", [])

    plots = []

    for study_name in studies_names:
        study = optuna.load_study(study_name=study_name, storage=get_optimization_storage(pool_size=3, max_overflow=2))
        plots.append(visualize_optim_plot_param_importances(study))

    plots_json = [json.loads(json.dumps(f, cls=PlotlyJSONEncoder)) for f in plots]

    return plots_json
