from dataclasses import dataclass

from market_alerts.domain.services import (
    define_empty_indicators_step,
    define_useful_strings,
    symbol_step,
    visualize_data_nodes,
    visualize_strategy_stats,
    visualize_trading_nodes,
    visualize_tree_data_nodes,
)


@dataclass
class Experiment:
    data_provider: str
    time_period: int
    interval: int
    tradable_symbols_prompt: str
    supplementary_symbols_prompt: str
    datasets_keys: list[str]
    economic_indicators: list[str]
    dividend_fields: list[str]

    def __getitem__(self, key):
        return getattr(self, key)

    def __setitem__(self, key, value):
        setattr(self, key, value)

    def __contains__(self, key):
        return key in vars(self).keys()

    def get(self, key, default_value=None):
        return vars(self).get(key, default_value)

    def setdefault(self, key, default_value=None):
        if key not in self:
            self.__setitem__(key, default_value)
        return self.get(key)


def draw_price_charts(e: Experiment):
    figs = visualize_data_nodes(
        {
            **getattr(e, "data_by_symbol"),
            **getattr(e, "data_by_synth", {}),
        },
        1,
        e.interval,
    )
    for fig in figs:
        fig.show()


def draw_indicators_charts(e: Experiment):
    figs = visualize_tree_data_nodes(
        {
            **getattr(e, "data_by_symbol"),
            **getattr(e, "data_by_synth", {}),
            **getattr(e, "data_by_indicator", {}),
        },
        getattr(e, "roots"),
        getattr(e, "main_roots"),
        "Indicators",
        e.interval,
    )
    for fig in figs:
        fig.show()


def draw_trading_charts_by_symbol(e: Experiment):
    figs = visualize_trading_nodes(
        getattr(e, "data_by_symbol"),
        getattr(e, "trading_stats_by_symbol"),
        getattr(e, "long_alert"),
        getattr(e, "short_alert"),
        "",
        1,
        "1day",
    )
    for f in figs:
        f.show()


def draw_trading_charts(e: Experiment):
    figs = visualize_strategy_stats(getattr(e, "strategy_stats"), "", 1, "1day")
    for f in figs:
        f.show()


def show_global_strategy_stats(e: Experiment):
    print(getattr(e, "global_strategy_stats"))


def load_data(e: Experiment):
    symbol_step(e)
    define_useful_strings(e)
    define_empty_indicators_step(e)


def build_table(experiment: Experiment, symbols=None, columns=None, condition=None):
    import pandas as pd

    dataframes = []

    if symbols is None:
        symbols = experiment.tradable_symbols

    for symbol in symbols:
        columns_ = columns if columns is not None else experiment.data_by_symbol[symbol].keys()
        for column in columns_:
            if column in experiment.data_by_symbol[symbol].keys():
                df = pd.DataFrame({f"{symbol}_{column}": experiment.data_by_symbol[symbol][column]})
                df.columns.names = ["symbol"]
                dataframes.append(df)

        columns_ = columns if columns is not None else experiment.lclsglbls.keys()
        for column in columns_:
            #            print(column, symbol)
            if (
                column in experiment.lclsglbls.keys()
                and isinstance(experiment.lclsglbls[column], dict)
                and symbol in experiment.lclsglbls[column].keys()
                and isinstance(experiment.lclsglbls[column][symbol], pd.Series)
            ):
                df = pd.DataFrame({f"{symbol}_{column}": experiment.lclsglbls[column][symbol]})
                df.columns.names = ["symbol"]
                dataframes.append(df)

        columns_ = columns if columns is not None else experiment.trading_stats_by_symbol[symbol].keys()
        for column in columns_:
            if (
                symbol in experiment.trading_stats_by_symbol.keys()
                and column in experiment.trading_stats_by_symbol[symbol].keys()
            ):
                df = pd.DataFrame({f"{symbol}_{column}": experiment.trading_stats_by_symbol[symbol][column]})
                df.columns.names = ["symbol"]
                dataframes.append(df)

    columns_ = columns if columns is not None else experiment.strategy_stats.keys()
    for column in columns_:
        if column in experiment.strategy_stats.keys():
            df = pd.DataFrame({f"{column}": experiment.strategy_stats[column]})
            df.columns.names = ["symbol"]
            dataframes.append(df)

    merged = pd.concat(dataframes, axis=1)

    # Apply condition if one is provided
    if condition is not None:
        merged = merged[condition(merged)]

    return merged


def show_dtale(experiment: Experiment, symbols=None, columns=None, condition=None):
    import dtale
    import dtale.app as dtale_app

    dtale_app.JUPYTER_SERVER_PROXY = True
    merged = build_table(experiment, symbols, columns, condition=condition)
    return dtale.show(merged, host="localhost", enable_custom_filters=True)
