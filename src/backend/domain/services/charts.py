import json
import os
from collections import defaultdict
from typing import Any, Dict

import optuna
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from market_alerts.domain.constants import ADDITIONAL_COLUMNS


# calculate spacing between subplots
def get_vertical_spacing(thickness):
    return thickness * 3.2


# calculate range slider thickness
def get_thickness(num_plots):
    return 0.105 / num_plots


# calculate position of update menus for subplots
def get_update_menus_position(vertical_spacing, num_plots, i):
    return (
        ((1 - vertical_spacing * (num_plots - 1)) / num_plots) * (num_plots - i)
        + vertical_spacing * (num_plots - 1 - i)
        + 0.11 / num_plots
    )


def get_lines_per_symbol_mapping(data):
    plots = defaultdict(list)
    for symbol_name in data.keys():
        plots[symbol_name].append("close")
        for a_column in ADDITIONAL_COLUMNS:
            if a_column in data[symbol_name]:
                plots[symbol_name].append(a_column)
    return plots


def visualize_data_nodes(data, num_cols, periodicity, alerts=None):
    if alerts is not None and isinstance(alerts, pd.DataFrame):
        alerts = alerts["close"]

    figs = []

    for symbol_name, columns in get_lines_per_symbol_mapping(data).items():
        for column in columns:
            fig = make_subplots(
                rows=1,
                cols=1,
                subplot_titles=[symbol_name + "." + column],
                horizontal_spacing=0,
                vertical_spacing=0,
            )
            value = data[symbol_name]
            if isinstance(value, pd.DataFrame):
                if value.get(column, None) is None:
                    continue
                value = value[column]
            fig.add_trace(
                go.Scatter(x=value.index, y=value.values, name=symbol_name),
                row=1,
                col=1,
            )
            if alerts is not None and alerts.shape[0] == value.shape[0]:
                fig.add_trace(
                    go.Scatter(
                        x=value[alerts].index,
                        y=value[alerts].values,
                        name=symbol_name,
                        mode="markers",
                        marker=dict(color="red", size=4, symbol="x", line_width=1),
                    ),
                    row=1,
                    col=1,
                )
            simple_updatemenus = [
                dict(
                    buttons=list(
                        [
                            dict(
                                args=[
                                    {
                                        "type": ["scatter"],
                                        "x": [fig.data[0]["x"]],
                                        #                                'y': [fig.data[0]['y']]
                                    },
                                    {"yaxis.type": "linear"},
                                    [0],
                                ],
                                label="Line plot",
                                method="update",
                            ),
                            dict(
                                args=[
                                    {
                                        "type": ["scatter"],
                                        "x": [fig.data[0]["x"]],
                                        #                                'y': [fig.data[0]['y']]
                                    },
                                    {"yaxis.type": "log"},
                                    [0],
                                ],
                                label="Log plot",
                                method="update",
                            ),
                            dict(
                                args=[
                                    {
                                        "type": ["histogram"],
                                        #                                'x': [fig.data[0]['y']],
                                        "x": [None]
                                        #                                'nbinsy': [2 * int(np.sqrt(len(fig.data[0]['x'])))]
                                    },
                                    {"yaxis.type": "linear"},
                                    [0],
                                ],
                                label="Histogram plot",
                                method="update",
                            ),
                        ]
                    ),
                    direction="down",
                    showactive=True,
                    xanchor="left",
                    y=1.15,
                    x=0,
                ),
            ]
            fig.update_layout(
                updatemenus=simple_updatemenus,
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                margin=dict(l=0, r=30, t=30, b=0),
                # title=title,
                width=650 * 1 / num_cols,
                height=400,
            )

            n_iter = (value.index.shape[0] - 1) // 5 + 1
            ttext = value.index[::n_iter].tolist()
            # TODO: Need to cut off zeroes: "2013-12-30T00:00:00"
            if periodicity == 86400000:
                for index, date in enumerate(ttext):
                    ttext[index] = date.strftime("%Y-%m-%d")

            fig.for_each_xaxis(lambda x: x.update(type="date", rangeslider={"thickness": 0.105, "yaxis": {"rangemode": "auto"}}))
            figs.append(fig)

    return figs


def visualize_tree_data_nodes(all_data, roots, main_roots, title, periodicity, alerts=None):
    if alerts is not None and isinstance(alerts, pd.DataFrame):
        alerts = alerts["close"]
    figs = []
    max_height = 0
    fig_confs = []
    fig_subplots_to_traces = []
    for main_root in main_roots:
        roots_on_fig = [main_root] + main_roots[main_root]
        num_plots = len(roots_on_fig)
        num_cols = 1
        num_rows = num_plots
        thickness = get_thickness(num_plots)
        vertical_spacing = get_vertical_spacing(thickness)
        fig = make_subplots(
            rows=num_rows,
            cols=num_cols,
            subplot_titles=roots_on_fig,
            horizontal_spacing=0.1,
            vertical_spacing=vertical_spacing,
            shared_xaxes=False,
        )
        fig_subplots_to_traces.append({i: [] for i in range(num_plots)})
        k = 0
        trace_k = 0
        for root in roots_on_fig:
            key = root
            if "." in root:
                root_, column = root.split(".")[0], root.split(".")[-1]
                value = all_data[root_][column]
            else:
                value = all_data[root]
            if isinstance(value, pd.DataFrame):
                value = value["close"]
            root_indexes = value.index
            fig.add_trace(
                go.Scatter(x=root_indexes, y=value.values, name=key),
                row=(k // num_cols) + 1,
                col=(k % num_cols) + 1,
            )
            fig_subplots_to_traces[-1][k].append(trace_k)
            trace_k += 1
            if alerts is not None and alerts.shape[0] == value.shape[0]:
                fig.add_trace(
                    go.Scatter(
                        x=value[alerts].index,
                        y=value[alerts].values,
                        name=key,
                        mode="markers",
                        marker=dict(color="red", size=4, symbol="x", line_width=1),
                    ),
                    row=(k // num_cols) + 1,
                    col=(k % num_cols) + 1,
                )
                fig_subplots_to_traces[-1][k].append(trace_k)
                trace_k += 1
            for key in roots[root]:
                if "." in key:
                    key_, column = key.split(".")[0], key.split(".")[-1]
                    value = all_data[key_][column]
                else:
                    value = all_data[key]
                if isinstance(value, pd.DataFrame):
                    value = value["close"]
                fig.add_trace(
                    go.Scatter(x=value.index, y=value.values, name=key),
                    row=(k // num_cols) + 1,
                    col=(k % num_cols) + 1,
                )
                fig_subplots_to_traces[-1][k].append(trace_k)
                trace_k += 1

            k += 1
        height = num_plots * 200
        if height > max_height:
            max_height = height

        fig.update_layout(
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            width=800,
            height=height,
        )
        n_iter = (root_indexes.shape[0] - 1) // 5 + 1

        ttext = root_indexes[::n_iter].tolist()
        if periodicity == "1day":
            for index, date in enumerate(ttext):
                ttext[index] = date.strftime("%Y-%m-%d")

        fig.for_each_xaxis(
            lambda x: (x.update(type="date", rangeslider={"thickness": thickness, "yaxis": {"rangemode": "auto"}}))
        )
        fig_confs.append(
            {
                "title": main_root,
                "num_plots": num_plots,
                "fig": fig,
                "subplots_to_traces": fig_subplots_to_traces[-1],
            }
        )

        figs.append(fig)

    for conf in fig_confs:
        height = max_height
        #         height_step = max_height//conf["num_plots"]
        local_updatemenus = []
        #         for i in range(conf["num_plots"]):
        #             local_updatemenus.append(
        #                 dict(
        #                     buttons=list([
        #                         dict(
        #                             args=[{('yaxis%d.type' % (i + 1) if i != 0 else 'yaxis.type'): 'linear'}],
        #                             label="Linear Scale",
        #                             method="relayout"
        #                         ),
        #                         dict(
        #                             args=[{('yaxis%d.type' % (i + 1) if i != 0 else 'yaxis.type'): 'log'}],
        #                             label="Log Scale",
        #                             method="relayout"
        #                         )
        #                     ]),
        #                     direction="down",
        #                     showactive=True,
        #                     xanchor="left",
        #                     y=(1. - i / conf["num_plots"]),
        #                 )
        #             )
        #             local_updatemenus.append(
        #                 dict(
        #                     buttons=list([
        #                         dict(
        #                             args=[{'type': ['scatter'] * len(conf["subplots_to_traces"][i]),
        #                                    'x': [conf['fig'].data[j]['x'] for j in conf["subplots_to_traces"][i]],
        # #                                    'y': [conf['fig'].data[j]['y'] for j in conf["subplots_to_traces"][i]],
        #                                   },
        #                                   conf["subplots_to_traces"][i]],
        #                             label="Line plot",
        #                             method="restyle"
        #                         ),
        #                         dict(
        #                             args=[{'type': ['histogram'] * len(conf["subplots_to_traces"][i]),
        #                                    'x': [None] * len(conf["subplots_to_traces"][i]),
        # #                                    'y': [None] * len(conf["subplots_to_traces"][i]),
        # #                                    'x': [conf['fig'].data[j]['y'] for j in conf["subplots_to_traces"][i]],
        # #                                    'nbinsy': [2 * int(np.sqrt(len(conf['fig'].data[0]['x'])))] * len(conf["subplots_to_traces"][i])
        # #                                    'nbinsx': [20] * len(conf["subplots_to_traces"][i]),
        #                                   },
        #                                   conf["subplots_to_traces"][i]],
        #                             label="Histogram plot",
        #                             method="restyle"
        #                         ),
        #                     ]),
        #                     direction="down",
        #                     showactive=True,
        #                     xanchor="left",
        #                     y=(1. - i / conf["num_plots"]) - 0.1,
        #                 )
        #             )
        for i in range(conf["num_plots"]):
            local_updatemenus.append(
                dict(
                    buttons=list(
                        [
                            dict(
                                args=[
                                    {
                                        "type": ["scatter"] * len(conf["subplots_to_traces"][i]),
                                        "x": [conf["fig"].data[j]["x"] for j in conf["subplots_to_traces"][i]],
                                    },
                                    {
                                        ("yaxis%d.type" % (i + 1) if i != 0 else "yaxis.type"): "linear",
                                        ("xaxis%d.type" % (i + 1) if i != 0 else "xaxis.type"): "date",
                                    },
                                    conf["subplots_to_traces"][i],
                                ],
                                label="Line plot",
                                method="update",
                            ),
                            dict(
                                args=[
                                    {
                                        "type": ["scatter"] * len(conf["subplots_to_traces"][i]),
                                        "x": [conf["fig"].data[j]["x"] for j in conf["subplots_to_traces"][i]],
                                    },
                                    {
                                        ("yaxis%d.type" % (i + 1) if i != 0 else "yaxis.type"): "log",
                                        ("xaxis%d.type" % (i + 1) if i != 0 else "xaxis.type"): "date",
                                    },
                                    conf["subplots_to_traces"][i],
                                ],
                                label="Log plot",
                                method="update",
                            ),
                            dict(
                                args=[
                                    {
                                        "type": ["histogram"] * len(conf["subplots_to_traces"][i]),
                                        "x": [None] * len(conf["subplots_to_traces"][i]),
                                    },
                                    {
                                        ("yaxis%d.type" % (i + 1) if i != 0 else "yaxis.type"): "linear",
                                        ("xaxis%d.type" % (i + 1) if i != 0 else "xaxis.type"): "category",
                                    },
                                    conf["subplots_to_traces"][i],
                                ],
                                label="Histogram plot",
                                method="update",
                            ),
                        ]
                    ),
                    direction="down",
                    showactive=True,
                    xanchor="left",
                    y=get_update_menus_position(vertical_spacing, conf["num_plots"], i),
                    x=0,
                )
            )
        conf["fig"].update_layout(
            updatemenus=local_updatemenus,
            height=height,
            margin=dict(l=0, r=30, t=20, b=0),
        )
    return [i["fig"] for i in fig_confs]


def visualize_trading_nodes(data_by_symbol, trading_stats_by_symbol, long_alert, short_alert, title, num_cols, periodicity):
    num_plots = len(trading_stats_by_symbol)
    figs = []

    for title in trading_stats_by_symbol:
        thickness = get_thickness(3)
        vertical_spacing = get_vertical_spacing(thickness)
        fig = make_subplots(
            rows=3,
            cols=1,
            subplot_titles=[title + ".close", title + ".pnl", title + ".market_value"],
            horizontal_spacing=0.1,
            vertical_spacing=vertical_spacing,
            shared_xaxes=False,
        )
        k = 0
        value = data_by_symbol[title]["close"]

        fig.add_trace(
            go.Scatter(x=value.index, y=value.values, name=title),
            row=1,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=value[long_alert[title]].index,
                y=value[long_alert[title]].values,
                name=title + ".buy_alert",
                mode="markers",
                marker=dict(color="green", size=6, symbol="arrow-up", line_width=1),
            ),
            row=1,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=value[short_alert[title]].index,
                y=value[short_alert[title]].values,
                name=title + ".sell_alert",
                mode="markers",
                marker=dict(color="red", size=6, symbol="arrow-down", line_width=1),
            ),
            row=1,
            col=1,
        )
        pnl_value = trading_stats_by_symbol[title]["acct_ccy_pnl"]
        fig.add_trace(
            go.Scatter(x=pnl_value.index, y=pnl_value.values, name=title),
            row=2,
            col=1,
        )
        acc_value = trading_stats_by_symbol[title]["acct_ccy_value"]

        fig.add_trace(
            go.Scatter(x=acc_value.index, y=acc_value.values, name=title),
            row=3,
            col=1,
        )
        k += 1
        simple_updatemenus = [
            dict(
                buttons=list(
                    [
                        dict(
                            args=[
                                {
                                    "type": ["scatter", "scatter", "scatter"],
                                    "x": [fig.data[0]["x"], fig.data[1]["x"], fig.data[2]["x"]],
                                },
                                {"yaxis.type": "linear", "xaxis.type": "date"},
                                [0, 1, 2],
                            ],
                            label="Line plot",
                            method="update",
                        ),
                        dict(
                            args=[
                                {
                                    "type": ["scatter", "scatter", "scatter"],
                                    "x": [fig.data[0]["x"], fig.data[1]["x"], fig.data[2]["x"]],
                                },
                                {"yaxis.type": "log", "xaxis.type": "date"},
                                [0, 1, 2],
                            ],
                            label="Log plot",
                            method="update",
                        ),
                        dict(
                            args=[
                                {"type": ["histogram", "histogram", "histogram"], "x": [None, None, None]},
                                {"yaxis.type": "linear", "xaxis.type": "category"},
                                [0, 1, 2],
                            ],
                            label="Histogram plot",
                            method="update",
                        ),
                    ]
                ),
                direction="down",
                showactive=True,
                xanchor="left",
                y=get_update_menus_position(vertical_spacing, 3, 0),
                x=0,
            ),
        ]
        simple_updatemenus.append(
            dict(
                buttons=list(
                    [
                        dict(
                            args=[
                                {
                                    "type": ["scatter"],
                                    "x": [fig.data[3]["x"]],
                                },
                                {"yaxis2.type": "linear", "xaxis2.type": "date"},
                                [3],
                            ],
                            label="Line plot",
                            method="update",
                        ),
                        dict(
                            args=[
                                {
                                    "type": ["scatter"],
                                    "x": [fig.data[3]["x"]],
                                },
                                {"yaxis2.type": "log", "xaxis2.type": "date"},
                                [3],
                            ],
                            label="Log plot",
                            method="update",
                        ),
                        dict(
                            args=[
                                {"type": ["histogram"], "x": [None]},
                                {"yaxis2.type": "linear", "xaxis2.type": "category"},
                                [3],
                            ],
                            label="Histogram plot",
                            method="update",
                        ),
                    ]
                ),
                direction="down",
                showactive=True,
                xanchor="left",
                y=get_update_menus_position(vertical_spacing, 3, 1),
                x=0,
            ),
        )
        simple_updatemenus.append(
            dict(
                buttons=list(
                    [
                        dict(
                            args=[
                                {
                                    "type": ["scatter"],
                                    "x": [fig.data[4]["x"]],
                                },
                                {"yaxis3.type": "linear", "xaxis3.type": "date"},
                                [4],
                            ],
                            label="Line plot",
                            method="update",
                        ),
                        dict(
                            args=[
                                {
                                    "type": ["scatter"],
                                    "x": [fig.data[4]["x"]],
                                },
                                {"yaxis3.type": "log", "xaxis3.type": "date"},
                                [4],
                            ],
                            label="Log plot",
                            method="update",
                        ),
                        dict(
                            args=[
                                {"type": ["histogram"], "x": [None]},
                                {"yaxis3.type": "linear", "xaxis3.type": "category"},
                                [4],
                            ],
                            label="Histogram plot",
                            method="update",
                        ),
                    ]
                ),
                direction="down",
                showactive=True,
                xanchor="left",
                y=get_update_menus_position(vertical_spacing, 3, 2),
                x=0,
            ),
        )
        fig.update_layout(
            updatemenus=simple_updatemenus,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            margin=dict(l=0, r=30, t=20, b=0),
            # title=title,
            width=650 * 1 / num_cols,
            height=600,
        )

        n_iter = (value.index.shape[0] - 1) // 5 + 1
        ttext = value.index[::n_iter].tolist()
        if periodicity == "1day":
            for index, date in enumerate(ttext):
                ttext[index] = date.strftime("%Y-%m-%d")

        fig.for_each_xaxis(lambda x: x.update(type="date", rangeslider={"thickness": thickness, "yaxis": {"rangemode": "auto"}}))
        figs.append(fig)

    return figs


def visualize_strategy_stats(strategy_stats, title, num_cols, periodicity):
    num_plots = 3
    thickness = get_thickness(num_plots)
    vertical_spacing = get_vertical_spacing(thickness)

    title = "strategy"
    fig = make_subplots(
        rows=3,
        cols=1,
        subplot_titles=[title + ".pnl", title + ".market_value", title + ".gross_value"],
        horizontal_spacing=0.1,
        vertical_spacing=vertical_spacing,
        shared_xaxes=False,
    )
    value = strategy_stats["acct_ccy_pnl"]

    fig.add_trace(
        go.Scatter(x=value.index, y=value.values, name="pnl"),
        row=1,
        col=1,
    )
    acc_value = strategy_stats["acct_ccy_value"]

    fig.add_trace(
        go.Scatter(x=acc_value.index, y=acc_value.values, name="market_value"),
        row=2,
        col=1,
    )
    gross_value = strategy_stats["gross_value"]

    fig.add_trace(
        go.Scatter(x=gross_value.index, y=gross_value.values, name="gross_value"),
        row=3,
        col=1,
    )

    simple_updatemenus = []
    for i in range(3):
        simple_updatemenus.append(
            dict(
                buttons=list(
                    [
                        dict(
                            args=[
                                {
                                    "type": ["scatter"],
                                    "x": [fig.data[i]["x"]],
                                },
                                {
                                    ("yaxis%d.type" % (i + 1) if i != 0 else "yaxis.type"): "linear",
                                    ("xaxis%d.type" % (i + 1) if i != 0 else "xaxis.type"): "date",
                                },
                                [
                                    i,
                                ],
                            ],
                            label="Line plot",
                            method="update",
                        ),
                        dict(
                            args=[
                                {
                                    "type": ["scatter"],
                                    "x": [fig.data[i]["x"]],
                                },
                                {
                                    ("yaxis%d.type" % (i + 1) if i != 0 else "yaxis.type"): "log",
                                    ("xaxis%d.type" % (i + 1) if i != 0 else "xaxis.type"): "date",
                                },
                                [i],
                            ],
                            label="Log plot",
                            method="update",
                        ),
                        dict(
                            args=[
                                {"type": ["histogram"], "x": [None]},
                                {
                                    ("yaxis%d.type" % (i + 1) if i != 0 else "yaxis.type"): "linear",
                                    ("xaxis%d.type" % (i + 1) if i != 0 else "xaxis.type"): "category",
                                },
                                [i],
                            ],
                            label="Histogram plot",
                            method="update",
                        ),
                    ]
                ),
                direction="down",
                showactive=True,
                xanchor="left",
                y=get_update_menus_position(vertical_spacing, 3, i),
                x=0,
            ),
        )

    fig.update_layout(
        updatemenus=simple_updatemenus,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=0, r=30, t=20, b=0),
        # title=title,
        width=650 * 1 / num_cols,
        height=400,
    )

    n_iter = (value.index.shape[0] - 1) // 5 + 1
    ttext = value.index[::n_iter].tolist()
    if periodicity == "1day":
        for index, date in enumerate(ttext):
            ttext[index] = date.strftime("%Y-%m-%d")

    fig.for_each_xaxis(lambda x: x.update(type="date", rangeslider={"thickness": thickness, "yaxis": {"rangemode": "auto"}}))

    return [fig]


def create_pnl_report(session_dict) -> str:
    return pd.concat(
        [
            session_dict["strategy_stats"]["acct_ccy_day_pnl"],
            session_dict["strategy_stats"]["acct_ccy_pnl"],
            session_dict["strategy_stats"]["acct_ccy_total_cost"],
            session_dict["strategy_stats"]["acct_ccy_value"],
            session_dict["strategy_stats"]["gross_value"],
            session_dict["strategy_stats"]["NOP_value"],
        ],
        axis=1,
    ).to_csv()


def create_stats_by_symbol_report(data: Dict[str, Any]) -> str:
    for ticker, df in data.items():
        df["ticker"] = [ticker] * df.shape[0]

    return pd.concat([value for value in data.values()]).to_csv()


def build_trade_stats_data(trade_stats_by_symbol, global_stats_by_symbol, trades_type):
    for ticker in trade_stats_by_symbol:
        type_temp = {"Trades type": trades_type}
        type_temp.update(trade_stats_by_symbol[ticker])
        trade_stats_by_symbol[ticker] = type_temp.copy()
    trading_stats_all_df = pd.DataFrame.from_dict(trade_stats_by_symbol, orient="index")
    type_temp = {"Trades type": trades_type}
    type_temp.update(global_stats_by_symbol)
    global_stats_by_symbol = type_temp.copy()
    strategy_stats_df = pd.DataFrame.from_dict({"Strategy": global_stats_by_symbol}, orient="index")
    return pd.concat([trading_stats_all_df, strategy_stats_df])


def read_chart_layout() -> dict:
    script_dir = os.path.dirname(os.path.abspath(__file__))
    config_file = os.path.join(script_dir, "chart_layout.json")

    if not os.path.exists(config_file):
        open(config_file, "w").close()

    with open(config_file, "r+") as file:
        content = file.read()
        if content:
            config = json.loads(content)
        else:
            config = dict()

    return config


def write_chart_layout(
    price_chart_col_number=None,
    indicators_chart_col_number=None,
    alerts_chart_col_number=None,
    trading_chart_col_number=None,
    trading_strategy_chart_col_number=None,
) -> dict:
    script_dir = os.path.dirname(os.path.abspath(__file__))
    config_file = os.path.join(script_dir, "chart_layout.json")

    if not os.path.exists(config_file):
        open(config_file, "w").close()

    with open(config_file, "r+") as file:
        content = file.read()
        if content:
            config = json.loads(content)
        else:
            config = dict()
        if price_chart_col_number is not None:
            config["price_chart_col_number"] = price_chart_col_number
        if indicators_chart_col_number is not None:
            config["indicators_chart_col_number"] = indicators_chart_col_number
        if alerts_chart_col_number is not None:
            config["alerts_chart_col_number"] = alerts_chart_col_number
        if trading_chart_col_number is not None:
            config["trading_chart_col_number"] = trading_chart_col_number
        if trading_strategy_chart_col_number is not None:
            config["trading_strategy_chart_col_number"] = trading_strategy_chart_col_number
        config_str = json.dumps(config)
        file.seek(0)
        file.truncate()
        file.write(config_str)

    return config


def visualize_optim_plot_slice(study, params: list[str] | None, mode: int = 0):
    # In-Sample
    if mode == 0:
        return optuna.visualization.plot_slice(study, params=params, target_name="In-Sample Objective Value")
    # Out-of-Sample
    elif mode == 1:
        return optuna.visualization.plot_slice(
            study,
            params=params,
            target=lambda x: x.user_attrs["test_value"],
            target_name="Out-of-Sample Objective Value",
        )
    # Both
    slice_in_sample = optuna.visualization.plot_slice(study, params=params, target_name="Objective Value")
    slice_out_of_sample = optuna.visualization.plot_slice(
        study,
        params=params,
        target=lambda x: x.user_attrs["test_value"],
        target_name="Out-of-Sample Objective Value",
    )
    slice_in_sample.data[0].marker["colorbar"]["title"]["text"] = "In Trial"
    slice_in_sample.data[0].marker["colorbar"]["xpad"] = 20
    slice_out_of_sample.data[0].marker["colorbar"]["title"]["text"] = "Out Trial"
    slice_out_of_sample.data[0].marker["colorbar"]["xpad"] = 100

    for i in range(len(slice_in_sample.data)):
        slice_in_sample.data[i].name = "in sample"

    for i in range(len(slice_out_of_sample.data)):
        slice_out_of_sample.data[i].name = "out of sample"
        slice_out_of_sample.data[i].marker["colorscale"] = (
            (0.0, "rgb(255,247,251)"),
            (0.125, "rgb(247,222,235)"),
            (0.25, "rgb(239,198,219)"),
            (0.375, "rgb(225,158,202)"),
            (0.5, "rgb(214,107,174)"),
            (0.625, "rgb(198,66,146)"),
            (0.75, "rgb(181,33,113)"),
            (0.875, "rgb(156,8,81)"),
            (1.0, "rgb(107,8,48)"),
        )
    return go.Figure(data=slice_in_sample.data + slice_out_of_sample.data, layout=slice_in_sample.layout)


def visualize_optim_plot_history(study, mode: int = 0):
    # In-Sample
    if mode == 0:
        return optuna.visualization.plot_optimization_history(study, target_name="In-Sample Objective Value")
    # Out-of-Sample
    elif mode == 1:
        return optuna.visualization.plot_optimization_history(
            study,
            target=lambda x: x.user_attrs["test_value"],
            target_name="Out-of-Sample Objective Value",
        )
    opt_hist_in_sample = optuna.visualization.plot_optimization_history(study, target_name="In-Sample Objective Value")
    opt_hist_out_of_sample = optuna.visualization.plot_optimization_history(
        study,
        target=lambda x: x.user_attrs["test_value"],
        target_name="Out-of-Sample Objective Value",
    )
    opt_hist_in_sample.layout.yaxis.title["text"] = "Objective Value"
    return go.Figure(data=opt_hist_in_sample.data + opt_hist_out_of_sample.data, layout=opt_hist_in_sample.layout)


def visualize_optim_plot_param_importances(study):
    return optuna.visualization.plot_param_importances(study)
