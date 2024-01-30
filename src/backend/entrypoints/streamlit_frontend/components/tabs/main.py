import json
import math
import re
from itertools import chain

import pandas as pd
import streamlit as st
from st_aggrid import AgGrid, ColumnsAutoSizeMode, GridUpdateMode
from streamlit_monaco import st_monaco

from market_alerts.config import (
    JUPYTERHUB_SERVICE_URL,
    JUPYTERHUB_TOKEN,
    JUPYTERHUB_UI_URL,
)
from market_alerts.containers import (
    alerts_backend_proxy_singleton,
    data_providers,
    datasets,
    trade_fill_prices,
)
from market_alerts.domain.constants import (
    CODE_EDITOR_MAX_HEIGHT,
    DATA_PERIODICITIES_NAMES_TO_BACKEND_KEY,
    DATA_PERIODICITIES_NAMES_TO_VALUES,
    DATA_TIME_RANGES_NAMES_TO_BACKEND_KEY,
    DEFAULT_MODEL_NAME,
    TRIGGER_TYPES,
)
from market_alerts.domain.exceptions import DataNotFoundError, LLMBadResponseError
from market_alerts.domain.services import (
    JupyterService,
    alert_step,
    build_trade_stats_data,
    create_pnl_report,
    create_stats_by_symbol_report,
    indicator_step,
    visualize_data_nodes,
    visualize_strategy_stats,
    visualize_trading_nodes,
    visualize_tree_data_nodes,
)
from market_alerts.domain.services.code import get_code_sections
from market_alerts.entrypoints.streamlit_frontend.constants import (
    DATA_PROVIDERS_NAMES_TO_BACKEND_KEY,
)
from market_alerts.entrypoints.streamlit_frontend.state import (
    add_symbol_button_callback,
    alert_submit_callback,
    alerts_add_column_callback,
    alerts_reset_layout_callback,
    backtest_callback,
    build_aggrid_options,
    check_code_block,
    check_prices_multisellect_result,
    clear_alerts_chat_history,
    clear_indicators_chat_history,
    fetch_data,
    indicator_submit_callback,
    indicators_add_column_callback,
    indicators_reset_layout_callback,
    price_add_column_callback,
    price_reset_layout_callback,
    publish_alerts,
    publish_trading_rule,
    send_trading_model_callback,
    trading_rule_import_callback,
    trading_symbols_add_column_callback,
    trading_symbols_reset_layout_callback,
)
from market_alerts.openai_utils import get_available_models
from market_alerts.utils import func_profile, session_state_getter


def draw_trading_symbol_charts():
    if len(st.session_state.trading_symbols_figs) == 0:
        st.session_state.trading_symbols_figs = visualize_trading_nodes(
            st.session_state["data_by_symbol"],
            st.session_state["trading_stats_by_symbol"],
            st.session_state["long_alert"],
            st.session_state["short_alert"],
            "",
            1,
            list(DATA_PERIODICITIES_NAMES_TO_VALUES.values())[st.session_state.data_periodicity_index],
        )

    if st.session_state.num_cols_tradings > len(st.session_state.trading_symbols_figs):
        st.session_state.num_cols_tradings = len(st.session_state.trading_symbols_figs)

    num_rows_tradings = math.ceil(len(st.session_state.trading_symbols_figs) / st.session_state["num_cols_tradings"])

    chart_width_percent = 1 / st.session_state["num_cols_tradings"]

    cols_spec = [chart_width_percent] * st.session_state["num_cols_tradings"]
    cols = []

    with st.expander("**Click to display/hide trading charts by symbol**"):
        col1, col2, _ = st.columns([0.1, 0.1, 0.8], gap="small")

        reset_layout = col1.empty()
        add_column = col2.empty()

        reset_layout.button(
            "Reset layout",
            on_click=trading_symbols_reset_layout_callback,
            key="reset_trading_by_symbol_layout",
        )
        add_column.button(
            "Split layout",
            on_click=trading_symbols_add_column_callback,
            key="split_trading_by_symbol_layout",
        )

        for _ in range(num_rows_tradings):
            cols.append(st.columns(cols_spec, gap="small"))

        figs_containers = []
        for row in cols:
            figs_containers.append([item.empty() for item in row])

        for index, fig in enumerate(st.session_state.trading_symbols_figs):
            row_counter = 0
            while index + 1 > st.session_state.num_cols_tradings:
                index = index - st.session_state.num_cols_tradings
                row_counter += 1

            figs_containers[row_counter][index].plotly_chart(fig, use_container_width=True)


def draw_strategy_chart():
    if len(st.session_state.trading_strategy_figs) == 0:
        st.session_state.trading_strategy_figs = visualize_strategy_stats(
            st.session_state["strategy_stats"],
            "",
            1,
            list(DATA_PERIODICITIES_NAMES_TO_VALUES.values())[st.session_state.data_periodicity_index],
        )

    if st.session_state.num_cols_strategy_tradings > len(st.session_state.trading_strategy_figs):
        st.session_state.num_cols_strategy_tradings = len(st.session_state.trading_strategy_figs)

    num_rows_tradings = math.ceil(len(st.session_state.trading_strategy_figs) / st.session_state["num_cols_strategy_tradings"])

    chart_width_percent = 1 / st.session_state["num_cols_strategy_tradings"]

    cols_spec = [chart_width_percent] * st.session_state["num_cols_strategy_tradings"]
    cols = []

    with st.expander("**Click to display/hide model/strategy chart**"):
        for _ in range(num_rows_tradings):
            cols.append(st.columns(cols_spec, gap="small"))

        figs_containers = []
        for row in cols:
            figs_containers.append([item.empty() for item in row])

        for index, fig in enumerate(st.session_state.trading_strategy_figs):
            row_counter = 0
            while index + 1 > st.session_state.num_cols_strategy_tradings:
                index = index - st.session_state.num_cols_strategy_tradings
                row_counter += 1

            figs_containers[row_counter][index].plotly_chart(fig, use_container_width=True)


def draw_strategy_stats():
    with st.expander("**Click to display/hide performance stats**"):
        data = pd.DataFrame(st.session_state["global_strategy_stats"], index=[0])
        st.dataframe(data, use_container_width=True, hide_index=True)


def draw_trading_stats():
    with st.expander("**Click to display/hide trade stats**"):
        trade_display = []

        all_trades_col, short_trades_col, long_trades_col, _ = st.columns([0.12, 0.12, 0.12, 0.64], gap="small")

        all_trades_phold = all_trades_col.empty()
        short_trades_phold = short_trades_col.empty()
        long_trades_phold = long_trades_col.empty()

        if all_trades_phold.checkbox(label="All trades", value=True):
            all_trades = build_trade_stats_data(
                st.session_state["trade_stats_by_symbol"], st.session_state["global_stats_by_symbol"], "All trades"
            )
            trade_display.append(all_trades)
        if short_trades_phold.checkbox(label="Short trades", value=False):
            short_trades = build_trade_stats_data(
                st.session_state["short_trade_stats_by_symbol"], st.session_state["short_global_trade_stats"], "Short trades"
            )
            trade_display.append(short_trades)
        if long_trades_phold.checkbox(label="Long trades", value=False):
            long_trades = build_trade_stats_data(
                st.session_state["long_trade_stats_by_symbol"], st.session_state["long_global_trade_stats"], "Long trades"
            )
            trade_display.append(long_trades)

        if len(trade_display) > 0:
            data = pd.concat(trade_display)
        else:
            data = pd.DataFrame()

        st.dataframe(data, use_container_width=True)


def draw_trading_mode(backtesting_disabled, ind_code_block, trading_code_block):
    actual_currency_col, bet_size_col, tg_limit_col, pig_limit_col, nop_limit_col = st.columns(
        [0.2, 0.2, 0.2, 0.2, 0.2], gap="small"
    )
    actual_currency_phold = actual_currency_col.empty()
    bet_size_phold = bet_size_col.empty()
    tg_limit_phold = tg_limit_col.empty()
    nop_limit_phold = nop_limit_col.empty()
    pig_limit_phold = pig_limit_col.empty()

    if "Split & Dividend adjusted prices" not in st.session_state["datasets_multiselect"]:
        use_dividends_col, fill_price_col, exec_cost_bps_col = st.columns([0.2, 0.4, 0.4], gap="small")
    else:
        fill_price_col, exec_cost_bps_col = st.columns([0.5, 0.5], gap="small")

    fill_price_phold = fill_price_col.empty()
    exec_cost_bps_phold = exec_cost_bps_col.empty()

    if st.session_state.jupyter_redirect_url:
        backtest_col, title_col, save_col, share_col, export_col, open_in_jupyter_col, jupyter_link_col = st.columns(
            [0.2, 0.25, 0.08, 0.08, 0.08, 0.2, 0.11], gap="small"
        )
    else:
        backtest_col, _, title_col, save_col, share_col, export_col, open_in_jupyter_col = st.columns(
            [0.2, 0.11, 0.25, 0.08, 0.08, 0.08, 0.2], gap="small"
        )

    if not backtesting_disabled:
        actual_currency = actual_currency_phold.text_input(
            "Account currency",
            value=st.session_state.actual_currency,
            max_chars=3,
        )
        st.session_state["actual_currency"] = actual_currency

        with bet_size_phold:
            bet_size = st.number_input(
                "Bet size",
                value=st.session_state["bet_size"],
                step=1,
            )
        st.session_state["bet_size"] = bet_size

        per_instrument_gross_limit = pig_limit_phold.number_input(
            "Per instrument gross limit",
            value=st.session_state["per_instrument_gross_limit"],
            step=1,
        )
        st.session_state["per_instrument_gross_limit"] = per_instrument_gross_limit

        total_gross_limit = tg_limit_phold.number_input(
            "Total gross limit",
            value=st.session_state["total_gross_limit"],
            step=1,
        )
        st.session_state["total_gross_limit"] = total_gross_limit

        nop_limit = nop_limit_phold.number_input(
            "NOP limit",
            value=st.session_state["nop_limit"],
            step=1,
        )
        st.session_state["nop_limit"] = nop_limit

        if "Split & Dividend adjusted prices" not in st.session_state["datasets_multiselect"]:
            use_dividends_phold = use_dividends_col.container()
            use_dividends_phold.markdown('<p class="move-font">T</p>', unsafe_allow_html=True)
            backtesting_divs = use_dividends_phold.checkbox(
                label="Account for dividends", value=st.session_state["use_dividends_trading"]
            )
            st.session_state["use_dividends_trading"] = backtesting_divs

        default_index = trade_fill_prices.get_price_index(st.session_state["fill_trade_price"])
        fill_trade_price = fill_price_phold.selectbox(
            label="Trade fill prices", options=trade_fill_prices.get_price_names(), index=default_index
        )
        st.session_state["fill_trade_price"] = trade_fill_prices.get_trade_price_by_name(fill_trade_price).BACKEND_KEY

        exec_cost_bps = exec_cost_bps_phold.number_input(
            "Execution cost (bps)",
            value=st.session_state["execution_cost_bps"],
            step=0.1,
        )
        st.session_state["execution_cost_bps"] = exec_cost_bps

        backtest_phold = backtest_col.container()
        backtest_phold.markdown('<p class="move-font">T</p>', unsafe_allow_html=True)
        try:
            with backtest_phold:
                if st.button(label="Backtest model/strategy", use_container_width=True):
                    backtest_callback()
                    st.session_state.application_flow.add_backtest_into_operation()
                    st.session_state.application_flow.set_prev_trading_code(trading_code_block, ind_code_block)
                    st.session_state.jupyter_redirect_url = ""
                    st.experimental_rerun()
        except (ValueError, DataNotFoundError, LLMBadResponseError) as e:
            st.error(e)

    title_phold = title_col.empty()
    trading_rule_title = title_phold.text_input("Model/Strategy title", value=st.session_state.trading_rule_title)
    st.session_state["trading_rule_title"] = trading_rule_title

    export_phold = export_col.container()
    export_phold.markdown('<p class="move-font">T</p>', unsafe_allow_html=True)
    save_phold = save_col.container()
    save_phold.markdown('<p class="move-font">T</p>', unsafe_allow_html=True)
    share_phold = share_col.container()
    open_in_jupyter_phold = open_in_jupyter_col.container()
    open_in_jupyter_phold.markdown('<p class="move-font">T</p>', unsafe_allow_html=True)

    share_phold.markdown('<p class="move-font">T</p>', unsafe_allow_html=True)

    if share_phold.button("Share", use_container_width=True):
        st.session_state.show_send_form = True

    if st.session_state.show_send_form:
        _, descr_col = st.columns([0.625, 0.375], gap="small")
        _, public_col, recepient_col, send_col = st.columns([0.625, 0.075, 0.2, 0.1], gap="small")

        recepient_phold = recepient_col.empty()
        send_phold = send_col.container()
        public_phold = public_col.container()
        descr_phold = descr_col.empty()

        public_phold.markdown('<p class="move-font">T</p>', unsafe_allow_html=True)
        if st.session_state["is_admin"]:
            public_phold.checkbox(label="Public", value=st.session_state["public_rule"], key="public_rule")

        recepient = recepient_phold.text_input(label="Recepient")

        send_phold.markdown('<p class="move-font">T</p>', unsafe_allow_html=True)
        if send_phold.button(
            label="Send",
            use_container_width=True,
        ):
            trading_rule = None
            if not st.session_state["public_rule"]:
                resp_status_code, resp_content = send_trading_model_callback(
                    st.session_state["keycloak_session"].access_token, recepient
                )
            else:
                trading_rule = alerts_backend_proxy_singleton.get_trading_rule_by_title_from_public(
                    st.session_state.trading_rule_title
                )
                resp_status_code, resp_content = None, None
                if trading_rule is not None:
                    warning_col, submit_col, cancel_col = st.columns([0.8, 0.1, 0.1])
                    warning_col_phold = warning_col.container()
                    submit_phold = submit_col.container()
                    cancel_phold = cancel_col.container()
                    warning_col_phold.warning(
                        f"Public model {st.session_state.trading_rule_title} already exists. Do you want to overwrite it?"
                    )

                    submit_phold.button(
                        "Confirm",
                        use_container_width=True,
                        key="modification_publ_confirm_btn",
                        on_click=publish_trading_rule,
                        args=[st.session_state["keycloak_session"].access_token, trading_rule, True],
                    )

                    cancel_phold.button("Cancel", use_container_width=True)
                else:
                    resp_status_code, resp_content = publish_trading_rule(
                        st.session_state["keycloak_session"].access_token, None, True
                    )

            if resp_status_code == 200:
                if not st.session_state["public_rule"]:
                    st.success(f"Trading model successfully sent to recepient: {recepient}")
                else:
                    if trading_rule is None:
                        st.success(f"Public model successfully published")
                    else:
                        st.success(f"Public model {trading_rule['title']} successfully modified")
            elif resp_status_code is not None:
                st.error(f"An error occured with following message: {resp_content}")

            st.session_state.show_send_form = False

        model_descr = descr_phold.text_area(
            label="Model description",
            height=100,
            value=st.session_state["model_descr"]
            # key="model_descr_area",
        )
        st.session_state["model_descr"] = model_descr

    if not backtesting_disabled:
        if st.session_state.application_flow.check_operation_finish():
            if not st.session_state.application_flow.compare_fetch_to_indicator_run():
                st.warning(
                    "New Data was loaded but indicators was not regenerated. Please consider to regenerate indicators to avoid errors.",
                    icon="⚠️",
                )
            if not st.session_state.application_flow.compare_indicators_run_to_send(ind_code_block):
                st.warning("Indicators code wasn't run. Please rerun indicators code to avoid errors.", icon="⚠️")

        if st.session_state.application_flow.check_first_code_run() and not check_code_block(ind_code_block):
            st.warning("Indicators code wasn't run. Please run indicators code to avoid errors.", icon="⚠️")

        if st.session_state.application_flow.check_backtesting_code(ind_code_block, trading_code_block):
            st.warning("Either indicators code or trading code was changed. Please rerun strategy backtesting.", icon="⚠️")

    st.session_state.application_flow.set_prev_code(ind_code_block)

    session = session_state_getter(st.session_state)
    session_json = json.dumps(session)
    export_phold.download_button(
        "Export",
        data=session_json,
        file_name=f"{st.session_state['trading_rule_title']}_trading_rule.json",
        use_container_width=True,
    )

    if save_phold.button(label="Save", use_container_width=True):
        trading_rule = alerts_backend_proxy_singleton.get_trading_rule_by_title_from_personal(st.session_state.trading_rule_title)

        publish_status_code = None

        if not st.session_state.trading_rule_title:
            warning_col = st.columns([1])[0]
            warning_col_phold = warning_col.container()
            warning_col_phold.warning(f"In order to save the model, please, specify the title.")
        elif trading_rule is not None:
            warning_col, submit_col, cancel_col = st.columns([0.8, 0.1, 0.1])
            warning_col_phold = warning_col.container()
            submit_phold = submit_col.container()
            cancel_phold = cancel_col.container()
            warning_col_phold.warning(f"Model {st.session_state.trading_rule_title} already exists. Do you want to overwrite it?")

            submit_phold.button(
                "Confirm",
                use_container_width=True,
                key="modification_confirm_btn",
                on_click=publish_trading_rule,
                args=[st.session_state["keycloak_session"].access_token, trading_rule],
            )

            cancel_phold.button("Cancel", use_container_width=True)
        else:
            publish_status_code, publish_content = publish_trading_rule(st.session_state["keycloak_session"].access_token, None)

        if publish_status_code is not None:
            if publish_status_code == 200:
                st.success("Trading rule successfully saved")
            else:
                st.error(f"An error occurred. Here is the server response: \n{publish_content}")

    def redirect_to_external_page(url, container):
        redirect_code = f'<br><a href="{url}" target="_blank">Open in Jupyter</a>'
        container.markdown(redirect_code, unsafe_allow_html=True)

    if st.session_state.jupyter_redirect_url:
        redirect_to_external_page(st.session_state.jupyter_redirect_url, jupyter_link_col.empty())

    if open_in_jupyter_phold.button(label="Generate Jupyter Notebook", use_container_width=True):
        jupyter_service = JupyterService.get_instance(JUPYTERHUB_UI_URL, JUPYTERHUB_SERVICE_URL, JUPYTERHUB_TOKEN)
        user_email = st.session_state["keycloak_session"].user_info["email"]

        jupyter_service.create_user(user_email)
        jupyter_service.start_user_server(user_email)
        jupyter_service.wait_until_user_server_ready(user_email, timeout=60)
        st.session_state.jupyter_redirect_url = jupyter_service.put_user_notebook(
            user_email, session_state_getter(st.session_state), st.session_state.trading_rule_title
        )
        st.experimental_rerun()

    if st.session_state.trading_data:
        draw_trading_symbol_charts()

        draw_strategy_chart()

        draw_strategy_stats()

        draw_trading_stats()

    if len(st.session_state["trading_code_log"]) > 0:
        with st.expander("**Click to display/hide backtesting print log**"):
            st.text("\n".join(st.session_state["trading_code_log"]))

    if st.session_state.trading_data:
        draw_download_buttons()


def draw_download_buttons():
    dwnld_pnl_report_col, dwnld_by_instrument_col, dwnld_trades_col, _ = st.columns([0.2, 0.2, 0.2, 0.5])

    dwnld_pnl_report_phold = dwnld_pnl_report_col.empty()
    dwnld_by_instrument_phold = dwnld_by_instrument_col.empty()
    dwnld_trades_phold = dwnld_trades_col.empty()

    pnl_report = create_pnl_report(st.session_state)
    trading_stats_by_symbol = create_stats_by_symbol_report(st.session_state["trading_stats_by_symbol"])
    trades_by_symbol = create_stats_by_symbol_report(st.session_state["trades_by_symbol"])

    dwnld_pnl_report_phold.download_button(
        label="Download pnl report",
        data=pnl_report,
        file_name=f"{st.session_state['trading_rule_title']}_pnl_report.csv"
        if st.session_state["trading_rule_title"]
        else "pnl_report.csv",
        mime="text/csv",
        use_container_width=True,
    )

    dwnld_by_instrument_phold.download_button(
        label="Download trading stats by symbol",
        data=trading_stats_by_symbol,
        file_name=f"{st.session_state['trading_rule_title']}_trading_stats_by_symbol.csv"
        if st.session_state["trading_rule_title"]
        else "trading_stats_by_symbol.csv",
        mime="text/csv",
        use_container_width=True,
    )

    dwnld_trades_phold.download_button(
        label="Download trades",
        data=trades_by_symbol,
        file_name=f"{st.session_state['trading_rule_title']}_trades.csv"
        if st.session_state["trading_rule_title"]
        else "trades.csv",
        mime="text/csv",
        use_container_width=True,
    )


def draw_alert_mode():
    with st.form("alerts_form"):
        with st.container():
            for i, msg in enumerate(st.session_state.alerts_dialogue):
                if i % 2 == 0:
                    prefix = "User : "
                else:
                    prefix = "Assistant :\n"
                st.text(prefix + msg)
                st.markdown("*****************\n")

        alerts_query = st.text_area(
            "Instruct Chat GPT on alert logic",
            value=st.session_state["alerts_query"],
            height=100,
            placeholder="Alerts query",
            # key="alerts_query",
            help="""
    E.g.: Close Price of AAPL / RSI(5) of AAPL Crossing up / Crossing down Value

    Close Price of AAPL / RSI(5) of AAPL  Entering Channel /Exiting Channel: [Lower Band, Upper Band]
    Close Price of AAPL / RSI(5) of AAPL  Moving Up / Moving Down X {cents/bps/%} over past N bars
                    """,
        )
        st.session_state.alerts_query = alerts_query

        if st.session_state.alert_error is not None:
            with st.container():
                st.write("\n")
                st.write("\n")
                st.error(st.session_state.alert_error)
                st.session_state.alert_error = None

        submit_alert_button_col, chart_build_col, clear_alert_history_col, _ = st.columns([0.1, 0.1, 0.1, 0.7], gap="small")
        submit_alert_button_phold = submit_alert_button_col.empty()
        if submit_alert_button_phold.form_submit_button(label="Send", use_container_width=True):
            if st.session_state.llm_chat_disabled:
                st.error("Sorry, but you have exceeded daily token limitation.")
            else:
                alert_submit_callback()
                st.session_state.jupyter_redirect_url = ""
                st.experimental_rerun()

        chart_build_phold = chart_build_col.empty()
        alerts_chat_clear_btn_container = clear_alert_history_col.empty()

    with chart_build_phold:
        if st.button(label="Generate alert", use_container_width=True):
            if len(st.session_state["alerts_dialogue"]) != 0:
                st.session_state.alerts_figs = []
                alert_step(st.session_state)

    if len(st.session_state.alerts_dialogue) > 0:
        with alerts_chat_clear_btn_container:
            st.button("Clear history", key="alerts_chat_clear", on_click=clear_alerts_chat_history, use_container_width=True)

        if st.session_state["roots"]:
            with st.expander("**Click to display/hide alerts charts**"):
                col1, col2, _ = st.columns([0.1, 0.1, 0.8], gap="small")

                reset_layout = col1.empty()
                add_column = col2.empty()

                reset_layout.button(
                    "Reset layout",
                    on_click=alerts_reset_layout_callback,
                    key="reset_alerts_layout",
                )
                add_column.button(
                    "Split layout",
                    on_click=alerts_add_column_callback,
                    key="split_alerts_layout",
                )

                if len(st.session_state.alerts_figs) == 0:
                    st.session_state.alerts_figs = visualize_tree_data_nodes(
                        {
                            **st.session_state["data_by_symbol"],
                            **st.session_state.get("data_by_synth", {}),
                            **st.session_state.get("data_by_indicator", {}),
                        },
                        st.session_state["roots"],
                        st.session_state["main_roots"],
                        "Alerts",
                        st.session_state.interval,
                        st.session_state["trigger_alert"],
                    )

                if st.session_state.num_cols_alerts > len(st.session_state.alerts_figs):
                    st.session_state.num_cols_alerts = len(st.session_state.alerts_figs)

                num_rows_alerts = math.ceil(len(st.session_state.alerts_figs) / st.session_state["num_cols_alerts"])

                chart_width_percent = 1 / st.session_state["num_cols_alerts"]

                cols_spec = [chart_width_percent] * st.session_state["num_cols_alerts"]
                cols = []
                for _ in range(num_rows_alerts):
                    cols.append(st.columns(cols_spec, gap="small"))

                figs_containers = []
                for row in cols:
                    figs_containers.append([item.empty() for item in row])

                for index, fig in enumerate(st.session_state.alerts_figs):
                    row_counter = 0
                    while index + 1 > st.session_state.num_cols_alerts:
                        index = index - st.session_state.num_cols_alerts
                        row_counter += 1

                    figs_containers[row_counter][index].plotly_chart(fig, use_container_width=True)

    time_disabled = "day" not in st.session_state.data_periodicity

    if time_disabled:
        title_col, trigger_type_col = st.columns([0.5, 0.5])
    else:
        title_col, trigger_type_col, timezone_col, trigger_time_col = st.columns([0.25, 0.25, 0.25, 0.25])

    trigger_type_phold = trigger_type_col.empty()
    title_phold = title_col.empty()

    if st.session_state.available_timezones is None:
        st.session_state.available_timezones = alerts_backend_proxy_singleton.get_available_timezones()

    title_phold.text_input("Alert title", key="alert_title", value=st.session_state.alert_title)
    trigger_type_phold.selectbox(
        "Alert trigger type", list(TRIGGER_TYPES.keys()), key="trigger_type", index=st.session_state.trigger_type_index
    )

    if not time_disabled:
        trigger_time_phold = trigger_time_col.empty()
        timezone_phold = timezone_col.empty()
        if st.session_state.available_timezones is None:
            st.session_state.available_timezones = alerts_backend_proxy_singleton.get_available_timezones()

        if st.session_state.timezone_index == 0:
            for index, zone in enumerate(st.session_state.available_timezones):
                if "New_York" in zone:
                    break
            st.session_state.timezone_index = index

        timezone_phold.selectbox(
            "Timezone", st.session_state.available_timezones, key="timezone", index=st.session_state.timezone_index
        )

        trigger_time_phold.text_input("Alert trigger time", key="trigger_time", value="00:00:00")

        if not re.match(r"\d{2}:\d{2}:\d{2}", st.session_state["trigger_time"]):
            st.write("\n")
            st.write("\n")
            st.error(f"Time {st.session_state.trigger_time} is not valid, please specify time in following format: '00:00:00'")

    st.text_area(
        "Please specify alert message",
        key="alert_message",
        help=f"You can output additional info by using {{ }} quotes.\n\n Possible values: {'; '.join(chain(st.session_state.data_by_symbol.keys(), st.session_state.data_by_synth.keys(), st.session_state.data_by_indicator.keys()))}",
    )
    receiver_col, publish_col = st.columns([0.9, 0.1])
    receiver_phold = receiver_col.empty()
    publish_phold = publish_col.container()

    if st.session_state.alert_email_receivers == "":
        st.session_state.alert_email_receivers = st.session_state["keycloak_session"].user_info["email"]

    receiver_phold.text_input(
        "Emails to notify",
        key="receiver_email",
        value=st.session_state.alert_email_receivers,
    )

    publish_phold.markdown('<p class="move-font">T</p>', unsafe_allow_html=True)

    modifying_alert = st.session_state["alert_edit"] is not None

    if modifying_alert:
        button_title = "Modify alert"
    else:
        button_title = "Publish alert"

    if publish_phold.button(button_title, use_container_width=True):
        publish_status_code, publish_content = publish_alerts(
            st.session_state["keycloak_session"].access_token, st.session_state["alert_edit"]
        )

        if publish_status_code == 200:
            if modifying_alert:
                st.success("Alert successfully modified")
            else:
                st.success("Alert successfully published")
        else:
            st.error(f"An error occurred. Here is the server response: \n{publish_content}")

    st.write("")


def main_tab_component():
    roles = alerts_backend_proxy_singleton.get_user_roles()

    if "admin" in roles:
        st.session_state.is_admin = True

    col1, col2 = st.columns([0.5, 0.5], gap="small")

    data_provider_phold = col1.empty()
    datasets_phold = col2.empty()

    # Dropdown for selecting data provider
    index_prov = 0
    if st.session_state.get("provider", "") != "":
        for index, prov_name in enumerate(DATA_PROVIDERS_NAMES_TO_BACKEND_KEY.keys()):
            if prov_name == st.session_state["provider"]:
                index_prov = index

    data_provider = data_provider_phold.selectbox(
        "Select Data Provider", DATA_PROVIDERS_NAMES_TO_BACKEND_KEY, key="provider", index=index_prov
    )
    selected_provider = DATA_PROVIDERS_NAMES_TO_BACKEND_KEY[data_provider]
    st.session_state.data_provider = selected_provider
    provider = data_providers[st.session_state["data_provider"]]

    available_datasets = [i.DATASET_NAME for i in datasets.values()]
    preselect_datasets = [i.DATASET_NAME for i in datasets.values() if i.IS_DEFAULT]
    datasets_selected = datasets_phold.multiselect(
        label="Datasets", options=available_datasets, default=preselect_datasets, key="datasets_multiselect"
    )
    datasets_selected_keys = [
        dataset_key for dataset_key, dataset in datasets.items() if dataset.DATASET_NAME in datasets_selected
    ]
    st.session_state["datasets_keys"] = datasets_selected_keys

    if check_prices_multisellect_result():
        st.experimental_rerun()

    col1, col2, col3 = st.columns([0.5, 0.4, 0.1], gap="small")

    periodicity_phold = col1.empty()
    lookup_search_phold = col2.empty()
    lookup_button_phold = col3.container()

    # Dropdown for selecting periodicity
    periodicity = periodicity_phold.selectbox(
        "Select Periodicity",
        DATA_PERIODICITIES_NAMES_TO_BACKEND_KEY,
        key="data_periodicity",
        index=st.session_state.data_periodicity_index,
    )
    selected_periodicity = DATA_PERIODICITIES_NAMES_TO_BACKEND_KEY[periodicity]
    st.session_state.interval = selected_periodicity

    lookup = lookup_search_phold.text_input("Tickers lookup", help="Test")

    lookup_button_phold.markdown('<p class="move-font">T</p>', unsafe_allow_html=True)
    lookup_button_pressed = lookup_button_phold.button("Search", use_container_width=True, key="lookup_button")

    st.session_state.tickers_expand = lookup_button_pressed or any(st.session_state.add_ticker_buttons)

    if lookup_button_pressed or len(st.session_state.symbols_lookup) > 0:
        if lookup != "" and lookup_button_pressed:
            st.session_state.symbols_lookup = provider.lookup_tickers(tickers_query=lookup)
        with st.expander(
            "**Click to display/hide found tickers**",
            expanded=st.session_state.tickers_expand,
        ):
            grid_keys = {
                "symbol": "Symbol",
                "instrument_name": "Instrument name",
                "exchange": "Exchange",
                "mic_code": "Mic code",
                "exchange_timezone": "Exchange timezone",
                "instrument_type": "Instrument type",
                "country": "Country",
                "currency": "Currency",
            }
            column_width = [0.1, 0.3, 0.05, 0.05, 0.2, 0.1, 0.1, 0.05, 0.05]
            columns = st.columns(column_width, gap="small")
            for index, key in enumerate(grid_keys.values()):
                phold = columns[index].container()
                phold.write(key)

            st.session_state.add_ticker_buttons = []

            for symbol in st.session_state.symbols_lookup:
                columns = st.columns(column_width, gap="small")
                for index, item in enumerate(list(grid_keys.keys())):
                    phold = columns[index].container()
                    phold.write(symbol[item])

                phold = columns[index + 1].container()
                phold.button(
                    "Add",
                    key="add_symbol" + symbol["symbol"] + ":" + symbol["exchange"] + ":" + symbol["mic_code"],
                    use_container_width=True,
                    on_click=add_symbol_button_callback,
                    args=[symbol["symbol"], symbol["exchange"]],
                )

    tickers = st.text_area(
        label="Enter comma-separated list of tickers or pricing formula",
        placeholder="Tickers",
        value=st.session_state["tradable_symbols_prompt"],
        height=100,
        help='Use comma separated list of instruments or pricing formulas: AAPL, GOOG, MSFT, SPY, QQQ.\n Extend ticker definition with exchange name to choose correct stock/index/etf, e.g.: AAPL\:LSE (the company with this ticker on London Stock Excchange is not Apple Inc)\n\nFormat of currencies for PolygonIO data provider\:\nCryptocurrencies should have the “X\:” prefix and currency pair without any special symbols.\nExample\: “X\:BTCUSD”\nFx currencies should have the “C\:” prefix and currency pair without any special symbols.\nExample\: “C\:EURUSD”\nStock exchange could not be specified for PolygonIO data provider.\n\nFormat of stock exchange for TwelveData data provider\:\nYou can specify only stock like that “AAPL”, but if you want to specify stock exchange you should use “\:” symbol.\nExample: “AAPL\:NASDAQ” or “AAPL\:LSE”\nFormat of currencies for TwelveData data provider\:\nCurrency pair fx or crypto can be specified like currency pair separated by “/” symbol.\nExample\: “EUR/USD” or “BTC/USD”\n\nFor pricing formulas of synthetic instruments use arithemtic expressions that include tickers, basic arithmetic operators +, -, *, / and numeric constants:\nExamples:\nQQQ - SPY (this is a spread between price of one share of QQQ and one share of SPY)\nAAPL + MSFT - 2 * GOOG (a portfolio containing one share of Apple, one share of Microsoft and minus 2 shares of Google (aka short position))\nAAPL / MSFT (ratio of the price of one share of Apple to one share of Microsoft)\n\nAssign an arbitrary name to the pricing formula of synthetic instrument:\nAAPL_TO_MSFT = AAPL / MSFT\n\nFor the instruments having numerical ticker or instruments containing various delimiters and special symbols enclose ticker into single of double quotes:\n"EUR/USD", "GBP/USD", EURGBP = "ERU/USD" / "GBP/USD" (Euro and Pound FX rates to US dollar, and Cross Rate EUR to GBP defined via sythhetic)\n"BTC/USD" (rate of Bitcoin to US Dollar)',
    )
    st.session_state["tradable_symbols_prompt"] = tickers

    if st.session_state["fetch_ticker_error"]:
        with st.container():
            st.error("Seems like some operator is missing in tickers field, please check your spelling")

        st.session_state["fetch_ticker_error"] = False

    if st.session_state.fetch_data_range_index is None:
        st.session_state.fetch_data_range_index = list(DATA_TIME_RANGES_NAMES_TO_BACKEND_KEY.keys()).index("10 years")

    time_period = st.selectbox(
        "Please specify time range for data fetch",
        [key for key in DATA_TIME_RANGES_NAMES_TO_BACKEND_KEY.keys()],
        key="time_period_",
        index=st.session_state.fetch_data_range_index,
    )
    selected_time_period = DATA_TIME_RANGES_NAMES_TO_BACKEND_KEY[time_period]
    st.session_state.time_period = selected_time_period

    fetch_col, _, import_col = st.columns([0.1, 0.75, 0.15], gap="small")
    fetch_phold = fetch_col.empty()
    import_phold = import_col.empty()

    import_pressed = import_phold.button("Import model", use_container_width=True)
    if not st.session_state["show_import"]:
        st.session_state["show_import"] = import_pressed

    if st.session_state["show_import"]:
        upload_col, cancel_col = st.columns([0.85, 0.15], gap="small")
        upload_phold = upload_col.empty()
        cancel_phold = cancel_col.empty()
        if cancel_phold.button("Cancel", use_container_width=True):
            st.session_state["show_import"] = False
            st.experimental_rerun()

        uploaded_file = upload_phold.file_uploader(
            "Import model", accept_multiple_files=False, type=["json"], label_visibility="collapsed"
        )

        if uploaded_file is not None:
            file_contents = uploaded_file.getvalue().decode("utf-8")
            imported_session = json.loads(file_contents)
            trading_rule_import_callback(imported_session)
            st.experimental_rerun()

            # st.markdown('''
            #     <style>
            #         .uploadedFile {display: none}
            #     <style>''',
            #     unsafe_allow_html=True)

    if fetch_phold.button("Fetch data", use_container_width=True):
        try:
            fetch_data()
            st.session_state.application_flow.add_fetch_into_operation()
            st.experimental_rerun()
        except (ValueError, DataNotFoundError) as e:
            st.session_state.fetched_data = False
            with st.container():
                st.error(e)

    if not st.session_state["dividends_currency_match"]:
        st.warning(
            "Fetched prices currency and dividends currency don't match.",
            icon="⚠️",
        )

    if st.session_state.get("symbols", None) is not None:
        if "bal" in st.session_state["datasets_keys"]:
            with st.expander("**Click to display/hide balance sheets**"):
                if st.session_state.balance_sheets is None or st.session_state.balance_currencies is None:
                    (
                        st.session_state.balance_sheets,
                        st.session_state.balance_currencies,
                    ) = datasets["bal"].populate_data(
                        None, None, provider, st.session_state["symbols"], st.session_state["true_symbols"], None, None, None
                    )

                tab_names = list(st.session_state.balance_sheets.keys())
                if tab_names:
                    balance_tabs = st.tabs(tab_names)

                for balance_tab_index, balance_tab_name in enumerate(tab_names):
                    with balance_tabs[balance_tab_index]:
                        gridOptions = build_aggrid_options(
                            st.session_state.balance_sheets[balance_tab_name],
                            st.session_state.balance_currencies[balance_tab_name],
                        )

                        AgGrid(
                            pd.DataFrame(st.session_state.balance_sheets[balance_tab_name]),
                            gridOptions=gridOptions,
                            allow_unsafe_jscode=True,
                            enable_enterprise_modules=True,
                            filter=True,
                            update_mode=GridUpdateMode.SELECTION_CHANGED,
                            theme="material",
                            tree_data=True,
                            columns_auto_size_mode=ColumnsAutoSizeMode.FIT_ALL_COLUMNS_TO_VIEW,
                            key="AgGrid_balance_" + balance_tab_name,
                        )

        if "inc" in st.session_state["datasets_keys"]:
            with st.expander("**Click to display/hide income statement**"):
                if st.session_state.income_statements is None or st.session_state.income_currencies is None:
                    (
                        st.session_state.income_statements,
                        st.session_state.income_currencies,
                    ) = datasets["inc"].populate_data(
                        None, None, provider, st.session_state["symbols"], st.session_state["true_symbols"], None, None, None
                    )
                tab_names = list(st.session_state.income_statements.keys())
                if tab_names:
                    income_tabs = st.tabs(tab_names)

                for income_tab_index, income_tab_name in enumerate(tab_names):
                    with income_tabs[income_tab_index]:
                        gridOptions = build_aggrid_options(
                            st.session_state.income_statements[income_tab_name],
                            st.session_state.income_currencies[income_tab_name],
                        )

                        AgGrid(
                            pd.DataFrame(st.session_state.income_statements[income_tab_name]),
                            gridOptions=gridOptions,
                            allow_unsafe_jscode=True,
                            enable_enterprise_modules=True,
                            filter=True,
                            update_mode=GridUpdateMode.SELECTION_CHANGED,
                            theme="material",
                            tree_data=True,
                            columns_auto_size_mode=ColumnsAutoSizeMode.FIT_ALL_COLUMNS_TO_VIEW,
                            key="AgGrid_income_" + income_tab_name,
                        )

    if st.session_state.get("fetched_data", False):
        with st.expander("**Click to display/hide price charts**"):
            col1, col2, _ = st.columns([0.1, 0.1, 0.8], gap="small")

            reset_layout = col1.empty()
            add_column = col2.empty()

            reset_layout.button(
                "Reset layout",
                on_click=price_reset_layout_callback,
                key="reset_price_layout",
            )
            add_column.button(
                "Split layout",
                on_click=price_add_column_callback,
                key="split_price_layout",
            )

            if len(st.session_state.price_figs) == 0:
                st.session_state.price_figs = visualize_data_nodes(
                    {
                        **st.session_state["data_by_symbol"],
                        **st.session_state.get("data_by_synth", {}),
                    },
                    st.session_state["num_cols_price"],
                    st.session_state.interval,
                )

            if st.session_state.num_cols_price > len(st.session_state.price_figs):
                st.session_state.num_cols_price = len(st.session_state.price_figs)

            num_rows_price = math.ceil(
                len(st.session_state.price_figs)
                / (st.session_state["num_cols_price"] if st.session_state["num_cols_price"] != 0 else 1)
            )

            chart_width_percent = 1 / (st.session_state["num_cols_price"] if st.session_state["num_cols_price"] != 0 else 1)

            cols_spec = [chart_width_percent]
            if st.session_state["num_cols_price"] > 0:
                cols_spec = cols_spec * st.session_state["num_cols_price"]

            cols = []
            for _ in range(num_rows_price):
                cols.append(st.columns(cols_spec, gap="small"))

            figs_containers = []
            for row in cols:
                figs_containers.append([item.empty() for item in row])

            for index, fig in enumerate(st.session_state.price_figs):
                row_counter = 0
                while index + 1 > st.session_state.num_cols_price:
                    index = index - st.session_state.num_cols_price
                    row_counter += 1

                figs_containers[row_counter][index].plotly_chart(fig, use_container_width=True)

    if st.session_state.get("fetched_data", False):
        if len(st.session_state.indicators_dialogue) > 0:
            with st.expander("**Click to display/hide chat history**", expanded=True):
                with st.container():
                    for i, msg in enumerate(st.session_state.indicators_dialogue):
                        if i % 2 == 0:
                            prefix = "User : "
                            st.text(prefix + msg)
                        else:
                            prefix = "Assistant :\n"
                            st.text(prefix)
                            st.code(body=msg, language="python", line_numbers=False)

            if st.session_state["show_indicators_code_editor"]:
                try:
                    code_sections = get_code_sections(st.session_state["indicators_dialogue"][-1])
                    if len(code_sections) < 2:
                        indicator_code = ""
                    else:
                        indicator_code = code_sections[0]
                except LLMBadResponseError as e:
                    with st.container():
                        st.error(repr(e))
                    indicator_code = ""

                indicator_line_number = indicator_code.count("\n") + 1
                indicator_code_editor_height = indicator_line_number * 20 + 50
                indicator_code_editor_height = min(CODE_EDITOR_MAX_HEIGHT, indicator_code_editor_height)
                modified_indicator_code = st_monaco(
                    value=indicator_code,
                    language="python",
                    height=str(indicator_code_editor_height) + "px",
                    lineNumbers=False,
                    theme="vs",
                )
                save_indicator_code_col, cancel_editing_indicator_col, _ = st.columns([0.1, 0.1, 0.8], gap="small")
                save_indicator_code_phold = save_indicator_code_col.empty()
                cancel_editing_indicator_phold = cancel_editing_indicator_col.empty()
                if save_indicator_code_phold.button("Save", key="indicators_code_save", use_container_width=True):
                    st.session_state["indicators_dialogue"][-1] = st.session_state["indicators_dialogue"][-1].replace(
                        indicator_code, modified_indicator_code
                    )
                    st.session_state["show_indicators_code_editor"] = False
                    st.experimental_rerun()
                if cancel_editing_indicator_phold.button("Cancel", key="cancel_editing_indicators", use_container_width=True):
                    st.session_state["show_indicators_code_editor"] = False
                    st.experimental_rerun()

            if st.session_state["show_trading_code_editor"]:
                try:
                    code_sections = get_code_sections(st.session_state["indicators_dialogue"][-1])
                    if len(code_sections) < 2:
                        trading_code = ""
                    else:
                        trading_code = code_sections[1]
                except LLMBadResponseError as e:
                    with st.container():
                        st.error(repr(e))
                    trading_code = ""

                trading_line_number = trading_code.count("\n") + 1
                trading_code_editor_height = trading_line_number * 20 + 50
                trading_code_editor_height = min(CODE_EDITOR_MAX_HEIGHT, trading_code_editor_height)
                modified_trading_code = st_monaco(
                    value=trading_code, language="python", height=str(trading_code_editor_height) + "px", lineNumbers=False
                )
                save_trading_code_col, cancel_editing_trading_col, _ = st.columns([0.1, 0.1, 0.8], gap="small")
                save_trading_code_phold = save_trading_code_col.empty()
                cancel_editing_trading_phold = cancel_editing_trading_col.empty()
                if save_trading_code_phold.button("Save", key="trading_code_save", use_container_width=True):
                    st.session_state["indicators_dialogue"][-1] = st.session_state["indicators_dialogue"][-1].replace(
                        trading_code, modified_trading_code
                    )
                    st.session_state["show_trading_code_editor"] = False
                    st.experimental_rerun()
                if cancel_editing_trading_phold.button("Cancel", key="cancel_editing_trading", use_container_width=True):
                    st.session_state["show_trading_code_editor"] = False
                    st.experimental_rerun()

        with st.form("indicators_form"):
            descr_col, _, model_name_col = st.columns([0.5, 0.3, 0.2], gap="small")

            for idx, model_name in enumerate(get_available_models()):
                if model_name == DEFAULT_MODEL_NAME:
                    break

            descr_phold = descr_col.container()
            descr_phold.write(
                "Design a model/trading strategy with assistance of the selected LLM:<br>[documentation/samples](https://trading-service-documentation.staging.deltixhub.io/developer-resources)",
                unsafe_allow_html=True,
            )

            model_name_phold = model_name_col.empty()
            model_name = model_name_phold.selectbox(
                "Select model", options=get_available_models(), index=idx, label_visibility="collapsed"
            )
            st.session_state.engine = get_available_models()[model_name]
            indicators_query = st.text_area(
                "Design a model/trading strategy with assistance of the selected LLM:",
                label_visibility="collapsed",
                value=st.session_state["indicators_query"],
                height=100,
            )
            st.session_state.indicators_query = indicators_query

            send_message_col, clear_history_col, _, submit_button_col, _, edit_indicators_col, edit_trading_col = st.columns(
                [0.14, 0.14, 0.15, 0.14, 0.15, 0.14, 0.14], gap="small"
            )
            send_message_phold = send_message_col.empty()
            if send_message_phold.form_submit_button(label="Send prompt", use_container_width=True):
                if st.session_state.llm_chat_disabled:
                    st.error("Sorry, but you have exceeded daily token limitation.")
                else:
                    indicator_submit_callback()
                    st.session_state.indicator_error = None
                    st.session_state.jupyter_redirect_url = ""
                    st.experimental_rerun()

            submit_button_phold = submit_button_col.empty()
            indicators_chat_clear_btn_container = clear_history_col.empty()
            edit_indicators_phold = edit_indicators_col.empty()
            edit_trading_phold = edit_trading_col.empty()

        indicators_dialogue_exists = len(st.session_state.indicators_dialogue) > 0
        contains = [True, True]
        ind_code_block = ""
        trading_code_block = ""
        if len(st.session_state.indicators_dialogue) > 0:
            try:
                code_sections = get_code_sections(st.session_state["indicators_dialogue"][-1])
                if len(code_sections) >= 2:
                    ind_code_block = code_sections[0]
                    trading_code_block = code_sections[1]
                    contains = [check_code_block(section) for section in code_sections]
                else:
                    contains = [False, False]
            except LLMBadResponseError as e:
                with st.container():
                    st.error(repr(e))
                contains = [False, False]

            try:
                with submit_button_phold:
                    if st.button(label="Run indicator code", use_container_width=True, disabled=contains[0]):
                        if indicators_dialogue_exists:
                            st.session_state.indicator_error = None
                            st.session_state.indicators_figs = []

                            _, _, execution_time = func_profile(indicator_step, st.session_state)

                            execution_time = math.ceil(execution_time)

                            st.session_state["log"].append(f"Indicator code execution time: {execution_time} seconds")

                            st.session_state.application_flow.clear_backtesting_run()
                            st.session_state.application_flow.add_indicators_run_into_operation()
                            st.session_state.application_flow.set_prev_code(ind_code_block)
                            st.experimental_rerun()

            except Exception as e:
                st.error(f"An error occured during running indicator code: {e}")
            finally:
                if len(st.session_state.indicators_dialogue) > 0:
                    with indicators_chat_clear_btn_container:
                        st.button(
                            "Clear prompt history",
                            key="indicators_chat_clear",
                            on_click=clear_indicators_chat_history,
                        )

            with edit_indicators_phold:
                if st.button(label="Edit indicators code", use_container_width=True):
                    st.session_state["show_indicators_code_editor"] = True
                    st.experimental_rerun()

            with edit_trading_phold:
                if st.button(label="Edit trading code", use_container_width=True):
                    st.session_state["show_trading_code_editor"] = True
                    st.experimental_rerun()

            if st.session_state.indicator_error is not None:
                with st.container():
                    st.error(st.session_state.indicator_error)

            if st.session_state["roots"]:
                with st.expander("**Click to display/hide indicators charts**"):
                    col1, col2, _ = st.columns([0.1, 0.1, 0.8], gap="small")

                    reset_layout = col1.empty()
                    add_column = col2.empty()

                    reset_layout.button(
                        "Reset layout",
                        on_click=indicators_reset_layout_callback,
                        key="reset_indicators_layout",
                    )
                    add_column.button(
                        "Split layout",
                        on_click=indicators_add_column_callback,
                        key="split_indicators_layout",
                    )

                    if len(st.session_state.indicators_figs) == 0:
                        st.session_state.indicators_figs = visualize_tree_data_nodes(
                            {
                                **st.session_state["data_by_symbol"],
                                **st.session_state.get("data_by_synth", {}),
                                **st.session_state.get("data_by_indicator", {}),
                            },
                            st.session_state["roots"],
                            st.session_state["main_roots"],
                            "Indicators",
                            st.session_state["interval"],
                        )

                    if st.session_state["num_cols_indicators"] > len(st.session_state.indicators_figs):
                        st.session_state["num_cols_indicators"] = len(st.session_state.indicators_figs)

                    num_rows_indicators = math.ceil(
                        len(st.session_state["indicators_figs"]) / st.session_state["num_cols_indicators"]
                    )

                    chart_width_percent = 1 / st.session_state["num_cols_indicators"]

                    cols_spec = [chart_width_percent] * st.session_state["num_cols_indicators"]
                    cols = []
                    for _ in range(num_rows_indicators):
                        cols.append(st.columns(cols_spec, gap="small"))

                    figs_containers = []
                    for row in cols:
                        figs_containers.append([item.empty() for item in row])

                    for index, fig in enumerate(st.session_state["indicators_figs"]):
                        row_counter = 0
                        while index + 1 > st.session_state["num_cols_indicators"]:
                            index = index - st.session_state["num_cols_indicators"]
                            row_counter += 1

                        figs_containers[row_counter][index].plotly_chart(fig, use_container_width=True)

            if len(st.session_state["indicators_code_log"]) > 0:
                with st.expander("**Click to display/hide indicators print log**"):
                    st.text("\n".join(st.session_state["indicators_code_log"]))

        draw_trading_mode(contains[1], ind_code_block, trading_code_block)
