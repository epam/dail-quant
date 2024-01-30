import json
import math
import re
import uuid
from datetime import datetime
from typing import Any, Callable, Dict, Tuple

import pandas as pd
import requests
import streamlit as st
from st_aggrid import GridOptionsBuilder, JsCode

from market_alerts.config import (
    ALERTS_BACKEND_SERVICE_URL,
    JUPYTERHUB_UI_URL,
    KEYCLOAK_LOGOUT_REDIRECT_URI,
)
from market_alerts.containers import (
    alerts_backend_proxy_singleton,
    data_providers,
    datasets,
    trade_fill_prices,
)
from market_alerts.domain.constants import (
    APP_MODES,
    DATA_PERIODICITIES_NAMES_TO_BACKEND_KEY,
    DATA_PERIODICITIES_NAMES_TO_VALUES,
    DATA_TIME_RANGES_NAMES_TO_BACKEND_KEY,
    DATA_TIME_RANGES_NAMES_TO_VALUES,
    EMAIL_REGEXP,
    PROMPTS_SEPARATOR,
    TRIGGER_TYPES,
)
from market_alerts.domain.services import (
    alert_chat,
    alert_step,
    define_empty_indicators_step,
    define_useful_strings,
    indicator_chat,
    indicator_step,
    symbol_step,
    trading_step,
)
from market_alerts.domain.services.charts import read_chart_layout, write_chart_layout
from market_alerts.domain.services.code import get_code_sections
from market_alerts.entrypoints.streamlit_frontend.constants import (
    DATA_PERIODICITIES_VALUES_TO_BACKEND_KEY,
    DATA_PROVIDERS_NAMES_TO_BACKEND_KEY,
    DATASETS_OPTIONS,
    DEFAULT_PAGE_SIZE,
    PAGE_SIZE_OPTIONS,
    PAGINATION_MIN_PAGE_SIZE,
)
from market_alerts.openai_utils import get_available_models
from market_alerts.utils import (
    func_profile,
    func_profile_gen,
    ms_to_string,
    string_to_ms,
)


class AppFlow:
    def __init__(self):
        self._data_fetch = ""
        self._indicator_run = ""
        self._backtest_run = ""

        self._prev_ind_code_run = ""
        self._prev_trading_code = ""
        self._prev_ind_code_trading = ""

    def add_fetch_into_operation(self):
        self._data_fetch = uuid.uuid4()

    def add_indicators_run_into_operation(self):
        self._indicator_run = self._data_fetch

    def add_backtest_into_operation(self):
        self._backtest_run = self._indicator_run

    def compare_fetch_to_indicator_run(self) -> bool:
        return self._data_fetch == self._indicator_run

    def compare_indicators_run_to_send(self, curr_ind_code) -> bool:
        if self._prev_ind_code_run == "":
            return True

        return curr_ind_code == self._prev_ind_code_run

    def clear_run(self):
        if self._indicator_run != "":
            self._indicator_run = "0"

    def clear_backtesting_run(self):
        if self._backtest_run != "":
            self._backtest_run = "0"

    def check_first_code_run(self) -> bool:
        return self._data_fetch != "" and self._indicator_run == ""

    def set_prev_code(self, prev_code):
        self._prev_ind_code_run = prev_code

    def set_prev_trading_code(self, prev_code_trading, prev_code_ind):
        self._prev_trading_code = prev_code_trading
        self._prev_ind_code_trading = prev_code_ind

    def check_operation_finish(self) -> bool:
        return self._data_fetch != "" and self._indicator_run != ""

    def check_backtesting_code(self, ind_code, trading_code):
        if self._backtest_run == "":
            return False

        if self._prev_trading_code != trading_code:
            return True

        if self._indicator_run != "" and self._backtest_run != self._indicator_run:
            return self._prev_ind_code_trading != ind_code
        else:
            return False


def set_default_state():
    if "log" not in st.session_state:
        st.session_state.log = []
    if "tradable_symbols" not in st.session_state:
        st.session_state.tradable_symbols = []
    if "tradable_symbols_prompt" not in st.session_state:
        st.session_state.tradable_symbols_prompt = ""
    if "fetched_data" not in st.session_state:
        st.session_state.fetched_data = False
    if "trading_data" not in st.session_state:
        st.session_state.trading_data = False
    if "indicators_query" not in st.session_state:
        st.session_state.indicators_query = ""
    if "indicators_dialogue" not in st.session_state:
        st.session_state.indicators_dialogue = []
    if "alerts_dialogue" not in st.session_state:
        st.session_state.alerts_dialogue = []
    if "trading_dialogue" not in st.session_state:
        st.session_state.trading_dialogue = []
    if "alerts_query" not in st.session_state:
        st.session_state.alerts_query = ""
    if "trading_query" not in st.session_state:
        st.session_state.trading_query = ""
    if "roots" not in st.session_state:
        st.session_state.roots = {}
    if "trigger_alert" not in st.session_state:
        st.session_state.trigger_alert = None

    if "start_date" not in st.session_state:
        st.session_state.start_date = "2022-01-01 00:00:00"
    if "end_date" not in st.session_state:
        st.session_state.end_date = "2022-02-01 00:00:00"

    if "num_cols_price" not in st.session_state:
        st.session_state.num_cols_price = 3
    if "price_figs" not in st.session_state:
        st.session_state.price_figs = []
    if "num_cols_indicators" not in st.session_state:
        st.session_state.num_cols_indicators = 3
    if "indicators_figs" not in st.session_state:
        st.session_state.indicators_figs = []
    if "num_cols_alerts" not in st.session_state:
        st.session_state.num_cols_alerts = 3
    if "num_cols_tradings" not in st.session_state:
        st.session_state.num_cols_tradings = 3
    if "num_cols_strategy_tradings" not in st.session_state:
        st.session_state.num_cols_strategy_tradings = 3
    if "alerts_figs" not in st.session_state:
        st.session_state.alerts_figs = []
    if "trading_symbols_figs" not in st.session_state:
        st.session_state.trading_symbols_figs = []
    if "trading_strategy_figs" not in st.session_state:
        st.session_state.trading_strategy_figs = []

    if "symbols_lookup" not in st.session_state:
        st.session_state.symbols_lookup = []
    # if "tickers" not in st.session_state:
    #     st.session_state.tickers = ""
    if "tickers_expand" not in st.session_state:
        st.session_state.tickers_expand = False
    if "datasets_expand" not in st.session_state:
        st.session_state.datasets_expand = False
    if "add_ticker_buttons" not in st.session_state:
        st.session_state.add_ticker_buttons = [False]
    if "keyclock_session" not in st.session_state:
        st.session_state.keycloak_session = None
    if "logging_out" not in st.session_state:
        st.session_state.logging_out = False
    if "added_datasets" not in st.session_state:
        st.session_state.added_datasets = False
    if "logged_out" not in st.session_state:
        st.session_state.logged_out = False
    if "available_timezones" not in st.session_state:
        st.session_state.available_timezones = None
    if "indicator_error" not in st.session_state:
        st.session_state.indicator_error = None
    if "alert_error" not in st.session_state:
        st.session_state.alert_error = None
    if "events" not in st.session_state:
        st.session_state.events = []
    if "trading_rules" not in st.session_state:
        st.session_state.trading_rules = []
    if "trading_page_num" not in st.session_state:
        st.session_state.trading_page_num = 1
    if "trading_page_size" not in st.session_state:
        st.session_state.trading_page_size = DEFAULT_PAGE_SIZE
    if "public_page_num" not in st.session_state:
        st.session_state.public_page_num = 1
    if "public_page_size" not in st.session_state:
        st.session_state.public_page_size = DEFAULT_PAGE_SIZE
    if "alert_page_num" not in st.session_state:
        st.session_state.alert_page_num = 1
    if "alert_page_size" not in st.session_state:
        st.session_state.alert_page_size = DEFAULT_PAGE_SIZE
    if "event_page_num" not in st.session_state:
        st.session_state.event_page_num = 1
    if "event_page_size" not in st.session_state:
        st.session_state.event_page_size = DEFAULT_PAGE_SIZE

    if "app_mode" not in st.session_state:
        st.session_state.app_mode = APP_MODES[0]

    if "alert_edit" not in st.session_state:
        st.session_state.alert_edit = None
    if "data_periodicity_index" not in st.session_state:
        st.session_state.data_periodicity_index = 0
    if "fetch_data_range_index" not in st.session_state:
        st.session_state.fetch_data_range_index = None
    if "trigger_type_index" not in st.session_state:
        st.session_state.trigger_type_index = 0
    if "timezone_index" not in st.session_state:
        st.session_state.timezone_index = 0
    if "alert_email_receivers" not in st.session_state:
        st.session_state.alert_email_receivers = ""
    if "alert_title" not in st.session_state:
        st.session_state.alert_title = ""

    if "llm_chat_disabled" not in st.session_state:
        st.session_state.llm_chat_disabled = False
    if "income_statements" not in st.session_state:
        st.session_state.income_statements = None
    if "balance_sheets" not in st.session_state:
        st.session_state.balance_sheets = None
    if "income_currencies" not in st.session_state:
        st.session_state.income_currencies = None
    if "balance_currencies" not in st.session_state:
        st.session_state.balance_currencies = None

    if "actual_currency" not in st.session_state:
        st.session_state.actual_currency = ""
    if "bet_size" not in st.session_state:
        st.session_state.bet_size = 0
    if "per_instrument_gross_limit" not in st.session_state:
        st.session_state.per_instrument_gross_limit = 0
    if "total_gross_limit" not in st.session_state:
        st.session_state.total_gross_limit = 0
    if "nop_limit" not in st.session_state:
        st.session_state.nop_limit = 0

    if "backtest_mem_usage" not in st.session_state:
        st.session_state.backtest_mem_usage = 0
    if "backtest_exec_time" not in st.session_state:
        st.session_state.backtest_exec_time = 0
    if "symbol_to_currency" not in st.session_state:
        st.session_state.symbol_to_currency = {}
    if "mode_index" not in st.session_state:
        st.session_state.mode_index = 0
    if "fetch_ticker_error" not in st.session_state:
        st.session_state.fetch_ticker_error = False

    if "jupyter_redirect_url" not in st.session_state:
        st.session_state.jupyter_redirect_url = ""
    if "trading_rule_title" not in st.session_state:
        st.session_state.trading_rule_title = ""
    if "lclsglbls" not in st.session_state:
        st.session_state.lclsglbls = {}
    if "time_range" not in st.session_state:
        st.session_state.time_range = ""
    if "engine" not in st.session_state:
        st.session_state.engine = "gpt-4"
    if "show_send_form" not in st.session_state:
        st.session_state.show_send_form = False
    if "application_flow" not in st.session_state:
        st.session_state.application_flow = AppFlow()
    if "show_import" not in st.session_state:
        st.session_state.show_import = False
    if "datasets_multiselect" not in st.session_state:
        st.session_state.datasets_multiselect = [dataset.DATASET_NAME for dataset in datasets.values() if dataset.IS_DEFAULT]
    if "is_admin" not in st.session_state:
        st.session_state.is_admin = False
    if "public_rule" not in st.session_state:
        st.session_state.public_rule = False
    if "dividends_currency_match" not in st.session_state:
        st.session_state.dividends_currency_match = True
    if "use_dividends_trading" not in st.session_state:
        st.session_state.use_dividends_trading = False
    if "indicators_code_log" not in st.session_state:
        st.session_state.indicators_code_log = []
    if "trading_code_log" not in st.session_state:
        st.session_state.trading_code_log = []
    if "show_indicators_code_editor" not in st.session_state:
        st.session_state.show_indicators_code_editor = False
    if "show_trading_code_editor" not in st.session_state:
        st.session_state.show_trading_code_editor = False
    if "model_descr" not in st.session_state:
        st.session_state.model_descr = ""

    if "execution_cost_bps" not in st.session_state:
        st.session_state.execution_cost_bps = 0.0
    if "fill_trade_price" not in st.session_state:
        st.session_state.fill_trade_price = trade_fill_prices.get_default_price().BACKEND_KEY
    if "fx_rates" not in st.session_state:
        st.session_state.fx_rates = dict()
    if "supplementary_symbols_prompt" not in st.session_state:
        st.session_state.supplementary_symbols_prompt = ""
    if "supplementary_symbols" not in st.session_state:
        st.session_state.supplementary_symbols = []
    if "datasets_keys" not in st.session_state:
        st.session_state.datasets_keys = []
    if "data_provider" not in st.session_state:
        st.session_state.data_provider = ""


def build_aggrid_options(data, currency):
    df = pd.DataFrame(data)
    gb = GridOptionsBuilder.from_dataframe(df)
    gb.configure_auto_height(True)
    gridOptions = gb.build()

    if len(data) == 0:
        return gridOptions

    # gridOptions["headerHeight"] = 0
    gridOptions["columnDefs"] = []
    for field in data[0].keys():
        if not field == "sheetHierarchy":
            gridOptions["columnDefs"].append(
                {"field": field, "type": "rightAligned", "valueFormatter": f"x.toLocaleString() + ' {currency}'"}
            )
            # gb.configure_column(field,
            #         type=["numericColumn","numberColumnFilter","customNumericFormat"], precision=0)

    gridOptions["defaultColDef"] = {
        "flex": 1,
    }
    gridOptions["autoGroupColumnDef"] = (
        {
            "headerName": "Balance sheet",
            "minWidth": 300,
            "cellRendererParams": {
                "suppressCount": True,
            },
        },
    )
    gridOptions["treeData"] = True
    gridOptions["animateRows"] = True
    gridOptions["groupDefaultExpanded"] = 1
    gridOptions["getDataPath"] = JsCode(
        """ function(data){
        return data.sheetHierarchy.split("/");
    }"""
    ).js_code

    return gridOptions


def restore_indicators(data):
    periodicity = DATA_PERIODICITIES_NAMES_TO_BACKEND_KEY[ms_to_string(data["periodicity"])]
    periodicity_duration = ms_to_string(data["periodicity"])
    fetch_data_range = ms_to_string(data["time_range"])

    for data_periodicity_index, periodicity_key in enumerate(DATA_PERIODICITIES_NAMES_TO_BACKEND_KEY.keys()):
        if periodicity_key == periodicity_duration:
            break

    for fetch_data_range_index, range_key in enumerate(DATA_TIME_RANGES_NAMES_TO_BACKEND_KEY.keys()):
        if range_key == fetch_data_range:
            break

    # TODO: used as a workaround, because of old data in db; need to clean db data
    provider = DATA_PROVIDERS_NAMES_TO_BACKEND_KEY.get(data["data_source"], "")

    if not provider:
        provider = data["data_source"] if data["data_source"] is not None else ""

    st.session_state["tradable_symbols_prompt"] = data["tickers_prompt"]
    st.session_state["data_periodicity_index"] = data_periodicity_index
    st.session_state["fetch_data_range_index"] = fetch_data_range_index

    st.session_state["time_period"] = DATA_TIME_RANGES_NAMES_TO_BACKEND_KEY[ms_to_string(data["time_range"])]
    datasets_keys = data.get("datasets", None)
    if datasets_keys is None:
        datasets_keys = ["split_adj_prices"]

    if "Prices" in datasets_keys:
        price_index = datasets_keys.index("Prices")
        datasets_keys[price_index] = "split_adj_prices"

    st.session_state["datasets_multiselect"] = [
        dataset.DATASET_NAME for dataset_key, dataset in datasets.items() if dataset_key in datasets_keys
    ]
    st.session_state["datasets_keys"] = datasets_keys
    st.session_state["interval"] = periodicity
    st.session_state["provider"] = data_providers[provider].PROVIDER_NAME
    st.session_state["data_provider"] = provider

    # Call symbol step to restore fields that are not presented in the alert json
    clear_fundamentals()
    request_timestamp, fetched_symbols_meta, synth_formulas = symbol_step(st.session_state)
    st.session_state.application_flow.add_fetch_into_operation()
    fetched_data = "\n\n".join(fetched_symbols_meta)
    calculated_synthetics = "\n\n".join(
        f"`{synth_name} = {synth_formula}`" for synth_name, synth_formula in synth_formulas.items()
    )
    data_source = [
        i for i in DATA_PROVIDERS_NAMES_TO_BACKEND_KEY if DATA_PROVIDERS_NAMES_TO_BACKEND_KEY[i] == st.session_state.data_provider
    ][0]
    calculated_synthetics_logs = f"Calculated synthetics:\n\n {calculated_synthetics}" if calculated_synthetics else ""
    fetched_data_logs = f"**Fetched from {DATA_PROVIDERS_NAMES_TO_BACKEND_KEY[data_source]} (periodicity: {st.session_state.interval}, time range: {st.session_state.time_period}) at {request_timestamp}**: \n {fetched_data}"
    define_useful_strings(st.session_state)
    define_empty_indicators_step(st.session_state)

    append_log(fetched_data_logs)
    if calculated_synthetics_logs:
        append_log(calculated_synthetics_logs)
    # Call indicator step to restore fields that are not presented in the alert json
    indicators_dialogue = data["indicators_prompt"].split(PROMPTS_SEPARATOR)

    if len(indicators_dialogue) == 1 and indicators_dialogue[0] == "":
        indicators_dialogue = []

    st.session_state["indicators_dialogue"] = indicators_dialogue

    if len(indicators_dialogue) == 1 and data["indicators_logic"] is not None and "```python" in data["indicators_logic"]:
        st.session_state["indicators_dialogue"].append(data["indicators_logic"])

    if indicators_dialogue != [] and any(indicators_dialogue):
        try:
            indicator_step(st.session_state)
            ind_code_block, _ = get_code_sections(st.session_state["indicators_dialogue"][-1])
            st.session_state.application_flow.clear_backtesting_run()
            st.session_state.application_flow.add_indicators_run_into_operation()
            st.session_state.application_flow.set_prev_code(ind_code_block)
        except Exception as e:
            st.session_state.indicator_error = repr(e)

    return indicators_dialogue


def trading_rule_import_callback(trading_rule):
    keycloak_session = st.session_state.keycloak_session
    st.session_state.clear()
    st.session_state["log"] = []
    set_default_state()
    st.session_state.keycloak_session = keycloak_session

    periodicity = trading_rule["interval"]

    fetch_data_range = trading_rule["time_period"]

    for data_periodicity_index, periodicity_key in enumerate(DATA_PERIODICITIES_NAMES_TO_VALUES.keys()):
        if periodicity_key == periodicity:
            break

    for fetch_data_range_index, range_key in enumerate(DATA_TIME_RANGES_NAMES_TO_VALUES.keys()):
        if range_key == fetch_data_range:
            break

    provider = DATA_PROVIDERS_NAMES_TO_BACKEND_KEY.get(trading_rule["data_provider"], "")

    if not provider:
        provider = trading_rule["data_provider"] if trading_rule["data_provider"] is not None else ""

    st.session_state["tradable_symbols_prompt"] = trading_rule["tradable_symbols_prompt"]
    st.session_state["supplementary_symbols_prompt"] = trading_rule["supplementary_symbols_prompt"]
    st.session_state["data_periodicity_index"] = data_periodicity_index
    st.session_state["fetch_data_range_index"] = fetch_data_range_index

    if isinstance(fetch_data_range, str):
        fetch_data_range = string_to_ms(fetch_data_range)
        st.session_state["time_period"] = fetch_data_range
    else:
        st.session_state["time_period"] = fetch_data_range

    datasets_keys = trading_rule.get("datasets_keys", None)
    if datasets_keys is None:
        datasets_keys = ["split_adj_prices"]

    if "Prices" in datasets_keys:
        price_index = datasets_keys.index("Prices")
        datasets_keys[price_index] = "split_adj_prices"

    st.session_state["datasets_multiselect"] = [
        dataset.DATASET_NAME for dataset_key, dataset in datasets.items() if dataset_key in datasets_keys
    ]
    st.session_state["datasets_keys"] = datasets_keys

    if isinstance(periodicity, str):
        periodicity = DATA_PERIODICITIES_VALUES_TO_BACKEND_KEY[periodicity]
        st.session_state["interval"] = periodicity
    else:
        st.session_state["interval"] = periodicity

    st.session_state["provider"] = data_providers[provider].PROVIDER_NAME
    st.session_state["data_provider"] = provider

    # Call symbol step to restore fields that are not presented in the alert json
    clear_fundamentals()
    request_timestamp, fetched_symbols_meta, synth_formulas = symbol_step(st.session_state)
    st.session_state.application_flow.add_fetch_into_operation()
    fetched_data = "\n\n".join(fetched_symbols_meta)
    calculated_synthetics = "\n\n".join(
        f"`{synth_name} = {synth_formula}`" for synth_name, synth_formula in synth_formulas.items()
    )
    data_source = [
        i for i in DATA_PROVIDERS_NAMES_TO_BACKEND_KEY if DATA_PROVIDERS_NAMES_TO_BACKEND_KEY[i] == st.session_state.data_provider
    ][0]
    calculated_synthetics_logs = f"Calculated synthetics:\n\n {calculated_synthetics}" if calculated_synthetics else ""
    fetched_data_logs = f"**Fetched from {DATA_PROVIDERS_NAMES_TO_BACKEND_KEY[data_source]} (periodicity: {st.session_state.interval}, time range: {st.session_state.time_period}) at {request_timestamp}**: \n {fetched_data}"
    define_useful_strings(st.session_state)
    define_empty_indicators_step(st.session_state)

    append_log(fetched_data_logs)
    if calculated_synthetics_logs:
        append_log(calculated_synthetics_logs)
    # Call indicator step to restore fields that are not presented in the alert json
    indicators_dialogue = trading_rule["indicators_dialogue"]

    if len(indicators_dialogue) == 1 and indicators_dialogue[0] == "":
        indicators_dialogue = []

    st.session_state["indicators_dialogue"] = indicators_dialogue

    if (
        len(indicators_dialogue) == 1
        and trading_rule["indicators_logic"] is not None
        and "```python" in trading_rule["indicators_logic"]
    ):
        st.session_state["indicators_dialogue"].append(trading_rule["indicators_logic"])

    if indicators_dialogue != [] and any(indicators_dialogue):
        try:
            indicator_step(st.session_state)
            ind_code_block, _ = get_code_sections(st.session_state["indicators_dialogue"][-1])
            st.session_state.application_flow.clear_backtesting_run()
            st.session_state.application_flow.add_indicators_run_into_operation()
            st.session_state.application_flow.set_prev_code(ind_code_block)
        except Exception as e:
            st.session_state.indicator_error = repr(e)

    st.session_state["trading_rule_title"] = trading_rule["title"]
    st.session_state["actual_currency"] = trading_rule["actual_currency"]
    st.session_state["bet_size"] = int(trading_rule["bet_size"])
    st.session_state["total_gross_limit"] = int(trading_rule["total_gross_limit"])
    st.session_state["per_instrument_gross_limit"] = int(trading_rule["per_instrument_gross_limit"])
    st.session_state["nop_limit"] = int(trading_rule["nop_limit"])
    st.session_state["use_dividends_trading"] = trading_rule.get("use_dividends_trading", False)
    st.session_state["execution_cost_bps"] = float(trading_rule.get("execution_cost_bps", 0.0))
    st.session_state["fill_trade_price"] = trading_rule.get("fill_trade_price", None)


def trading_rule_edit_callback(trading_id, keycloak_access_token, is_public=False):
    request_headers = {"Authorization": f"Bearer {keycloak_access_token}"}

    public = "public/" if is_public else ""

    trading_json = requests.get(f"{ALERTS_BACKEND_SERVICE_URL}/api/v0/trading-info/{public}{trading_id}", headers=request_headers)

    if trading_json.status_code != 200:
        raise ConnectionError(f"Cant obtain trading rule \n{trading_json.content}")

    trading = json.loads(trading_json.content)

    # Full current session clear
    keycloak_session = st.session_state.keycloak_session
    st.session_state.clear()
    st.session_state["log"] = []
    set_default_state()
    st.session_state.keycloak_session = keycloak_session
    st.session_state["is_admin"] = "admin" in alerts_backend_proxy_singleton.get_user_roles()

    st.session_state["trading_edit"] = {}
    st.session_state["trading_edit"]["id"] = trading_id
    st.session_state["trading_edit"]["created"] = trading["created"]

    restore_indicators(trading)

    st.session_state["trading_rule_title"] = trading["title"]
    st.session_state["actual_currency"] = trading["account_currency"]
    st.session_state["bet_size"] = int(float(trading["bet_size"]))
    st.session_state["total_gross_limit"] = int(float(trading["total_gross_limit"]))
    st.session_state["per_instrument_gross_limit"] = int(float(trading["per_instrument_gross_limit"]))
    st.session_state["nop_limit"] = int(float(trading["nop_limit"]))
    st.session_state["use_dividends_trading"] = (
        trading.get("account_for_dividends", False) if trading.get("account_for_dividends", False) is not None else False
    )
    st.session_state["execution_cost_bps"] = float(
        trading.get("execution_cost_bps", None) if trading.get("execution_cost_bps", None) is not None else 0.0
    )
    st.session_state["fill_trade_price"] = trading.get("trade_fill_price", None)

    if st.session_state["is_admin"]:
        st.session_state["public_rule"] = trading["public"]

    st.session_state["model_descr"] = trading.get("description", "") if trading.get("description", "") is not None else ""

    st.session_state["app_mode"] = "Trading"
    st.session_state["mode_index"] = 1

    # Don't know why, but this thing switches to the first tab, which we exactly need
    st.empty()


def alert_edit_callback(alert_id, keycloak_access_token):
    # Get alert info for session restoring
    request_headers = {"Authorization": f"Bearer {keycloak_access_token}"}

    alert_json = requests.get(f"{ALERTS_BACKEND_SERVICE_URL}/api/v0/alerts/{alert_id}", headers=request_headers)

    if alert_json.status_code != 200:
        raise ConnectionError(f"Cant obtain alert \n{alert_json.content}")

    alert = json.loads(alert_json.content)

    # Full current session clear
    st.session_state.clear()

    restore_indicators(alert)

    # Call alert step to restore fields that are not presented in the alert json
    alerts_dialogue = alert["alerts_prompt"].split(PROMPTS_SEPARATOR)
    st.session_state["alerts_dialogue"] = alerts_dialogue
    alert_step(st.session_state)

    # Restore alert creation fields
    st.session_state["alert_title"] = alert["title"]

    for trigger_type_index, trigger_type in enumerate(TRIGGER_TYPES.values()):
        if trigger_type == alert["trigger_type"]:
            break

    st.session_state["trigger_type_index"] = trigger_type_index
    st.session_state["alert_message"] = alert["alert_text_template"]
    st.session_state["alert_email_receivers"] = alert["alert_emails"]

    # Restore timezone and activation time if periodicity == 1 day (86400000 ms)
    if alert["periodicity"] == 86400000:
        available_timezones = alerts_backend_proxy_singleton.get_available_timezones()

        for index, zone in enumerate(available_timezones):
            if zone == alert["time_zone"]:
                break

        st.session_state["timezone_index"] = index

        st.session_state["trigger_time"] = convert_ms_to_activation_time(alert["activation_time"])
    # Don't know why, but this thing switches to the first tab, which we exactly need
    st.empty()


def convert_ms_to_activation_time(ms):
    activation_time = ""
    seconds = int(ms / 1000)
    hours = int(seconds // 3600)
    activation_time += str(hours) + ":" if hours not in range(0, 10) else f"0{str(hours)}:"
    seconds -= int(hours * 3600)
    minutes = seconds // 60
    activation_time += str(minutes) + ":" if minutes not in range(0, 10) else f"0{str(minutes)}:"
    seconds -= int(minutes * 60)
    activation_time += str(seconds) if seconds not in range(0, 10) else f"0{str(seconds)}"

    return activation_time


def clear_indicators_chat_history():
    st.session_state.indicators_dialogue = []


def clear_alerts_chat_history():
    st.session_state.alerts_dialogue = []


def clear_trading_chat_history():
    st.session_state.trading_data = False
    st.session_state.trading_dialogue = []


def append_log(text: str) -> None:
    if "log" in st.session_state:
        st.session_state["log"].append(text)


def clear_log() -> None:
    if "log" in st.session_state:
        st.session_state.log = []


def price_reset_layout_callback():
    st.session_state.num_cols_price = 1
    write_chart_layout(price_chart_col_number=st.session_state.num_cols_price)


def price_add_column_callback():
    st.session_state.num_cols_price += 1
    write_chart_layout(price_chart_col_number=st.session_state.num_cols_price)


def indicators_reset_layout_callback():
    st.session_state.num_cols_indicators = 1
    write_chart_layout(indicators_chart_col_number=st.session_state.num_cols_indicators)


def indicators_add_column_callback():
    st.session_state.num_cols_indicators += 1
    write_chart_layout(indicators_chart_col_number=st.session_state.num_cols_indicators)


def alerts_reset_layout_callback():
    st.session_state.num_cols_alerts = 1
    write_chart_layout(alerts_chart_col_number=st.session_state.num_cols_alerts)


def alerts_add_column_callback():
    st.session_state.num_cols_alerts += 1
    write_chart_layout(alerts_chart_col_number=st.session_state.num_cols_alerts)


def trading_symbols_reset_layout_callback():
    st.session_state.num_cols_tradings = 1
    write_chart_layout(trading_chart_col_number=st.session_state.num_cols_tradings)


def trading_symbols_add_column_callback():
    st.session_state.num_cols_tradings += 1
    write_chart_layout(trading_chart_col_number=st.session_state.num_cols_tradings)


def trading_strategy_reset_layout_callback():
    st.session_state.num_cols_strategy_tradings = 1
    write_chart_layout(trading_strategy_chart_col_number=st.session_state.num_cols_strategy_tradings)


def trading_strategy_add_column_callback():
    st.session_state.num_cols_strategy_tradings += 1
    write_chart_layout(trading_strategy_chart_col_number=st.session_state.num_cols_strategy_tradings)


def backtest_callback():
    update_trading_symbols_chart_layouts()
    update_trading_strategy_chart_layouts()

    _, max_mem_usage, execution_time = func_profile_gen(
        trading_step, st.session_state, None, None, st.session_state.get("use_dividends_trading", False)
    )

    max_mem_usage = math.ceil(max_mem_usage)
    execution_time = math.ceil(execution_time)

    st.session_state["log"].append(
        f"Backtesting memory usage: {max_mem_usage} MB \n\n Backtesting execution time: {execution_time} seconds"
    )

    st.session_state.trading_data = True


def update_price_chart_layouts():
    chart_layout_config = read_chart_layout()
    st.session_state.num_cols_price = chart_layout_config.get("price_chart_col_number", 3)
    st.session_state.price_figs = []


def update_indicators_chart_layouts():
    chart_layout_config = read_chart_layout()
    st.session_state.num_cols_indicators = chart_layout_config.get("indicators_chart_col_number", 3)
    st.session_state.indicators_figs = []


def update_alerts_chart_layouts():
    chart_layout_config = read_chart_layout()
    st.session_state.num_cols_alerts = chart_layout_config.get("alerts_chart_col_number", 3)
    st.session_state.alerts_figs = []


def update_trading_symbols_chart_layouts():
    chart_layout_config = read_chart_layout()
    st.session_state.num_cols_tradings = chart_layout_config.get("trading_chart_col_number", 3)
    st.session_state.trading_symbols_figs = []


def update_trading_strategy_chart_layouts():
    chart_layout_config = read_chart_layout()
    st.session_state.num_cols_indicators = chart_layout_config.get("trading_strategy_chart_col_number", 3)
    st.session_state.trading_strategy_figs = []


def publish_trading_rule(keycloak_access_token, trading_rule, public=False):
    request_headers = {
        "Authorization": "Bearer " + keycloak_access_token,
        "Content-Type": "application/json",
    }

    request_body = {
        "model": st.session_state["engine"],
        "title": st.session_state["trading_rule_title"],
        "user_id": st.session_state.keycloak_session.user_info["email"],
        "data_source": st.session_state["data_provider"],
        "datasets": st.session_state.get("datasets_keys", None),
        "periodicity": string_to_ms(st.session_state.data_periodicity),
        "time_range": string_to_ms(st.session_state.time_period_),
        "tickers_prompt": st.session_state.tradable_symbols_prompt,
        "tickers": list(st.session_state["true_symbols"].values()),
        "account_for_dividends": st.session_state.get("use_dividends_trading", False),
        "active": True,
        "end_time": None,
        "public": public,
        "indicators_prompt": PROMPTS_SEPARATOR.join(st.session_state["indicators_dialogue"]),
        "indicators_logic": st.session_state.get("indicators_code", ""),
        "trading_prompt": "",
        "trading_logic": "",
        "account_currency": st.session_state["actual_currency"],
        "bet_size": str(st.session_state["bet_size"]),
        "total_gross_limit": str(st.session_state["total_gross_limit"]),
        "per_instrument_gross_limit": str(st.session_state["per_instrument_gross_limit"]),
        "nop_limit": str(st.session_state["nop_limit"]),
        "description": st.session_state["model_descr"],
        "trade_fill_price": st.session_state["fill_trade_price"]
        if st.session_state.get("fill_trade_price", "") is not None
        else "",
        "execution_cost_bps": str(st.session_state["execution_cost_bps"]),
    }

    if trading_rule is not None:
        request_body["id"] = trading_rule["id"]
        request_body["created"] = trading_rule["created"]
        resp = requests.put(f"{ALERTS_BACKEND_SERVICE_URL}/api/v0/trading-info", json=request_body, headers=request_headers)
    else:
        resp = requests.post(f"{ALERTS_BACKEND_SERVICE_URL}/api/v0/trading-info", json=request_body, headers=request_headers)

    return resp.status_code, resp.content


def publish_alerts(keycloak_access_token, alert):
    if not st.session_state.receiver_email:
        with st.container():
            st.error("Email field is empty.")
        return

    receivers = [i.strip() for i in st.session_state.receiver_email.split(",")]

    for receiver in receivers:
        if not re.match(EMAIL_REGEXP, receiver):
            with st.container():
                st.error(f"Email {receiver} is invalid.")
            return

    if len(receivers) > 1:
        receivers = "; ".join(receivers)
    else:
        receivers = receivers[0]

    activation_time = st.session_state["trigger_time"] if "day" in st.session_state.data_periodicity else 0

    if activation_time != 0:
        activation_time = datetime.strptime(activation_time, "%H:%M:%S") - datetime.strptime("00:00:00", "%H:%M:%S")
        activation_time = activation_time.seconds * 1000

    request_headers = {
        "Authorization": "Bearer " + keycloak_access_token,
        "Content-Type": "application/json",
    }

    indicators_prompt = PROMPTS_SEPARATOR.join(st.session_state["indicators_dialogue"])
    alerts_prompt = PROMPTS_SEPARATOR.join(st.session_state["alerts_dialogue"])

    request_body = {
        "user_id": st.session_state.keycloak_session.user_info["email"],
        "data_source": st.session_state.provider,
        "title": st.session_state.alert_title,
        "active": "true",
        "periodicity": string_to_ms(st.session_state.data_periodicity),
        "time_range": string_to_ms(st.session_state.time_period_),
        "tickers_prompt": st.session_state.tickers,
        "tickers": list(st.session_state["true_symbols"].values()),
        "indicators_prompt": indicators_prompt,
        "indicators_logic": st.session_state["indicators_code"],
        "alerts_prompt": alerts_prompt,
        "alerts_logic": st.session_state["alert_code"],
        "trigger_type": TRIGGER_TYPES[st.session_state["trigger_type"]],
        "alert_text_template": st.session_state["alert_message"],
        "alert_emails": receivers,
        "end_time": None,
        "activation_time": activation_time,
        "time_zone": st.session_state.get("timezone", None),
        "datasets": st.session_state.get("datasets_multiselect", None),
    }

    if alert is not None:
        request_body["id"] = alert["id"]
        request_body["created"] = alert["created"]
        resp = requests.put(f"{ALERTS_BACKEND_SERVICE_URL}/api/v0/alerts", json=request_body, headers=request_headers)
    else:
        resp = requests.post(f"{ALERTS_BACKEND_SERVICE_URL}/api/v0/alerts", json=request_body, headers=request_headers)

    return resp.status_code, resp.content


def indicator_submit_callback():
    update_indicators_chart_layouts()
    st.session_state.indicators_dialogue.append(st.session_state.indicators_query)

    try:
        result, _, execution_time = func_profile(indicator_chat, st.session_state)

        execution_time = math.ceil(execution_time)

        indicators_code, token_usage, request_timestamp = result

        model_name = ""
        for model_name, model_id in get_available_models().items():
            if model_id == st.session_state["engine"]:
                break
        token_usage = f"`Total sent: {token_usage.get('prompt_tokens')}` `Total received: {token_usage.get('completion_tokens')}`"
        logs = f"**Indicators prompt sent to OpenAI at {request_timestamp}**\n\nPrompt:`%s`\n\nModel:`{model_name}`\n\nToken usage:{token_usage}"

        log = logs % st.session_state.indicators_dialogue[-2]

        append_log(log + f"\n\nExecution time: {execution_time} seconds")
    except Exception as e:
        st.session_state.indicator_error = repr(e)


def alert_submit_callback():
    update_alerts_chart_layouts()
    st.session_state.alerts_dialogue.append(st.session_state.alerts_query)

    try:
        alert_code, token_usage, request_timestamp = alert_chat(st.session_state)

        token_usage = f"`Total sent: {token_usage.get('prompt_tokens')}` `Total received: {token_usage.get('completion_tokens')}`"
        logs = f"**Alerts prompt sent to OpenAI at {request_timestamp}**\n\nPrompt:\n\n`%s`\n\nToken usage:\n\n{token_usage}"

        append_log(logs % st.session_state.alerts_dialogue[-2])
    except Exception as e:
        st.session_state.alert_error = repr(e)


def add_symbol_button_callback(symbol, exchange):
    provider = data_providers[st.session_state["data_provider"]]
    full_symbol = provider.build_ticker_for_lookup(symbol, exchange)
    special_symbols = ["/"]

    if (
        symbol
        and symbol[0].isdigit()
        or exchange
        and exchange[0].isdigit()
        or any([i in symbol for i in special_symbols])
        or any([i in exchange for i in special_symbols])
    ):
        full_symbol = '"' + full_symbol + '"'

    if st.session_state.tradable_symbols_prompt == "":
        st.session_state.tradable_symbols_prompt = st.session_state.tradable_symbols_prompt + full_symbol
    else:
        st.session_state.tradable_symbols_prompt = st.session_state.tradable_symbols_prompt + ", " + full_symbol

    st.session_state.add_ticker_buttons.append(True)


def logout_callback(id_token):
    requests.get(
        f"https://kc.staging.deltixhub.io/realms/market_alerts/protocol/openid-connect/logout?post_logout_redirect_uri={KEYCLOAK_LOGOUT_REDIRECT_URI}&id_token_hint={id_token}"
    )

    st.session_state.logged_out = True


def trading_delete_callback(trading_id, keycloak_access_token, is_public=False):
    request_headers = {
        "Authorization": "Bearer " + keycloak_access_token,
    }

    public = "public/" if is_public else ""

    requests.delete(f"{ALERTS_BACKEND_SERVICE_URL}/api/v0/trading-info/{public}{trading_id}", headers=request_headers)


def alert_delete_callback(alert_id, keycloak_access_token):
    request_headers = {
        "Authorization": "Bearer " + keycloak_access_token,
    }

    requests.delete(f"{ALERTS_BACKEND_SERVICE_URL}/api/v0/alerts/{alert_id}", headers=request_headers)


def alert_active_callback(alert_body, keycloak_access_token, chb_key):
    alert_body["active"] = st.session_state.get(chb_key, None)

    if alert_body["active"] is None:
        return

    request_headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {keycloak_access_token}",
    }

    requests.put(f"{ALERTS_BACKEND_SERVICE_URL}/api/v0/alerts", json=alert_body, headers=request_headers)


def clear_fundamentals():
    st.session_state.income_statements = None
    st.session_state.balance_sheets = None
    st.session_state.income_currencies = None
    st.session_state.balance_currencies = None


def fetch_data() -> None:
    if not st.session_state.tradable_symbols_prompt:
        with st.container():
            st.write("\n")
            st.write("\n")
            st.error("Tickers field is empty")
        return

    # ticker_list = re.split(r"[,|+|\-|*|/|(|)]", st.session_state.tickers)
    # ticker_list = [i.strip() for i in ticker_list]
    # check_list = [" " in i for i in ticker_list]

    # if any(check_list):
    #     st.session_state["fetch_ticker_error"] = True
    #     return

    update_price_chart_layouts()
    clear_fundamentals()

    with st.spinner("Fetching data..."):
        result, max_mem_usage, execution_time = func_profile(symbol_step, st.session_state)

        max_mem_usage = math.ceil(max_mem_usage)
        execution_time = math.ceil(execution_time)

        request_timestamp, fetched_symbols_meta, synth_formulas = result
        fetched_data = "\n\n".join(fetched_symbols_meta)
        calculated_synthetics = "\n\n".join(
            f"`{synth_name} = {synth_formula}`" for synth_name, synth_formula in synth_formulas.items()
        )
        calculated_synthetics_logs = f"Calculated synthetics:\n\n {calculated_synthetics}" if calculated_synthetics else ""
        data_source = [
            p
            for p in DATA_PROVIDERS_NAMES_TO_BACKEND_KEY
            if DATA_PROVIDERS_NAMES_TO_BACKEND_KEY[p] == st.session_state.data_provider
        ][0]
        fetched_data_logs = f"**Fetched from {data_source} (periodicity: {st.session_state.interval}, time range: {st.session_state.time_period}) at {request_timestamp}**: \n {fetched_data}"
        define_useful_strings(st.session_state)
        # define_empty_indicators_step(st.session_state)

        append_log(fetched_data_logs + f"\n\nExecution time: {execution_time} seconds")
        if calculated_synthetics_logs:
            append_log(calculated_synthetics_logs)
    st.session_state.jupyter_redirect_url = ""


def event_list_update_callback():
    request_headers = {"Authorization": f"Bearer {st.session_state.keycloak_session.access_token}"}
    resp = requests.get(f"{ALERTS_BACKEND_SERVICE_URL}/api/v0/alert-events", headers=request_headers)

    if resp.status_code == 200:
        events = json.loads(resp.content)
        st.session_state.events = events
        if st.session_state.events:
            st.session_state.event_page_num = 1
            st.session_state.event_page_size = PAGINATION_MIN_PAGE_SIZE


def paginate(
    data,
    show_func: Callable,
    page_num: int,
    page_num_key: str,
    page_size: int,
    page_size_key: str,
):
    start_idx = (page_num - 1) * page_size
    end_idx = start_idx + page_size

    show_func(data[start_idx:end_idx])
    total_pages = math.ceil(len(data) / (page_size if page_size else 1))
    bottom_menu = st.columns((4, 1, 1))
    with bottom_menu[2]:
        st.selectbox("Page Size", PAGE_SIZE_OPTIONS, key=page_size_key)
    with bottom_menu[1]:
        st.number_input("Page Number", min_value=1 if len(data) else 0, max_value=total_pages, step=1, key=page_num_key)
    with bottom_menu[0]:
        st.markdown(f"[Jupyterhub workspace]({JUPYTERHUB_UI_URL})")


def check_trading_rule_title():
    return st.session_state["trading_rule_title"] is not None and st.session_state["trading_rule_title"] != ""


def send_trading_model_callback(keycloak_access_token: str, recepient: str, public: bool = False) -> Tuple[int, str]:
    # if not check_trading_rule_title():
    #     return -1, "Trading rule title is empty"

    if not re.match(EMAIL_REGEXP, recepient):
        return -1, f"Email {recepient} is invalid."

    request_headers = {
        "Authorization": "Bearer " + keycloak_access_token,
        "Content-Type": "application/json",
    }

    request_body = {
        "model": st.session_state["engine"],
        "title": st.session_state["trading_rule_title"],
        "user_id": st.session_state.keycloak_session.user_info["email"],
        "data_source": st.session_state.provider,
        "datasets": st.session_state.get("datasets_keys", None),
        "periodicity": string_to_ms(st.session_state.data_periodicity),
        "time_range": string_to_ms(st.session_state.time_period_),
        "tickers_prompt": st.session_state.tradable_symbols_prompt,
        "tickers": list(st.session_state["true_symbols"].values()),
        "account_for_dividends": st.session_state.get("use_dividends_trading", False),
        "active": True,
        "end_time": None,
        "indicators_prompt": PROMPTS_SEPARATOR.join(st.session_state["indicators_dialogue"]),
        "indicators_logic": st.session_state["indicators_code"],
        "trading_prompt": PROMPTS_SEPARATOR.join(st.session_state["trading_dialogue"]),
        "trading_logic": st.session_state.get("trading_code", ""),
        "account_currency": st.session_state["actual_currency"],
        "bet_size": str(st.session_state["bet_size"]),
        "total_gross_limit": st.session_state["total_gross_limit"],
        "per_instrument_gross_limit": st.session_state["per_instrument_gross_limit"],
        "nop_limit": st.session_state["nop_limit"],
        "description": st.session_state["model_descr"],
        "trade_fill_price": st.session_state["fill_trade_price"]
        if st.session_state.get("fill_trade_price", "") is not None
        else "",
        "execution_cost_bps": str(st.session_state["execution_cost_bps"]),
    }

    if not public:
        resp = requests.post(
            f"{ALERTS_BACKEND_SERVICE_URL}/api/v0/trading-info/share?destination={recepient}",
            json=request_body,
            headers=request_headers,
        )
    # TODO: fix this
    return resp.status_code, resp.text


def check_code_block(code_block):
    if not code_block:
        return True

    code_lines = code_block.split("\n")
    code_lines = [line.strip() for line in code_lines]

    for line in code_lines:
        if line[0] != "#":
            return False

    return True


def check_prices_multisellect_result():
    if "split_adj_prices" in st.session_state["datasets_keys"] and "split_div_adj_prices" in st.session_state["datasets_keys"]:
        price_index = st.session_state["datasets_keys"].index("split_adj_prices")
        div_price_index = st.session_state["datasets_keys"].index("split_div_adj_prices")
        remove_index = min(price_index, div_price_index)
        st.session_state["datasets_multiselect"].pop(remove_index)
        tmp = st.session_state["datasets_multiselect"]
        del st.session_state.datasets_multiselect
        st.session_state.datasets_multiselect = tmp
        return True

    return False
