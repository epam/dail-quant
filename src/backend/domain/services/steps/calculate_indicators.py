import inspect
import json
import os
import types
from collections import defaultdict
from datetime import datetime
from functools import partial
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from market_alerts.containers import (
    UserPromptTypes,
    alerts_backend_proxy_singleton,
    data_periodicities,
    data_providers,
)
from market_alerts.domain.constants import (
    DATA_PROVIDER_FX_RATES,
    DEFAULT_INDICATORS_PROMPT,
    DEFAULT_TRADING_PROMPT,
    RESOURCES_PATH,
    SYSTEM_PROMPT,
)
from market_alerts.domain.exceptions import LLMBadResponseError
from market_alerts.domain.services.steps.utils import (
    get_fx_rate,
    get_globals,
    get_print_redirect,
)
from market_alerts.infrastructure.services.code import exec_code, get_code_sections


def get_indicators_promt() -> Tuple[str, str, str]:
    if os.environ.get("DEBUG_PROMPT", "False") == "True":
        with open(os.path.join(RESOURCES_PATH, "prompts", "system_prompt.txt"), "r") as system_prompt_file:
            system_prompt = system_prompt_file.read()

        with open(os.path.join(RESOURCES_PATH, "prompts", "initialization_block_prompt.txt"), "r") as indicators_prompt_file:
            indicators_prompt = indicators_prompt_file.read()

        with open(os.path.join(RESOURCES_PATH, "prompts", "trading_block_prompt.txt"), "r") as trading_prompt_file:
            trading_prompt = trading_prompt_file.read()

        return system_prompt, indicators_prompt, trading_prompt
    else:
        return SYSTEM_PROMPT, DEFAULT_INDICATORS_PROMPT, DEFAULT_TRADING_PROMPT


if not os.environ.get("AWS_LAMBDA_FUNCTION_NAME"):
    from market_alerts.openai_utils import (
        get_openai_stream_result,
        make_history_context,
    )

    def _get_indicators(session_dict, user_prompt_ids, engine: str):
        session_dict["user_prompt_ids"] = user_prompt_ids

        if not user_prompt_ids:
            for prompt_type in UserPromptTypes:
                session_dict[prompt_type] = ""
        else:
            user_prompts = alerts_backend_proxy_singleton.get_prompts(session_dict["user_prompt_ids"])

            for prompt_type in UserPromptTypes:
                prompts = list(filter(lambda x: x["scope"] == prompt_type, user_prompts))
                prompts = [prompt["prompt"] for prompt in prompts]

                session_dict[prompt_type] = "\n".join(prompts)

        system_prompt, indicators_prompt, trading_prompt = get_indicators_promt()

        indicators_prompt = indicators_prompt % (
            session_dict["u_strs"]["additional_columns_table"],
            session_dict["interval"],
        )

        prompt = system_prompt % (
            session_dict[UserPromptTypes.Generic],
            indicators_prompt,
            session_dict[UserPromptTypes.IndicatorBlockOnly],
            trading_prompt,
            session_dict[UserPromptTypes.TradingBlockOnly],
            session_dict["u_strs"]["list_securities_str"],
            session_dict["u_strs"]["list_sparse_securities_str"],
            session_dict["u_strs"]["list_economic_indicator_symbols_str"],
            session_dict["u_strs"]["input_securities_str"],
            session_dict["synth_formulas"],
            session_dict["u_strs"]["all_indicators_str"],
        )

        context = make_history_context(
            prompt,
            session_dict["indicators_dialogue"],
            "If there is code in the answer, make sure it is complete. Don't forget to send both indicators and trading blocks of code.",
        )

        if os.environ.get("DEBUG_PROMPT", "False") == "True":
            with open("market_alerts/prompt-debug.txt", "w") as file:
                file.write(str(context).replace("\\n", "\n"))

        session_dict["last_llm_context"] = json.dumps({"messages": context})

        result = get_openai_stream_result(context, engine)

        if os.environ.get("DEBUG_PROMPT", "False") == "True":
            with open("market_alerts/prompt-debug.txt", "a") as file:
                file.write("")
                file.write("################################################################################################")
                file.write("")
                file.write(str(result).replace("\\n", "\n"))

        return result

else:

    def _get_indicators(session_dict, user_prompt_ids, engine: str):
        return session_dict["indicators_logic"], {"prompt_tokens": 0, "completion_tokens": 0}


def indicator_chat(session_dict, user_prompt_ids, engine: str = "gpt-4") -> Tuple[str, Dict[str, int], str]:
    request_timestamp = datetime.now().strftime("%d/%m/%Y %H:%M:%S")

    try:
        indicators_code, token_usage = _get_indicators(session_dict, user_prompt_ids, engine)
        session_dict["engine"] = engine
    except Exception as e:
        if len(session_dict["indicators_dialogue"]) % 2 != 0:
            session_dict["indicators_dialogue"].append("Something went wrong, please try again later.")
        raise e

    session_dict["indicators_dialogue"].append(indicators_code)

    return indicators_code, token_usage, request_timestamp


def convert_using_fx_rate(
    data_provider, fx_rates, start_date, end_date, interval, data: pd.Series | pd.DataFrame, from_currency: str, to_currency: str
) -> pd.Series | pd.DataFrame:
    fx_rate_template = DATA_PROVIDER_FX_RATES[data_provider]
    provider = data_providers[data_provider]

    time_line = data.index
    fx_rate_symbol = fx_rate_template % (to_currency, from_currency)
    if not fx_rate_symbol in fx_rates:
        if from_currency == to_currency:
            fx_rates[fx_rate_symbol] = pd.DataFrame(data=1.0, columns=["open", "high", "low", "close"], index=time_line)
        else:
            fx_rates[fx_rate_symbol] = get_fx_rate(
                time_line,
                fx_rate_symbol,
                start_date,
                end_date,
                data_periodicities[interval]["value"],
                provider,
            )
    if type(data) == pd.Series:
        return data / fx_rates[fx_rate_symbol]["close"]
    elif type(data) == pd.DataFrame:
        data = data.copy()
        for column in data.columns:
            if column != "volume":
                data[column] = data[column] / fx_rates[fx_rate_symbol]["close"]
        return data
    else:
        raise LLMBadResponseError("Wrong parameters in convert_using_fx_rate.")


def indicator_step(session_dict, handler_func: Optional[Callable[[Any], Any]] = None) -> None:
    lclsglbls = get_globals()
    for key in session_dict["data_by_symbol"]:
        lclsglbls[key] = session_dict["data_by_symbol"][key]
    for key in session_dict["data_by_synth"]:
        lclsglbls[key] = session_dict["data_by_synth"][key]

    lclsglbls["symbols"] = session_dict["symbols"]
    lclsglbls["supplementary_symbols"] = session_dict["supplementary_symbols"]
    lclsglbls["tradable_symbols"] = session_dict["tradable_symbols"]
    lclsglbls["sparse_symbols"] = session_dict["sparse_symbols"]
    lclsglbls["data_by_symbol"] = session_dict["data_by_symbol"]
    lclsglbls["symbol_to_currency"] = session_dict["symbol_to_currency"]
    lclsglbls["time_line"] = session_dict["time_line"]

    lclsglbls["economic_indicator_symbols"] = session_dict["economic_indicator_symbols"]

    lclsglbls["convert_using_fx_rate"] = partial(
        convert_using_fx_rate,
        session_dict["data_provider"],
        session_dict["fx_rates"],
        session_dict["start_date"],
        session_dict["end_date"],
        session_dict["interval"],
    )

    if handler_func is not None:
        handler_func_with_updated_globals = types.FunctionType(handler_func.__code__, lclsglbls, handler_func.__name__)
        lclsglbls.update(handler_func_with_updated_globals())
        indicators_code = inspect.getsource(handler_func)
        indicators_list = _get_indicators_list(indicators_code)
    else:
        lclsglbls["print"], session_dict["indicators_code_log"] = get_print_redirect()

        llm_response = session_dict["indicators_dialogue"][-1]

        indicators_code, _ = get_code_sections(llm_response)

        exec_code(indicators_code, lclsglbls, lclsglbls)

        del lclsglbls["__builtins__"]
        indicators_list = _get_indicators_list(indicators_code)

    data_by_indicator = dict()
    key_by_indicator = dict()
    for ind in indicators_list:
        if ind in lclsglbls and type(lclsglbls[ind]) == pd.Series:
            data_by_indicator[ind] = lclsglbls[ind]
            key_by_indicator[ind] = None

    for key in indicators_list:
        if key in lclsglbls:
            if type(lclsglbls[key]) == dict and len(lclsglbls[key]) >= 1 and type(list(lclsglbls[key].values())[0]) == pd.Series:
                for k in lclsglbls[key]:
                    data_by_indicator[key + "_" + "".join(l for l in str(k) if l.isalnum())] = lclsglbls[key][k]
                    key_by_indicator[key + "_" + "".join(l for l in str(k) if l.isalnum())] = k
            if type(lclsglbls[key]) == list and len(lclsglbls[key]) >= 1 and type(lclsglbls[key][0]) == pd.Series:
                for i in range(len(lclsglbls[key])):
                    data_by_indicator[key + "_" + str(i)] = lclsglbls[key][i]
                    key_by_indicator[ind] = None
            if (
                type(lclsglbls[key]) == pd.DataFrame
                and lclsglbls[key].shape[1] >= 1
                and key not in session_dict["data_by_symbol"]
            ):
                for k in lclsglbls[key]:
                    data_by_indicator[key + "_" + "".join(l for l in str(k) if l.isalnum())] = lclsglbls[key][k]
                    key_by_indicator[key + "_" + "".join(l for l in str(k) if l.isalnum())] = k

    session_dict["lclsglbls"] = lclsglbls
    session_dict["key_by_indicator"] = key_by_indicator

    data_by_indicator = unify_indexes_of_indicators(session_dict["data_by_symbol"], data_by_indicator)
    indicators_list = list(data_by_indicator.keys())
    indicators_str = ", ".join(indicators_list)
    session_dict["indicators"] = indicators_list
    session_dict["data_by_indicator"] = data_by_indicator
    session_dict["u_strs"]["indicators_str"] = indicators_str
    session_dict["indicators_code"] = indicators_code

    roots, main_roots = _get_graph(session_dict)
    session_dict["roots"] = roots
    session_dict["main_roots"] = main_roots


def _get_indicators_list(indicators_code: str) -> List[str]:
    return [
        i.split(" = ")[0].strip()
        for i in indicators_code.split("\n")
        if " = " in i and i.strip()[0] != "#" and ("#DONTSHOW" not in i and "# DONTSHOW" not in i)
    ]


def get_sim_coef(x, y):
    abs_sum = np.abs(x) + np.abs(y)
    diff_abs = np.abs(x - y)
    mask = (~np.isnan(abs_sum)) & (~np.isnan(diff_abs))
    if np.any(mask):
        coefs = 2 * diff_abs[mask] / (abs_sum[mask] + 1e-20)
        coef = coefs.mean()
        return coef
    else:
        return 1.0


def _get_graph(session_dict):
    vectors = {key: session_dict["data_by_symbol"][key]["close"] for key in session_dict["data_by_symbol"]}
    for key in session_dict["data_by_indicator"]:
        try:
            vectors[key] = session_dict["data_by_indicator"][key].values.astype(np.float64)
        except:
            if key in session_dict["key_by_indicator"]:
                del session_dict["key_by_indicator"][key]
                del session_dict["data_by_indicator"][key]
    max_len = max([len(vectors[i]) for i in vectors])
    for key in vectors:
        vectors[key] = np.pad(vectors[key], (0, max_len - vectors[key].shape[0]), mode="constant", constant_values=np.nan)

    keys = list(vectors.keys())
    sim_coef = defaultdict(lambda: dict())
    for i in range(len(keys)):
        for j in range(i + 1, len(keys)):
            sim = get_sim_coef(vectors[keys[i]], vectors[keys[j]])
            sim_coef[keys[i]][keys[j]] = sim
            sim_coef[keys[j]][keys[i]] = sim

    threshold = 0.2

    roots = dict()
    for s in session_dict["data_by_symbol"]:
        roots[s] = []

    root_by_id = {r: r for r in session_dict["data_by_symbol"]}

    main_root_by_id = {r: r for r in session_dict["data_by_symbol"]}
    main_roots = dict()
    for s in session_dict["data_by_symbol"]:
        main_roots[s] = []

    for key, value in session_dict["key_by_indicator"].items():
        if value is not None and value in main_root_by_id:
            main_roots[main_root_by_id[value]].append(key)
            main_root_by_id[key] = main_root_by_id[main_root_by_id[value]]
        elif value is None:
            for m_root in main_roots:
                if m_root.lower() in [i.lower() for i in key.split("_")]:
                    main_roots[main_root_by_id[m_root]].append(key)
                    main_root_by_id[key] = main_root_by_id[main_root_by_id[m_root]]
                    break

    for key in keys:
        try:
            if key in main_root_by_id and key not in root_by_id:
                sorted_sims = sorted(sim_coef[key].items(), key=lambda x: x[1])
                for sim_pair in sorted_sims:
                    if sim_pair[1] >= threshold:
                        break
                    if (
                        sim_pair[0] in main_root_by_id
                        and sim_pair[0] in root_by_id
                        and main_root_by_id[key] == main_root_by_id[sim_pair[0]]
                    ):
                        main_roots[main_root_by_id[key]] = [i for i in main_roots[main_root_by_id[key]] if i != key]
                        roots[root_by_id[sim_pair[0]]].append(key)
                        root_by_id[key] = root_by_id[sim_pair[0]]
                        break
                if key not in root_by_id:
                    roots[key] = []
                    root_by_id[key] = key

            elif key not in main_root_by_id and key not in root_by_id:
                sorted_sims = sorted(sim_coef[key].items(), key=lambda x: x[1])
                for sim_pair in sorted_sims:
                    if sim_pair[1] >= threshold:
                        break
                    if sim_pair[0] in main_root_by_id and sim_pair[0] in root_by_id:
                        main_root_by_id[key] = main_root_by_id[sim_pair[0]]
                        roots[root_by_id[sim_pair[0]]].append(key)
                        root_by_id[key] = root_by_id[sim_pair[0]]
                        break
                if key not in root_by_id:
                    roots[key] = []
                    root_by_id[key] = key
                    main_roots[key] = []
                    main_root_by_id[key] = key
        except KeyError:
            roots[key] = []
            root_by_id[key] = key
            main_roots[key] = []
            main_root_by_id[key] = key

    for key in session_dict["data_by_indicator"]:
        if not key in root_by_id:
            roots[key] = []
        if not key in main_root_by_id:
            main_roots[key] = []
    return roots, main_roots


def unify_indexes_of_indicators(data_by_symbol: Dict[str, Any], data_by_indicator: Dict[str, Any]) -> Dict[str, Any]:
    idxes_union = pd.Index([])
    for key in data_by_symbol:
        idxes_union = idxes_union.union(data_by_symbol[key].index)
    idxes_union = idxes_union.sort_values()
    for key in data_by_indicator:
        data_by_indicator[key] = data_by_indicator[key][~data_by_indicator[key].index.duplicated(keep="first")]
        data_by_indicator[key] = data_by_indicator[key].reindex(idxes_union).fillna(method="ffill")
    return data_by_indicator
