import os
from datetime import datetime
from time import time
from typing import Any, Callable, Generator, List, Tuple

from memory_profiler import memory_usage


# TODO: when we are done with supporting streamlit, I would delete convertation functions
def string_to_ms(time_string: str) -> int:
    quantifier, unit = time_string.split(" ")
    num_ms = int(quantifier.strip()) * 1000
    if "minute" in unit:
        num_ms *= 60
    elif "hour" in unit:
        num_ms *= 3600
    elif "day" in unit:
        num_ms *= 86400
    elif "week" in unit:
        num_ms *= 604800
    elif "month" in unit:
        num_ms *= 2592000
    elif "year" in unit:
        num_ms *= 31536000

    return num_ms


def ms_to_string(milliseconds: int) -> str:
    units = [
        ("year", 31536000000),
        ("month", 2592000000),
        ("week", 604800000),
        ("day", 86400000),
        ("hour", 3600000),
        ("minute", 60000),
        ("second", 1000),
    ]

    for unit, value in units:
        if milliseconds >= value:
            num_units = milliseconds // value
            return f"{num_units} {unit}{'s' if num_units > 1 else ''}"

    return f"{milliseconds} milliseconds"


def convert_date(date_str: str) -> str:
    try:
        return datetime.strptime(date_str, "%Y-%m-%d %H:%M:%S").strftime("%Y-%m-%d")
    except ValueError:
        return date_str


def func_profile(func, *args, **kwargs) -> (Any, float, float):
    start_time = time()
    max_mem_usage, result = memory_usage((func, args, kwargs), interval=0.01, max_usage=True, retval=True)
    end_time = time()

    execution_time = end_time - start_time

    return result, max_mem_usage, execution_time


def func_profile_gen(func: Callable[..., Any], *args: Any, **kwargs: Any) -> Tuple[Any, float, float]:
    start_time = time()
    max_mem_usage, result = memory_usage(
        (_generator_consumer, (func,) + args, kwargs), interval=0.01, max_usage=True, retval=True
    )
    end_time = time()

    execution_time = end_time - start_time

    return result, max_mem_usage, execution_time


def _generator_consumer(func: Callable[..., Any], *args: Any, **kwargs: Any) -> List[Any]:
    return list(func(*args, **kwargs))


def time_profile(func: Callable[..., Any], *args, **kwargs) -> Tuple[Any, float]:
    start_time = time()
    result = func(*args, **kwargs)
    end_time = time()

    return result, end_time - start_time


def progress_and_time_generator(
    progress_done: float = 0,
    progress_time_passed: float = 0,
    progress_weight: float = 1,
) -> Callable[..., Generator[Tuple[int, float, float], None, None]]:
    def decorator(
        progress_generator: Callable[..., Generator[int, None, None]], *args, **kwargs
    ) -> Generator[Tuple[int, float, float], None, None]:
        gen = progress_generator(*args, **kwargs)
        start_time = time()

        while True:
            try:
                progress = next(gen)
                progress *= progress_weight
                progress += progress_done
                time_taken_so_far = (time() - start_time) + progress_time_passed
                completion_percentage_so_far = progress / 100
                estimated_total_time = time_taken_so_far / completion_percentage_so_far
                estimated_remaining_time = estimated_total_time * (1 - completion_percentage_so_far)

                yield progress, time_taken_so_far, estimated_remaining_time
            except StopIteration as e:
                return e.value

    return decorator


def delete_temp_file(file_path: str):
    if os.path.exists(file_path):
        os.remove(file_path)


def session_state_getter(session_dict):
    return {
        "data_provider": session_dict.get("data_provider"),
        "time_period": session_dict["time_period"],
        "interval": session_dict["interval"],
        "tradable_symbols_prompt": session_dict["tradable_symbols_prompt"],
        "supplementary_symbols_prompt": session_dict["supplementary_symbols_prompt"],
        "economic_indicators": session_dict.get("economic_indicators") or [],
        "dividend_fields": session_dict.get("dividend_fields") or [],
        "fx_rates": {},
        "datasets_keys": session_dict["datasets_keys"],
        "indicators_dialogue": session_dict.get("indicators_dialogue"),
        "indicators_query": session_dict.get("indicators_dialogue")[-2] if session_dict.get("indicators_dialogue") else "",
        "indicators_logic": session_dict.get("indicators_dialogue")[-1] if session_dict.get("indicators_dialogue") else "",
        "alerts_query": session_dict.get("alerts_dialogue")[-2] if session_dict.get("alerts_dialogue") else "",
        "alerts_logic": session_dict.get("alerts_dialogue")[-1] if session_dict.get("alerts_dialogue") else "",
        "strategy_title": session_dict.get("trading_rule_title") or session_dict.get("strategy_title"),
        "strategy_description": session_dict.get("model_descr") or session_dict.get("strategy_description"),
        "actual_currency": session_dict["actual_currency"],
        "bet_size": session_dict["bet_size"],
        "per_instrument_gross_limit": session_dict["per_instrument_gross_limit"],
        "total_gross_limit": session_dict["total_gross_limit"],
        "nop_limit": session_dict["nop_limit"],
        "use_dividends_trading": session_dict.get("use_dividends_trading", False),
        "fill_trade_price": session_dict["fill_trade_price"] if session_dict.get("fill_trade_price", None) is not None else "",
        "execution_cost_bps": session_dict.get("execution_cost_bps", 0.0),
        "optimization_trials": session_dict.get("optimization_trials", 5),
        "optimization_train_size": session_dict.get("optimization_train_size", 1.0),
        "optimization_params": session_dict.get("optimization_params", []),
        "optimization_minimize": session_dict.get("optimization_minimize", True),
        "optimization_maximize": session_dict.get("optimization_maximize", True),
        "optimization_sampler": session_dict.get("optimization_sampler"),
        "optimization_target_func": session_dict.get("optimization_target_func"),
    }
