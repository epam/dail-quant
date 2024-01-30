import inspect
import os
import types
from datetime import datetime
from typing import Any, Callable, Dict, Optional, Tuple

from market_alerts.containers import alerts_backend_proxy_singleton
from market_alerts.domain.prompts import ALERTRULE_PROMPT
from market_alerts.domain.services.steps.utils import get_globals
from market_alerts.infrastructure.services.code import exec_code, get_code_sections
from market_alerts.infrastructure.services.proxy.alerts_backend.exceptions import (
    LimitsDisabled,
)

# If not running inside AWS Lambda, then we use OpenAI
if not os.environ.get("AWS_LAMBDA_FUNCTION_NAME"):
    from market_alerts.openai_utils import (
        get_openai_stream_result,
        make_history_context,
    )

    def _get_alert_code(session_dict, engine: str):
        prompt = ALERTRULE_PROMPT % (
            session_dict["interval"],
            session_dict["u_strs"]["list_securities_str"],
            session_dict["u_strs"]["input_securities_str"],
            session_dict["synth_formulas"],
            session_dict["indicators_code"],
            session_dict["u_strs"]["all_indicators_str"],
        )
        context = make_history_context(
            prompt, session_dict["alerts_dialogue"], "If there is code in the answer, make sure it is complete."
        )
        return get_openai_stream_result(context, engine)

else:

    def _get_alert_code(session_dict, engine: str):
        return session_dict["alerts_logic"], {"prompt_tokens": 0, "completion_tokens": 0}


def alert_chat(session_dict, engine: str = "gpt-4") -> Tuple[str, Dict[str, int], str]:
    request_timestamp = datetime.now().strftime("%d/%m/%Y %H:%M:%S")

    try:
        alert_code, token_usage = _get_alert_code(session_dict, engine)
    except Exception as e:
        if len(session_dict["alerts_dialogue"]) % 2 != 0:
            session_dict["alerts_dialogue"].append("Something went wrong, please try agail later.")
        raise e

    try:
        alerts_backend_proxy_singleton.send_used_tokens_info(
            token_usage["prompt_tokens"],
            token_usage["completion_tokens"],
        )
    except TokenLimitsDisabled:
        pass

    session_dict["alerts_dialogue"].append(alert_code)

    return alert_code, token_usage, request_timestamp


def alert_step(session_dict, handler_func: Optional[Callable[[Any], Any]] = None) -> None:
    if handler_func is not None:
        handler_func_with_updated_globals = types.FunctionType(
            handler_func.__code__, _get_handler_globals(session_dict), handler_func.__name__
        )
        locals().update(handler_func_with_updated_globals())
        alert_code = inspect.getsource(handler_func)
        llm_response = f"""```python
{alert_code}
```"""
    else:
        llm_response = session_dict["alerts_dialogue"][-1]
        [alert_code] = get_code_sections(llm_response, expected_sections_amount=1)
        dummy_code = _get_dummy_code(session_dict, alert_code)
        exec_code(dummy_code, get_globals(), locals())

    session_dict["alert_code"] = llm_response
    session_dict["trigger_alert"] = locals()["trigger_alert"]


def _get_handler_globals(session_dict) -> Dict[str, Any]:
    return {
        **get_globals(),
        **session_dict["data_by_symbol"],
        **session_dict["data_by_synth"],
        **session_dict["data_by_indicator"],
    }


def _get_dummy_code(session_dict, alert_code: str) -> str:
    dummy_code = """
%s
%s
""" % (
        _get_def_code(session_dict),
        alert_code,
    )
    return dummy_code


def _get_def_code(session_dict):
    def_code = "\n".join(["%s = session_dict['data_by_symbol']['%s']" % (key, key) for key in session_dict["data_by_symbol"]])
    def_code += "\n" + "\n".join(
        ["%s = session_dict['data_by_synth']['%s']" % (key, key) for key in session_dict["data_by_synth"]]
    )
    def_code += "\n" + "\n".join(
        ["%s = session_dict['data_by_indicator']['%s']" % (key, key) for key in session_dict["data_by_indicator"]]
    )
    return def_code
