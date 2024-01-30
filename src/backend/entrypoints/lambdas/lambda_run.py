import logging
import os.path
import re
import sys
from datetime import datetime
from typing import Any, Callable, Dict, List, Tuple

import pandas as pd
import requests

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))))

logging.basicConfig(
    level=os.environ.get("LOG_LEVEL", "INFO").upper(),
    format="[%(asctime)s] [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %I:%M:%S",
    force=True,
)

_logger = logging.getLogger(__name__)


def lambda_handler(event, context):
    from market_alerts.domain.services import (
        alert_chat,
        alert_step,
        define_useful_strings,
        indicator_chat,
        indicator_step,
        symbol_step,
    )

    _logger.info("Received event: %s", event)

    send_event_to_alert_manager(event, "STARTED", "Lambda started")

    flow = [symbol_step, define_useful_strings, indicator_chat, indicator_step, alert_chat, alert_step]

    try:
        status, message = handle_alert(event, {}, flow)
    except Exception as e:
        status, message = "FAILED", f"Internal error: {str(e)}"

    send_event_to_alert_manager(event, status, message)

    return status, message


def send_event_to_alert_manager(lambda_event: Dict[str, Any], status: str, message: str) -> None:
    from market_alerts.config import ALERTS_BACKEND_SERVICE_URL

    url = f"{ALERTS_BACKEND_SERVICE_URL}/api/v0/send-alert-event"
    headers = {"Content-Type": "application/json"}
    data = {
        "alert_id": lambda_event["id"],
        "timestamp": datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%S.%fZ"),
        "title": "MY EVENT TITLE",
        "address": "SOME EVENT ADDRESS",
        "message": message,
        "status": status,
    }
    response = requests.post(url, json=data, headers=headers)
    if response.status_code == 200:
        _logger.info(f"Sent event to backend: {status}, {message}")
    else:
        _logger.error(f"POST request failed with status code: {response.status_code}. Reason: {response.text}")


def handle_alert(lambda_event: Dict[str, Any], lambda_session: Dict[str, Any], flow: List[Callable]) -> Tuple[str, str]:
    from market_alerts.config import NOTIFICATION_METHODS
    from market_alerts.domain.constants import (
        DATA_PERIODICITIES_NAMES_TO_BACKEND_KEY,
        DATA_TIME_RANGES_NAMES_TO_BACKEND_KEY,
        DATA_TIME_RANGES_NAMES_TO_VALUES,
    )
    from market_alerts.domain.services import notify
    from market_alerts.domain.services.notifier import EmailMethod
    from market_alerts.utils import ms_to_string

    lambda_session = {
        "data_provider": lambda_event["data_source"],
        "tradable_symbols_prompt": lambda_event["tradable_symbols_prompt"],
        "supplementary_symbols_prompt": lambda_event["supplementary_symbols_prompt"],
        "fx_rates": lambda_event["fx_rates"],
        "indicators_query": lambda_event["indicators_prompt"],
        "indicators_logic": lambda_event["indicators_logic"],
        "alerts_query": lambda_event["alerts_prompt"],
        "alerts_logic": lambda_event["alerts_logic"],
        "datasets_keys": lambda_event["datasets"] if lambda_event.get("datasets") is not None else ["Prices"],
        "interval": DATA_PERIODICITIES_NAMES_TO_BACKEND_KEY[ms_to_string(lambda_event["periodicity"])],
        "time_period": DATA_TIME_RANGES_NAMES_TO_BACKEND_KEY[ms_to_string(lambda_event["time_range"])],
        "indicators_dialogue": [lambda_event["indicators_prompt"]],
        "alerts_dialogue": [lambda_event["alerts_prompt"]],
        **lambda_session,
    }

    end_date = datetime.now()
    start_date = (end_date - DATA_TIME_RANGES_NAMES_TO_VALUES[ms_to_string(lambda_event["time_range"])]).strftime(
        "%Y-%m-%d %H:%M:%S"
    )
    end_date = end_date.strftime("%Y-%m-%d %H:%M:%S")

    lambda_session["start_date"] = start_date
    lambda_session["end_date"] = end_date

    for func in flow:
        func(lambda_session)

    is_alert_series = lambda_session["trigger_alert"]
    is_alert = int(is_alert_series.iloc[-1])

    if is_alert:
        notification_methods = [
            method
            for method in [
                EmailMethod(
                    recipients=lambda_event["alert_emails"],
                    subject=lambda_event["title"],
                    body=build_notification_body(lambda_event, lambda_session),
                ),
            ]
            if method.METHOD_NAME in [m.strip() for m in NOTIFICATION_METHODS.split(";")]
        ]
        notify(notification_methods)
        return "ALERTED", build_notification_body(lambda_event, lambda_session)
    else:
        return "FINISHED", "Alert was not triggered"


def build_notification_body(lambda_event: Dict[str, Any], lambda_session: Dict[str, Any]) -> str:
    alert_text_template = lambda_event["alert_text_template"]
    alert_text_template = re.sub(r"{\s*([^}]+)\s*}", lambda match: "{" + match.group(1).strip() + "}", alert_text_template)

    template_keys_and_columns = re.findall(r"{([^}\[\]]+)(\[[^]]+])?}", alert_text_template)

    def get_template_value(session_key: str, template_key: str, column: str, lambda_session: Dict[str, Any]) -> float:
        value = lambda_session[session_key][template_key]
        if isinstance(value, pd.DataFrame):
            try:
                value = value[column].iloc[-1]
            except KeyError:
                value = None
        elif isinstance(value, pd.Series):
            value = value.iloc[-1]
        return value

    mappings = {}
    for template_key, column in template_keys_and_columns:
        allowed_key = f"{template_key}_{column[2:-2]}" if column else template_key
        mappings[allowed_key] = next(
            (
                get_template_value(session_key, template_key, "close" if column == "" else column[2:-2], lambda_session)
                for session_key in ["data_by_symbol", "data_by_synth", "data_by_indicator"]
                if template_key in lambda_session[session_key]
            ),
            None,
        )

    template_with_allowed_keys = re.sub(
        r"(\w+)\['(\w+)']", lambda match: match.group(1) + "_" + match.group(2), alert_text_template
    )

    alert_text = template_with_allowed_keys.format(**mappings)
    return alert_text
