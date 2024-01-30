import logging
from datetime import datetime
from typing import Any, Dict, List, Tuple

from market_alerts.containers import data_timeranges
from market_alerts.utils import string_to_ms


def check_data(tickers: List[str], twlv_data: Dict[str, Any], alph_data: Dict[str, Any], volume_diff_treshold=0.03):
    for ticker in tickers:
        logging.info("Checking data for ticker: " + ticker)

        twlv_dict = {
            value["datetime"]: {
                "open": value["open"],
                "high": value["high"],
                "low": value["low"],
                "close": value["close"],
                "volume": value["volume"],
            }
            for value in twlv_data[ticker]["values"]
        }
        alph_dict = {
            value["datetime"]: {
                "open": value["open"],
                "high": value["high"],
                "low": value["low"],
                "close": value["close"],
                "volume": value["volume"],
            }
            for value in alph_data[ticker]["values"]
        }

        for datetime, alph_value in alph_dict.items():
            twlv_value = twlv_dict[datetime]

            if not all(
                [
                    round(float(alph_value[field]), 2) == round(float(twlv_value[field]), 2)
                    for field in ["open", "high", "low", "close"]
                ]
            ):
                warning_msg = f"Following prices at {datetime} dont match:\n" + build_price_table(twlv_value, alph_value)

                logging.warning(warning_msg)

            volume_diff = abs(float(alph_value["volume"]) - float(twlv_value["volume"]))
            percentage_diff = volume_diff / float(twlv_value["volume"])

            if percentage_diff > volume_diff_treshold:
                warning_msg = f"Following volumes in prices at {datetime} dont match:\n" + build_price_table(
                    twlv_value, alph_value
                )

                logging.warning(warning_msg)


def build_price_table(twlv_value, alph_value):
    fields = ["open", "high", "low", "close", "volume"]
    providers = ["TW", "AV"]
    row_format = "{:>15}" * (len(providers) + 1)
    row_format += "\n"

    warning_msg = row_format.format("", providers[0], providers[1])
    for field in fields:
        warning_msg += row_format.format(field, round(float(twlv_value[field]), 2), round(float(alph_value[field]), 2))

    return warning_msg


def convert_timerange(timerange: str) -> Tuple[str, str]:
    data_timerange = data_timeranges[string_to_ms(timerange)]

    request_end_date = datetime.now()
    request_start_date = request_end_date - data_timerange["value"]

    end_date = request_end_date.strftime("%Y-%m-%d %H:%M:%S")
    start_date = request_start_date.strftime("%Y-%m-%d %H:%M:%S")

    return start_date, end_date
