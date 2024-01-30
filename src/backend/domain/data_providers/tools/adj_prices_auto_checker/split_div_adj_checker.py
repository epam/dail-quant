import argparse
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

path = str(Path(__file__).parents[5]).replace("\\", "\\\\").replace("/", "//")
sys.path.append(path)

os.environ["TWELVE_API_KEY"] = ""
os.environ["ALPHAVANTAGE_API_KEY"] = ""

from utils import check_data, convert_timerange

from market_alerts.config import ALPHAVANTAGE_API_KEY, TWELVE_API_KEY
from market_alerts.domain.data_providers import (
    AlphaVantageDataProvider,
    TwelveDataProvider,
)

parser = argparse.ArgumentParser(
    prog="SplitAndDividendAdjustedPricesAutoChecker",
    description="CLI tool for compairing twelve split and dividend adjusted prices to alphavantage split adjusted prices",
)

parser.add_argument("tickers", metavar="T", type=str, nargs="+", help="Tickers that will be fetched and checked")
parser.add_argument(
    "-t",
    "--timerange",
    dest="timerange",
    choices=["1 day", "1 week", "1 month", "3 months", "6 months", "1 year", "5 years", "10 years", "20 years", "30 years"],
    default="10 years",
    nargs="?",
    help="Data fetch time range (10 years is default)",
)

args = parser.parse_args()


def main(args):
    tickers = args.tickers
    timerange = args.timerange

    start_date, end_date = convert_timerange(timerange)

    twlv_data, alph_data = get_data(tickers, start_date, end_date)

    check_data(tickers, twlv_data, alph_data)


def get_data(tickers: List[str], start_date: str, end_date: str) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    alphavantage_provider = AlphaVantageDataProvider(alpha_vantage_api_key=ALPHAVANTAGE_API_KEY)
    twelve_provider = TwelveDataProvider(twelve_api_key=TWELVE_API_KEY)

    alph_data = alphavantage_provider.get_parallel_dividend_adjusted_prices(tickers, start_date, end_date, "1day")

    twlv_data = twelve_provider.get_parallel_dividend_adjusted_prices(tickers, start_date, end_date, "1day")

    return twlv_data, alph_data


main(args)
