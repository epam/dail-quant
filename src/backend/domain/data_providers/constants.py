from enum import Enum
from string import Template


class AdditionalDividendFields(str, Enum):
    DividendAmount = "div_amt"
    DividendAdjustmentFactor = "div_adj_factor"
    ForwardDividendYield = "forward_div_yield"
    Trailing12MonthDividendYield = "trail_12mo_div_yield"


class Datasets(str, Enum):
    BalanceSheet = "bal"
    Dividends = "divs"
    Earnings = "earn"
    EconomicIndicators = "econ_ind"
    IncomeStatement = "inc"
    SplitAdjustedPrices = "split_adj_prices"
    SplitAndDividendAdjustedPrices = "split_div_adj_prices"


DATASETS_META = {
    Datasets.SplitAdjustedPrices: {
        "label": "Split Adjusted Prices",
        "description": "Price of the stock for the specified period.",
        "default_checked": True,
    },
    Datasets.SplitAndDividendAdjustedPrices: {
        "label": "Split & Dividend adjusted prices",
        "description": "Dividends adjusted prices.",
        "default_checked": False,
    },
    Datasets.Earnings: {
        "label": "Earnings",
        "description": "Earnings data for a given company for the specified period, including EPS estimate and EPS actual.",
        "default_checked": False,
    },
    Datasets.Dividends: {
        "label": "Dividends",
        "description": "Amount of dividends paid out for the specified period.",
        "default_checked": False,
    },
    Datasets.EconomicIndicators: {"label": "Economic Indicators", "description": "", "default_checked": False},
    Datasets.BalanceSheet: {"label": "Balance Sheet", "description": "", "default_checked": False},
    Datasets.IncomeStatement: {"label": "Income Statement", "description": "", "default_checked": False},
}


INDICATORS_PERIODICITY = {"D": {"av_key": "daily"}, "W": {"av_key": "weekly"}, "M": {"av_key": "monthly"}}

ECONOMIC_INDICATORS = {
    "DCOILWTICO": {
        "av_fetch_url": Template("https://www.alphavantage.co/query?function=WTI&interval=$periodicity&apikey=$api_key")
    },
    "MCOILWTICOA": {
        "av_fetch_url": Template("https://www.alphavantage.co/query?function=WTI&interval=$periodicity&apikey=$api_key")
    },
    "DCOILBRENTEU": {
        "av_fetch_url": Template("https://www.alphavantage.co/query?function=BRENT&interval=$periodicity&apikey=$api_key")
    },
    "DHHNGSP": {
        "av_fetch_url": Template("https://www.alphavantage.co/query?function=NATURAL_GAS&interval=$periodicity&apikey=$api_key")
    },
    "DGS10": {
        "av_fetch_url": Template(
            "https://alphavantage.co/query?function=TREASURY_YIELD&interval=$periodicity&maturity=10year&apikey=$api_key"
        )
    },
    "WGS10YR": {
        "av_fetch_url": Template(
            "https://alphavantage.co/query?function=TREASURY_YIELD&interval=$periodicity&maturity=10year&apikey=$api_key"
        )
    },
    "GS10": {
        "av_fetch_url": Template(
            "https://alphavantage.co/query?function=TREASURY_YIELD&interval=$periodicity&maturity=10year&apikey=$api_key"
        )
    },
    "DGS2": {
        "av_fetch_url": Template(
            "https://alphavantage.co/query?function=TREASURY_YIELD&interval=$periodicity&maturity=2year&apikey=$api_key"
        )
    },
    "DGS7": {
        "av_fetch_url": Template(
            "https://alphavantage.co/query?function=TREASURY_YIELD&interval=$periodicity&maturity=7year&apikey=$api_key"
        )
    },
    "DGS5": {
        "av_fetch_url": Template(
            "https://alphavantage.co/query?function=TREASURY_YIELD&interval=$periodicity&maturity=5year&apikey=$api_key"
        )
    },
    "DGS30": {
        "av_fetch_url": Template(
            "https://alphavantage.co/query?function=TREASURY_YIELD&interval=$periodicity&maturity=30year&apikey=$api_key"
        )
    },
    "DGS3MO": {
        "av_fetch_url": Template(
            "https://alphavantage.co/query?function=TREASURY_YIELD&interval=$periodicity&maturity=3month&apikey=$api_key"
        )
    },
    "DFF": {
        "av_fetch_url": Template(
            "https://alphavantage.co/query?function=FEDERAL_FUNDS_RATE&interval=$periodicity&apikey=$api_key"
        )
    },
}

WORK_UNITS_PER_DATASET = {
    # Datasets.BalanceSheet: ...,
    Datasets.Dividends: 1,
    # Datasets.Earnings: ...,
    # Datasets.EconomicIndicators: ...,
    # Datasets.IncomeStatement: ...,
    Datasets.SplitAdjustedPrices: 1,
    Datasets.SplitAndDividendAdjustedPrices: 2,
}
