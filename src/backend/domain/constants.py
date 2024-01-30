import json
import os

from market_alerts.containers import data_periodicities, data_timeranges
from market_alerts.domain.data_providers.constants import AdditionalDividendFields

DATA_PERIODICITIES_NAMES_TO_VALUES = {periodicity["label"]: periodicity["value"] for periodicity in data_periodicities.values()}

DATA_PERIODICITIES_NAMES_TO_BACKEND_KEY = {
    periodicity["label"]: backend_key for backend_key, periodicity in data_periodicities.items()
}

DATA_TIME_RANGES_NAMES_TO_VALUES = {dtr["label"]: dtr["value"] for dtr in data_timeranges.values()}

DATA_TIME_RANGES_NAMES_TO_BACKEND_KEY = {dtr["label"]: backend_key for backend_key, dtr in data_timeranges.items()}

TRIGGER_TYPES = {"Once": "ONCE", "Multiple": "MULTIPLE"}

RESOURCES_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "resources")

with open(os.path.join(RESOURCES_PATH, "indicators.json"), "r") as indicators_file:
    INDICATORS = json.load(indicators_file)

with open(os.path.join(RESOURCES_PATH, "reverse_indicators.json"), "r") as reverse_indicators_file:
    REVERSE_INDICATORS = json.load(reverse_indicators_file)

with open(os.path.join(RESOURCES_PATH, "list_symbols.json"), "r") as symbols_file:
    LIST_SYMBOLS = json.load(symbols_file)

with open(os.path.join(RESOURCES_PATH, "economic_indicators_structure.json")) as economic_indicators_file:
    ECONOMIC_INDICATORS_STRUCTURE = json.load(economic_indicators_file)

with open(os.path.join(RESOURCES_PATH, "fundamentals_structure.json")) as fundamentals_file:
    FUNDAMENTALS_STRUCTURE = json.load(fundamentals_file)

with open(os.path.join(RESOURCES_PATH, "prompts", "system_prompt.txt"), "r", encoding="utf-8") as system_prompt_file:
    SYSTEM_PROMPT = system_prompt_file.read()

with open(
    os.path.join(RESOURCES_PATH, "prompts", "initialization_block_prompt.txt"), "r", encoding="utf-8"
) as indicators_prompt_file:
    DEFAULT_INDICATORS_PROMPT = indicators_prompt_file.read()

with open(os.path.join(RESOURCES_PATH, "prompts", "trading_block_prompt.txt"), "r", encoding="utf-8") as trading_prompt_file:
    DEFAULT_TRADING_PROMPT = trading_prompt_file.read()

DIVIDEND_FIELDS = {
    "dividends": [
        {
            "label": "Dividend Amount",
            "query_param": AdditionalDividendFields.DividendAmount,
        },
        {
            "label": "Trailing 12 month Dividend Yield",
            "query_param": AdditionalDividendFields.Trailing12MonthDividendYield,
        },
        {
            "label": "Dividend Adjustment Factor",
            "query_param": AdditionalDividendFields.DividendAdjustmentFactor,
        },
        {
            "label": "Forward Dividend Yield",
            "query_param": AdditionalDividendFields.ForwardDividendYield,
        },
    ]
}

ADDITIONAL_COLUMNS = ["dividends", "earnings", "div_amt", "div_adj_factor", "trail_12mo_div_yield", "forward_div_yield"]

PROMPTS_SEPARATOR = "````message"

PAGINATION_MIN_PAGE_SIZE = 10

PAGE_SIZE_OPTIONS = [10, 20, 30, 50, 100]

DEFAULT_PAGE_SIZE = 100

APP_MODES = ["Alert", "Trading"]

DATA_PROVIDER_FX_RATES = {"TW": "%s/%s", "PI": "C:%s%s", "AV": "%s/%s"}

MODELS_WHITE__LIST = ["GPT-3.5", "GPT-4", "GPT-4-0613", "GPT-4-Turbo-1106", "Anthropic (Claude V2)"]

DEFAULT_MODEL_NAME = "GPT-4"

PUBSUB_END_OF_DATA = "END"

CODE_EDITOR_MAX_HEIGHT = 600

EMAIL_REGEXP = "^[\w\-\.]+@([\w-]+\.)+[\w-]{2,4}$"

KEYS_TO_COLUMNS = {
    "div": "dividends",
    "earn": "earnings",
    "div_amt": "div_amt",
    "div_adj_factor": "div_adj_factor",
    "trail_12mo_div_yield": "trail_12mo_div_yield",
    "forward_div_yield": "forward_div_yield",
}

COLUMNS_TO_DESCRIPTION = {
    "dividends": "Dividends.",
    "earnings": "Earnings.",
    "div_amt": "Dividend Amount.",
    "div_adj_factor": "Dividend Adjustment Factor.",
    "trail_12mo_div_yield": "Trailing 12 month Dividend Yield.",
    "forward_div_yield": "Forward Dividend Yield.",
}
