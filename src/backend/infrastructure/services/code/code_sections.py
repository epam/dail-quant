import re
from typing import Any, List, Type, Union

from market_alerts.domain.exceptions import LLMBadResponseError

PYTHON_CODE_SEGMENT_PATTERN = r"```python([\s\S]*?)```"
PARAMETER_LABEL = "# parameter"
EQUALS_SEPARATOR = " = "
COMMENT_SIGN = "#"


def get_code_sections(text: str, tabs: int = 0, expected_sections_amount: int = 2) -> List[str]:
    code_sections: list[str] = re.findall(PYTHON_CODE_SEGMENT_PATTERN, text)
    code_sections = [segment.strip() for segment in code_sections]
    code_sections = ["\n".join(["    " * tabs + line for line in segment.splitlines()]) for segment in code_sections]

    if len(code_sections) != expected_sections_amount:
        raise LLMBadResponseError(
            f"Invalid code: the amount of code sections is different than {expected_sections_amount}. Generated response: {text}"
        )

    return code_sections


def update_code_sections(text: str, indicators_code: str, trading_code: str) -> str:
    indicators_code = f"```python\n{indicators_code}\n```"
    trading_code = f"```python\n{trading_code}\n```"

    replacements = [indicators_code, trading_code]

    def replace_with_new_code(match):
        return replacements.pop(0)

    text, _ = re.subn(PYTHON_CODE_SEGMENT_PATTERN, replace_with_new_code, text, count=2)

    return text


def get_optimization_params_and_code_template(code: str) -> tuple[dict[str, tuple[Any, type]], str]:
    code_template_lines = []
    optimization_params = {}

    for code_line in code.splitlines():
        if not code_line.strip().startswith(COMMENT_SIGN) and EQUALS_SEPARATOR in code_line and PARAMETER_LABEL in code_line:
            indicator_name, indicator_value = _extract_indicator_name_and_value(code_line)
            optimization_params[indicator_name] = _get_value_and_type_name(indicator_value)
            code_template_lines.append(f"{indicator_name}{EQUALS_SEPARATOR}%s {PARAMETER_LABEL}")
        else:
            code_template_lines.append(code_line)

    return optimization_params, "\n".join(code_template_lines)


def _extract_indicator_name_and_value(code_line) -> tuple[str, Union[int, float, str]]:
    indicator_name, indicator_value = [i.strip() for i in code_line.split(EQUALS_SEPARATOR)]
    indicator_value, _ = [i.strip() for i in indicator_value.split(PARAMETER_LABEL)]
    return indicator_name, _try_parse_num(indicator_value)


def _try_parse_num(num_str: str) -> Union[int, float, str]:
    try:
        return int(num_str)
    except ValueError:
        try:
            return float(num_str)
        except ValueError:
            if num_str.startswith(("'", '"')) and num_str.endswith(("'", '"')):
                return num_str[1:-1]
            else:
                return num_str


def _get_value_and_type_name(val: Any) -> tuple[Any, Type[Any]]:
    return val, type(val).__name__


def are_all_lines_comments_or_empty(text: str) -> bool:
    return all(not line.strip() or line.strip() == "pass" or line.strip().startswith(COMMENT_SIGN) for line in text.splitlines())
