import sys
import traceback
import types
from typing import Any, Optional

from billiard.exceptions import SoftTimeLimitExceeded
from RestrictedPython import compile_restricted, safe_globals

from market_alerts.config import CODE_EXEC_RESTRICTED
from market_alerts.domain.exceptions import LLMBadResponseError


def _custom_getitem(obj, key):
    import pandas as pd

    if isinstance(obj, (pd.DataFrame, pd.Series, dict)):
        key = str(key)
    elif isinstance(obj, (list, tuple)):
        key = int(key)

    return obj[key]


def exec_code(code_str: str, glbls: dict[str, Any], lcls: dict[str, Any], code_obj: Optional[types.CodeType] = None) -> None:
    if CODE_EXEC_RESTRICTED:
        glbls = {
            "_getitem_": _custom_getitem,
            **glbls,
            **safe_globals,
        }

    code_obj = compile_code(code_str, code_obj)

    try:
        exec(code_obj, glbls, lcls)
    except SoftTimeLimitExceeded:
        code_line, line_number = _get_failed_code_line_and_number(code_str)

        error_message = f"Code execution timed out at line {str(line_number)}: '{code_line}'"

        raise SoftTimeLimitExceeded(error_message) from None
    except Exception as e:
        code_line, line_number = _get_failed_code_line_and_number(code_str)

        error_message = f"Invalid code in line {str(line_number)}: '{code_line}'. Error message: {e.__class__.__name__}: {str(e)}"

        raise LLMBadResponseError(error_message) from None


def _get_failed_code_line_and_number(code_str: str) -> tuple[str, int]:
    _, _, tb = sys.exc_info()
    line_number = traceback.extract_tb(tb)[1][1]
    code_line = code_str.split("\n")[line_number - 1] if code_str else ""
    return code_line, line_number


def compile_code(code_str: str, code_obj: Optional[types.CodeType] = None) -> types.CodeType:
    try:
        if code_obj is None:
            if CODE_EXEC_RESTRICTED:
                return compile_restricted(code_str, filename="<restricted>", mode="exec")
            else:
                return compile(code_str, filename="<non-restricted>", mode="exec")
        return code_obj
    except SyntaxError as e:
        error_message = f"Syntax error in line {str(e.lineno)}: '{e.text}'"
        raise LLMBadResponseError(error_message)
    except Exception as e:
        raise LLMBadResponseError(str(e))
