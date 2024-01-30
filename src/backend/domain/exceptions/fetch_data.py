from .base import InputError, NotFoundError, NotImplementedError


class DataNotFoundError(NotFoundError):
    code = "data_not_found_error"


class TickerInputError(InputError):
    code = "ticker_input_error"


class MethodIsNotImplementedError(NotImplementedError):
    code = "method_is_not_implemented_error"
