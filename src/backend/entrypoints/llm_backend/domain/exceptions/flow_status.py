from market_alerts.domain.exceptions.base import InputError, NotFoundError


class DataNotFetchedError(NotFoundError):
    code = "data_not_fetched_error"

    def __init__(self, detail: str = "Data not fetched"):
        super().__init__(detail)


class DatasetNotRequestedError(NotFoundError):
    code = "dataset_not_requested_error"

    def __init__(self, detail: str = "Dataset not requested"):
        super().__init__(detail)


class LLMChatNotSubmittedError(NotFoundError):
    code = "llm_chat_not_submitted_error"

    def __init__(self, detail: str = "LLM chat not submitted"):
        super().__init__(detail)


class LLMChatHistoryClearedError(NotFoundError):
    code = "llm_chat_history_cleared_error"

    def __init__(self, detail: str = "LLM chat history cleared"):
        super().__init__(detail)


class IndicatorsBlockIsNotPresentError(NotFoundError):
    code = "indicators_block_is_not_present_error"

    def __init__(self, detail: str = "Indicators block is not present"):
        super().__init__(detail)


class TradingBlockIsNotPresentError(NotFoundError):
    code = "trading_block_is_not_present_error"

    def __init__(self, detail: str = "Trading block is not present"):
        super().__init__(detail)


class IndicatorsNotGeneratedError(NotFoundError):
    code = "indicators_not_generated_error"

    def __init__(self, detail: str = "Indicators not generated"):
        super().__init__(detail)


class BacktestingNotPerformedError(NotFoundError):
    code = "backtesting_not_performed_error"

    def __init__(self, detail: str = "Backtesting not performed"):
        super().__init__(detail)


class OptimizationNotPerformedError(NotFoundError):
    code = "optimization_not_performed_error"

    def __init__(self, detail: str = "Optimization not performed"):
        super().__init__(detail)


class InvalidOptimizationParamError(InputError):
    code = "invalid_optimization_param_error"

    def __init__(self, detail: str = "Invalid optimization param"):
        super().__init__(detail)
