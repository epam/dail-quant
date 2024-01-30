from .base import (
    AlreadyExistsError,
    APILimitsError,
    ForbiddenError,
    InputError,
    NotFoundError,
)


class TradingRuleAlreadyExistsError(AlreadyExistsError):
    pass


class TradingRuleNotFoundError(NotFoundError):
    pass


class SaveModelInputError(InputError):
    pass


class SaveModelForbiddenError(ForbiddenError):
    pass


class ProviderAPILimitsError(APILimitsError):
    pass
