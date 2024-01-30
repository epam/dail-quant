class AlertsBackendProxyError(Exception):
    pass


class LimitsInfoFetchError(AlertsBackendProxyError):
    pass


class LimitsExceededError(AlertsBackendProxyError):
    pass


class LimitsDisabled(AlertsBackendProxyError):
    pass


class UserRolesFetchError(AlertsBackendProxyError):
    pass


class TimezonesFetchError(AlertsBackendProxyError):
    pass


class SendUsedTokensInfoError(AlertsBackendProxyError):
    pass


class TradingRuleError(AlertsBackendProxyError):
    pass


class GetTradingRuleError(TradingRuleError):
    pass


class CreateTradingRuleError(TradingRuleError):
    pass


class UpdateTradingRuleError(TradingRuleError):
    pass


class ShareTradingRuleError(TradingRuleError):
    pass
