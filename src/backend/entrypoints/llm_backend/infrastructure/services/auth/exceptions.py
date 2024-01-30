class MarketAlertsAuthError(Exception):
    pass


class JWKFetchError(MarketAlertsAuthError):
    pass


class InvalidJWTError(MarketAlertsAuthError):
    pass


class EmptyAuthHeaderError(MarketAlertsAuthError):
    pass


class InvalidAuthHeaderFormatError(MarketAlertsAuthError):
    pass
