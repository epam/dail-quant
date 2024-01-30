PROJECT_NAME = "market_alerts"
VERSION = "0.0.1"
DESCRIPTION = ""
V1_PREFIX = "/v1"
SWAGGER_DOC_URL = "/docs"

MARKET_ALERTS_BASE_API_PREFIX = "/api/market_alerts"
MARKET_ALERTS_API_PREFIX = MARKET_ALERTS_BASE_API_PREFIX + V1_PREFIX
MARKET_ALERTS_PUBLIC_ENDPOINTS = (
    "/api/market_alerts/v1/docs",
    "/api/market_alerts/debug/500",
    "/api/market_alerts/healthcheck",
    "/favicon.ico",
    "/api/market_alerts/v1/openapi.json",
    "/websockets/progress",
)
MARKET_ALERTS_NO_SESSION_ENDPOINTS = ("/api/market_alerts/v1/session",)
MARKET_ALERTS_READ_ONLY_SESSION_ENDPOINTS = (
    "/api/market_alerts/v1/tasks",
    "/api/market_alerts/v1/tickers",
    "/api/market_alerts/v1/plots",
    "/api/market_alerts/v1/ui",
    "/api/market_alerts/v1/jupyter",
    "/api/market_alerts/v1/alerts_backend",
    "/api/market_alerts/v1/files",
    "/api/market_alerts/v1/optimization",
    "/api/market_alerts/v1/llm/chat/v2",
    "/api/market_alerts/v1/llm/calculate_indicators",
    "/api/market_alerts/v1/llm/calculate_backtesting",
)
