import os

LOG_LEVEL = os.environ.get("LOG_LEVEL", "INFO")

TWELVE_API_KEY = os.environ.get("TWELVE_API_KEY")

POLYGON_API_KEY = os.environ.get("POLYGON_API_KEY")

ALPHAVANTAGE_API_KEY = os.environ.get("ALPHAVANTAGE_API_KEY")

if not TWELVE_API_KEY and not POLYGON_API_KEY and not ALPHAVANTAGE_API_KEY:
    raise ValueError("Either TWELVE_API_KEY or POLYGON_API_KEY or ALPHAVANTAGE_API_KEY must be set")

ALERTS_BACKEND_SERVICE_LIMITS_ENABLED = bool(os.environ.get("ALERTS_BACKEND_SERVICE_LIMITS_ENABLED", True))
ALERTS_BACKEND_SERVICE_HOST = os.environ.get("ALERTS_BACKEND_SERVICE_HOST", "alerts-backend")
ALERTS_BACKEND_SERVICE_PORT = os.environ.get("ALERTS_BACKEND_SERVICE_PORT", 8770)
ALERTS_BACKEND_SERVICE_URL = os.environ.get(
    "ALERTS_BACKEND_SERVICE_URL", f"{ALERTS_BACKEND_SERVICE_HOST}:{ALERTS_BACKEND_SERVICE_PORT}"
)

SES_SENDER_EMAIL = os.environ.get("SES_SENDER_EMAIL", "noreply@quantoffice.cloud")
SES_REGION = os.environ.get("SES_REGION", "us-east-1")

KEYCLOAK_LOGOUT_REDIRECT_URI = os.environ.get("KEYCLOAK_LOGOUT_REDIRECT_URI", "http://localhost:8501")

NOTIFICATION_METHODS = os.environ.get("NOTIFICATION_METHODS", "email")

HEALTHCHECK_ALERT_ID = os.environ.get("HEALTHCHECK_ALERT_ID")
# Format: seconds-minutes-hours-days-weeks
HEALTHCHECK_PERIOD = os.environ.get("HEALTHCHECK_PERIOD", "0-0-1-0-0")
HEALTHCHECK_EMAIL_RECIPIENTS = os.environ.get("HEALTHCHECK_EMAIL_RECIPIENTS")
HEALTHCHECK_MS_TEAMS_WEBHOOK = os.environ.get("MS_TEAMS_WEBHOOK")
HEALTHCHECK_NOTIFICATION_METHODS = os.environ.get("HEALTHCHECK_NOTIFICATION_METHODS", "teams;email")

DB_HOST = os.environ.get("DB_HOST")
DB_PORT = os.environ.get("DB_PORT", 3306)
DB_USER = os.environ.get("DB_USER")
DB_PASSWORD = os.environ.get("DB_PASSWORD")
DB_DATABASE = os.environ.get("DB_DATABASE", "market_alerts_service")
DB_DIALECT = os.environ.get("DB_DIALECT", "mysql")
DB_DRIVER = os.environ.get("DB_DRIVER", "mysqlconnector")

CODE_EXEC_RESTRICTED = os.environ.get("CODE_EXEC_RESTRICTED", False)

JUPYTERHUB_UI_URL = os.environ.get("JUPYTERHUB_UI_URL", "http://localhost:8001")
JUPYTERHUB_SERVICE_URL = os.environ.get("JUPYTERHUB_SERVICE_URL", "http://mas-jupyter:8000")
JUPYTERHUB_TOKEN = os.environ.get("JUPYTERHUB_TOKEN", "12345678")

KEYCLOAK_CLIENT_ID = os.environ.get("KEYCLOAK_CLIENT_ID", "market_alerts_service")
