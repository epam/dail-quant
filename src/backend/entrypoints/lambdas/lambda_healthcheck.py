import logging
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

logging.basicConfig(
    level=os.environ.get("LOG_LEVEL", "INFO").upper(),
    format="[%(asctime)s] [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %I:%M:%S",
    force=True,
)

_logger = logging.getLogger(__name__)

EVENT_SUCCESS_STATUS = 0


def lambda_handler(event, context):
    import datetime

    from market_alerts.config import (
        DB_HOST,
        HEALTHCHECK_ALERT_ID,
        HEALTHCHECK_EMAIL_RECIPIENTS,
        HEALTHCHECK_MS_TEAMS_WEBHOOK,
        HEALTHCHECK_NOTIFICATION_METHODS,
        HEALTHCHECK_PERIOD,
    )
    from market_alerts.domain.services import notify
    from market_alerts.domain.services.notifier import EmailMethod, TeamsWebhookMethod
    from market_alerts.infrastructure.datasource import database
    from market_alerts.infrastructure.repositories import AlertEventRepository

    _logger.info("Received event: %s", event)

    try:
        seconds, minutes, hours, days, weeks = map(int, HEALTHCHECK_PERIOD.split("-"))
        time_period = datetime.timedelta(seconds=seconds, minutes=minutes, hours=hours, days=days, weeks=weeks)
    except Exception as e:
        _logger.error(f"Error parsing HEALTHCHECK_PERIOD env variable: {e}")
        return

    _logger.info(f"Connecting to the database {DB_HOST}...")

    database.connect()
    alert_event_repository = AlertEventRepository(database)

    _logger.info(f"Fetching events...")

    events = alert_event_repository.get_events_of_alert_for_period(HEALTHCHECK_ALERT_ID, time_period)
    is_any_event_successful = any(e[-1] == EVENT_SUCCESS_STATUS for e in events)

    if is_any_event_successful:
        _logger.info("System is healthy!")
    else:
        error_message = f"System is unhealthy: no successful events for the period {time_period}"
        _logger.error(error_message)
        notification_methods = [
            method
            for method in [
                EmailMethod(
                    recipients=HEALTHCHECK_EMAIL_RECIPIENTS,
                    subject="Market Alerts Service healthcheck",
                    body=error_message,
                ),
                TeamsWebhookMethod(
                    msg_title="<p style='color:red;'>ALARM</p>",
                    msg_text=f"<b>Details:</b> {error_message}",
                    teams_webhook=HEALTHCHECK_MS_TEAMS_WEBHOOK,
                ),
            ]
            if method.METHOD_NAME in [m.strip() for m in HEALTHCHECK_NOTIFICATION_METHODS.split(";")]
        ]
        notify(notification_methods)
