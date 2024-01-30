import datetime

from sqlalchemy import select
from sqlalchemy.engine import Connection

from market_alerts.infrastructure.datasource import Database
from market_alerts.infrastructure.models import alert_event_table


class AlertEventRepository:
    def __init__(self, database: Database):
        self._database = database

    def get_events_of_alert_for_period(self, alert_id: int, passed_time: datetime.timedelta):
        time_ago = datetime.datetime.utcnow() - passed_time

        query = (
            select(
                alert_event_table.c.id,
                alert_event_table.c.alert_id,
                alert_event_table.c.timestamp,
                alert_event_table.c.title,
                alert_event_table.c.address,
                alert_event_table.c.message,
                alert_event_table.c.status,
            )
            .where(alert_event_table.c.timestamp >= time_ago)
            .where(alert_event_table.c.alert_id == alert_id)
        )

        return self._connection.execute(query).fetchall()

    @property
    def _connection(self) -> Connection:
        return self._database.get_connection()
