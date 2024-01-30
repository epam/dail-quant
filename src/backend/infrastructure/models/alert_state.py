from sqlalchemy import BigInteger, Column, DateTime, SmallInteger, Table

from market_alerts.infrastructure.datasource import metadata

alert_state_table = Table(
    "alert_state",
    metadata,
    Column("id", BigInteger, primary_key=True),
    Column("last_event_status", SmallInteger),
    Column("last_event_time", DateTime),
)
