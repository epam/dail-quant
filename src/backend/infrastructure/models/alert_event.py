from sqlalchemy import (
    BigInteger,
    Column,
    DateTime,
    ForeignKey,
    Index,
    SmallInteger,
    Table,
    Text,
)

from market_alerts.infrastructure.datasource import metadata

alert_event_table = Table(
    "alert_event",
    metadata,
    Column("id", BigInteger, primary_key=True, autoincrement=True),
    Column("alert_id", BigInteger, ForeignKey("alert.id"), nullable=False),
    Column("timestamp", DateTime, nullable=False),
    Column("title", Text),
    Column("address", Text),
    Column("message", Text),
    Column("status", SmallInteger, nullable=False),
    Index("index_alert_id", "alert_id"),
)
