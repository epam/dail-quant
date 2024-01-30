from sqlalchemy import MetaData, create_engine
from sqlalchemy.engine.url import URL
from sqlalchemy.pool import QueuePool

from market_alerts.config import (
    DB_DATABASE,
    DB_DIALECT,
    DB_DRIVER,
    DB_HOST,
    DB_PASSWORD,
    DB_PORT,
    DB_USER,
)

metadata = MetaData()


class Database:
    def __init__(
        self,
        username: str,
        password: str,
        host: str,
        port: int,
        database: str,
        dialect: str,
        driver: str,
    ) -> None:
        self.username = username
        self.password = password
        self.host = host
        self.port = port
        self.database = database
        self.dialect = dialect
        self.driver = driver

    def connect(self) -> None:
        self.engine = create_engine(
            URL(
                username=self.username,
                password=self.password,
                host=self.host,
                port=self.port,
                database=self.database,
                drivername=f"{self.dialect}+{self.driver}",
                query={},
            ),
            poolclass=QueuePool,
            pool_size=5,
            max_overflow=10,
            pool_timeout=30,
        )

    def get_connection(self):
        return self.engine.connect()


database = Database(DB_USER, DB_PASSWORD, DB_HOST, DB_PORT, DB_DATABASE, DB_DIALECT, DB_DRIVER)
