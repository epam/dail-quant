import logging

from market_alerts.entrypoints.llm_backend.config import Settings

logging.basicConfig(
    level=Settings().logger_level.upper(),
    format="[%(asctime)s] [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %I:%M:%S",
)
for logger_name in ["uvicorn.error", "uvicorn.protocols.websockets", "uvicorn.protocols.websocket"]:
    logger = logging.getLogger(logger_name)
    logger.setLevel(Settings().logger_level.upper())
