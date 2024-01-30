import logging
from typing import List

from market_alerts.domain.services.notifier.methods.base import NotificationMethod

_logger = logging.getLogger(__name__)


def notify(methods: List[NotificationMethod]) -> None:
    for method in methods:
        try:
            method.execute()
        except Exception as e:
            _logger.warning(f"'{method.METHOD_NAME}' notification method failed. Reason: {e}")
