import json
import logging

import requests

from .base import NotificationMethod

_logger = logging.getLogger(__name__)


class TeamsWebhookMethod(NotificationMethod):
    METHOD_NAME = "teams"

    def __init__(self, msg_title: str, msg_text: str, teams_webhook: str) -> None:
        self.message_title = msg_title
        self.message_text = msg_text
        self.teams_webhook = teams_webhook

    def execute(self) -> None:
        msg = {"title": self.message_title, "text": self.message_text}
        encoded_msg = json.dumps(msg).encode("utf-8")
        response = requests.post(self.teams_webhook, data=encoded_msg)
        response.raise_for_status()
        _logger.info("MS Teams notification sent!")
