import logging

try:
    import boto3
    from botocore.exceptions import ClientError
except ImportError:
    pass

from market_alerts.config import SES_REGION, SES_SENDER_EMAIL

from .base import NotificationMethod

_logger = logging.getLogger(__name__)


class EmailMethod(NotificationMethod):
    METHOD_NAME = "email"

    def __init__(self, recipients: str, subject: str, body: str):
        self.recipients = [r.strip() for r in recipients.split(";")]
        self.charset = "UTF-8"
        self.subject = subject
        # The email body for recipients with non-HTML email clients.
        self.body_text = body
        self.body_html = f"""<html>
<head></head>
<body>
  <h1>{self.subject}</h1>
  <p>{body}</p>
</body>
</html>
"""

    def execute(self) -> None:
        client = boto3.client("ses", region_name=SES_REGION)
        try:
            response = client.send_email(
                Destination={
                    "ToAddresses": self.recipients,
                },
                Message={
                    "Body": {
                        "Html": {
                            "Charset": self.charset,
                            "Data": self.body_html,
                        },
                        "Text": {
                            "Charset": self.charset,
                            "Data": self.body_text,
                        },
                    },
                    "Subject": {
                        "Charset": self.charset,
                        "Data": self.subject,
                    },
                },
                Source=SES_SENDER_EMAIL,
            )
        except ClientError as e:
            raise e
        else:
            _logger.info(f"Email sent! Message ID: {response['MessageId']}")
