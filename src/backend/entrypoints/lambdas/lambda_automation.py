import logging
import os

import boto3

logging.basicConfig(
    level=os.environ.get("LOG_LEVEL", "INFO").upper(),
    format="[%(asctime)s] [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %I:%M:%S",
    force=True,
)

_logger = logging.getLogger(__name__)

S3_BUCKET_NAME = os.environ.get("S3_BUCKET_NAME")

MAIN_FUNCTION_NAME = os.environ.get("MAIN_FUNCTION_NAME")
MAIN_ZIP_ARCHIVE_NAME = os.environ.get("MAIN_ZIP_ARCHIVE_NAME")

HEALTHCHECK_FUNCTION_NAME = os.environ.get("HEALTHCHECK_FUNCTION_NAME")

s3_client = boto3.client("s3")
lambda_client = boto3.client("lambda")


def lambda_handler(event, context):
    record = event["Records"][0]
    bucket: str = record["s3"]["bucket"]["name"]
    key: str = record["s3"]["object"]["key"]

    if bucket == S3_BUCKET_NAME and key.startswith("public/"):
        function_name = MAIN_FUNCTION_NAME if MAIN_ZIP_ARCHIVE_NAME in key else HEALTHCHECK_FUNCTION_NAME
        params = {
            "FunctionName": function_name,
            "S3Key": key,
            "S3Bucket": bucket,
        }

        try:
            response = lambda_client.update_function_code(**params)
            _logger.info(f"Function code updated successfully: {response}")
        except Exception as e:
            _logger.error(f"Error updating function code: {str(e)}")
