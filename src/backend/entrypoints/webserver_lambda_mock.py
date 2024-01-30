import logging

from flask import Flask, jsonify, request
from lambdas.lambda_run import lambda_handler

app = Flask(__name__)

_logger = logging.getLogger(__name__)


@app.route("/mocked_lambda", methods=["POST"])
def mocked_lambda():
    if not request.is_json:
        return jsonify({"error": "Data is not json"}), 400

    try:
        data = request.get_json()
        status, message = lambda_handler(data, {})
        return (
            jsonify(
                {
                    "status": status,
                    "message": message,
                }
            ),
            200,
        )
    except Exception as e:
        _logger.error(e)
        return jsonify({"error": "Internal error", "details": str(e)}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
