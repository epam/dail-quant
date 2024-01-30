import re
import time
from typing import Any, Dict, Optional

import requests
from requests import RequestException

from market_alerts.domain.exceptions import JupyterError
from market_alerts.infrastructure.mixins import SingletonMixin

from .artifact_builders import JupyterNotebookModelExportArtifactBuilder


class JupyterService(SingletonMixin):
    BASE_NOTEBOOK_FILENAME = "trading_strategy_"

    def __init__(self, ui_url: str, service_url: str, token: str) -> None:
        self._artifact_builder = JupyterNotebookModelExportArtifactBuilder()
        self._ui_url = ui_url
        self._service_url = service_url
        self._api_url = f"{service_url}/hub/api"
        self._token = token

    def create_user(self, username: str) -> None:
        user_url = self._get_user_url(username)
        try:
            response = requests.get(user_url, headers=self._get_headers())
            if not response.status_code == 200:
                requests.post(user_url, headers=self._get_headers())
        except RequestException as e:
            raise JupyterError(str(e))

    def start_user_server(self, username: str) -> None:
        server_url = f"{self._get_user_url(username)}/server"
        try:
            response = requests.post(server_url, headers=self._get_headers())
            # 400 if server is already running
            if response.status_code not in (201, 202, 400):
                raise JupyterError(f"Error starting server for user {username}: {response.status_code}")
        except RequestException as e:
            raise JupyterError(f"Error starting server for user {username}: {e}")

    def wait_until_user_server_ready(self, username: str, timeout: Optional[int] = None) -> None:
        start = time.time()
        while not self._is_user_server_ready(username):
            time.sleep(1)
            if timeout is not None and time.time() > start + timeout:
                raise TimeoutError(f"User's jupyterhub server was not available after timeout of {timeout} seconds")

    def _is_user_server_ready(self, username: str) -> bool:
        user_url = self._get_user_url(username)
        try:
            response_json = requests.get(user_url, headers=self._get_headers()).json()
            return response_json["servers"][""]["ready"]
        except RequestException as e:
            raise JupyterError(str(e))

    def put_user_notebook(self, username: str, data: Dict[str, Any], new_notebook_filename: Optional[str] = None) -> str:
        if new_notebook_filename is not None and new_notebook_filename:
            new_notebook_filename = f"{new_notebook_filename}.ipynb"
        else:
            new_notebook_filename = self._get_new_user_notebook_filename(username)
        notebook_put_url = self._get_contents_api_url(username, new_notebook_filename)
        try:
            notebook_json = self._artifact_builder.build_artifact(data)
            response = requests.put(notebook_put_url, json=notebook_json, headers=self._get_headers())
            if response.status_code not in (200, 201):
                raise JupyterError(f"Error creating notebook for user {username}: {response.status_code}")
        except RequestException as e:
            raise JupyterError(f"Error creating notebook for user {username}: {e}")
        return self._get_notebook_redirect_url(username, new_notebook_filename)

    def _get_new_user_notebook_filename(self, username: str) -> str:
        contents_api_url = self._get_contents_api_url(username)

        try:
            response = requests.get(contents_api_url, headers=self._get_headers())
        except RequestException as e:
            raise JupyterError(f"Error creating a unique filename for user's {username} notebook: {e}")

        response_json = response.json()

        name_pattern = rf"{self.BASE_NOTEBOOK_FILENAME}(\d+)\.ipynb"

        trading_rule_files = (entry["name"] for entry in response_json["content"] if re.match(name_pattern, entry["name"]))

        max_number = 0
        for name in trading_rule_files:
            match = re.match(name_pattern, name)
            number = int(match.group(1))
            max_number = max(max_number, number)

        new_filename = rf"{self.BASE_NOTEBOOK_FILENAME}{max_number + 1}.ipynb"

        return new_filename

    def _get_user_url(self, username: str) -> str:
        return f"{self._api_url}/users/{username}"

    def _get_notebook_redirect_url(self, username: str, notebook_filename: str) -> str:
        return f"{self._ui_url}/user/{username}/lab/tree/{notebook_filename}"

    def _get_contents_api_url(self, username: str, notebook_filename: str = "") -> str:
        return f"{self._service_url}/user/{username}/api/contents/{notebook_filename}"

    def _get_headers(self) -> Dict[str, str]:
        return {"Authorization": f"token {self._token}"}
