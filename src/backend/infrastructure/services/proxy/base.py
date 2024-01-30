from json import JSONDecodeError
from typing import Any

from market_alerts.domain.exceptions import ServiceProxyError
from market_alerts.infrastructure.mixins import AuthDriverMixin


class Proxy(AuthDriverMixin):
    SERVICE_NAME = "Proxy"

    def __init__(self, service_url: str) -> None:
        self._service_url = service_url
        super().__init__()

    def _call_api(self, url: str, method: str = "GET", **kwargs) -> Any:
        response = self.session.request(method=method, url=url, **kwargs)
        if not response.ok:
            try:
                message = response.json().get("message", "")
            except JSONDecodeError:
                message = response.text
            finally:
                raise ServiceProxyError(
                    status_code=response.status_code,
                    message=message,
                    service_name=self.SERVICE_NAME,
                )

        return response.json()
