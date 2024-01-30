import logging
from typing import Any, Dict, List, Optional, Tuple

from strenum import StrEnum

from market_alerts.domain.exceptions import ServiceProxyError
from market_alerts.infrastructure.mixins import SingletonMixin

from ..base import Proxy
from .exceptions import (
    CreateTradingRuleError,
    GetTradingRuleError,
    LimitsDisabled,
    LimitsExceededError,
    LimitsInfoFetchError,
    SendUsedTokensInfoError,
    ShareTradingRuleError,
    TimezonesFetchError,
    UpdateTradingRuleError,
    UserRolesFetchError,
)

logger = logging.getLogger(__name__)


class UserLimitsAggregationRanges(StrEnum):
    DAY = "DAY"
    MONTH = "MONTH"


class AlertsBackendProxy(Proxy, SingletonMixin):
    SERVICE_NAME = "AlertsBackendProxy"

    def __init__(self, service_url: str, limits_enabled: bool = True):
        super().__init__(f"{service_url}/api/v0")
        self._limits_enabled = limits_enabled

    def check_token_user_limits(self):
        try:
            used_resources = self.get_used_limits_amount()
            limits = self.get_user_limits()
        except LimitsDisabled:
            return

        for aggregation in UserLimitsAggregationRanges:
            if used_resources[aggregation]["tokens"] >= limits[aggregation]["tokens"]:
                raise LimitsExceededError(
                    f"Consumed tokens ({used_resources[aggregation]['tokens']}) exceeded the limit of ({limits[aggregation]['tokens']}) for user {self.email} for following time span: {aggregation}"
                )

    def check_cpu_user_limits(self):
        try:
            used_resources = self.get_used_limits_amount()
            limits = self.get_user_limits()
        except LimitsDisabled:
            return

        for aggregation in UserLimitsAggregationRanges:
            if used_resources[aggregation]["cpu_usage"] >= limits[aggregation]["cpu_usage"]:
                raise LimitsExceededError(
                    f"Consumed cpu time ({used_resources[aggregation]['cpu_usage']}) exceeded the limit of ({limits[aggregation]['cpu_usage']}) for user {self.email} for following time span: {aggregation}"
                )

    def get_used_limits_amount(self) -> Tuple[int, int]:
        if not self._limits_enabled:
            logger.warning("Tried getting used resources amount, but they were disabled")
            raise LimitsDisabled
        try:
            response_json_daily = self._call_api(
                f"{self._service_url}/user-info/limits/spent?aggregation={UserLimitsAggregationRanges.DAY}"
            )
            response_json_monthly = self._call_api(
                f"{self._service_url}/user-info/limits/spent?aggregation={UserLimitsAggregationRanges.MONTH}"
            )
        except ServiceProxyError as e:
            raise LimitsInfoFetchError(f"Error while fetching limits amount: {e}")
        return {
            UserLimitsAggregationRanges.DAY: {
                "tokens": response_json_daily["inbound_tokens"] + response_json_daily["outbound_tokens"],
                "cpu_usage": response_json_daily["cpu_usage"],
            },
            UserLimitsAggregationRanges.MONTH: {
                "tokens": response_json_monthly["inbound_tokens"] + response_json_monthly["outbound_tokens"],
                "cpu_usage": response_json_monthly["cpu_usage"],
            },
        }

    def get_user_limits(self) -> Dict[str, Any]:
        if not self._limits_enabled:
            logger.warning("Tried getting used resources amount, but they were disabled")
            raise LimitsDisabled
        try:
            response_json = self._call_api(f"{self._service_url}/user-info/limits")
        except ServiceProxyError as e:
            raise LimitsInfoFetchError(f"Error while fetching limits: {e}")

        return {
            UserLimitsAggregationRanges.DAY: {"tokens": response_json["tokens"], "cpu_usage": response_json["cpu_usage"]},
            UserLimitsAggregationRanges.MONTH: {
                "tokens": response_json["tokens_monthly"],
                "cpu_usage": response_json["cpu_usage_monthly"],
            },
        }

    def get_user_roles(self) -> List[str]:
        try:
            return self._call_api(f"{self._service_url}/user-info/roles")
        except ServiceProxyError as e:
            raise UserRolesFetchError(f"Error while fetching user roles: {e}")

    def get_available_timezones(self) -> List[str]:
        try:
            return self._call_api(f"{self._service_url}/info/zones")
        except ServiceProxyError as e:
            raise TimezonesFetchError(f"Error while fetching timezones: {e}")

    def send_used_resources_info(self, inbound_tokens: int, outbound_tokens: int, cpu_usage: float) -> None:
        if not self._limits_enabled:
            raise LimitsDisabled

        used_tokens_info = {
            "user_id": self.email,
            "inbound_tokens": inbound_tokens,
            "outbound_tokens": outbound_tokens,
            "cpu_usage": cpu_usage,
        }
        try:
            self._call_api(f"{self._service_url}/user-info/limits/spent", method="PUT", json=used_tokens_info)
            logger.info("Sent limits to backend: %s", used_tokens_info)
        except ServiceProxyError as e:
            raise SendUsedTokensInfoError(f"Error while sending used tokens info: {e}")

    def get_trading_rule_by_id(self, rule_id: int) -> Dict[str, Any]:
        try:
            return self.get_public_trading_rule_by_id(rule_id)
        except GetTradingRuleError:
            return self.get_personal_trading_rule_by_id(rule_id)

    def get_personal_trading_rule_by_id(self, rule_id: int) -> Dict[str, Any]:
        try:
            response_json = self._call_api(f"{self._service_url}/trading-info/{rule_id}")
        except ServiceProxyError as e:
            raise GetTradingRuleError(f"Error while fetching personal trading rule with id {rule_id}: {e}")

        return response_json

    def get_public_trading_rule_by_id(self, rule_id: int) -> Dict[str, Any]:
        try:
            response_json = self._call_api(f"{self._service_url}/trading-info/public/{rule_id}")
        except ServiceProxyError as e:
            raise GetTradingRuleError(f"Error while fetching public trading rule with id {rule_id}: {e}")

        return response_json

    def get_trading_rule_by_title_from_all(self, title: str) -> Optional[Dict[str, Any]]:
        try:
            personal_rules = self._call_api(f"{self._service_url}/trading-info")
            public_rules = self._call_api(f"{self._service_url}/trading-info/public")
        except ServiceProxyError as e:
            raise GetTradingRuleError(f"Error while fetching trading rule with title {title}: {e}")

        return next(filter(lambda tr: tr["title"] == title, public_rules + personal_rules), None)

    def get_trading_rule_by_title_from_personal(self, title: str) -> Optional[Dict[str, Any]]:
        try:
            personal_rules = self._call_api(f"{self._service_url}/trading-info")
        except ServiceProxyError as e:
            raise GetTradingRuleError(f"Error while fetching trading rule with title {title}: {e}")

        return next(filter(lambda tr: tr["title"] == title, personal_rules), None)

    def get_trading_rule_by_title_from_public(self, title: str) -> Optional[Dict[str, Any]]:
        try:
            public_rules = self._call_api(f"{self._service_url}/trading-info/public")
        except ServiceProxyError as e:
            raise GetTradingRuleError(f"Error while fetching trading rule with title {title}: {e}")

        return next(filter(lambda tr: tr["title"] == title, public_rules), None)

    def get_public_trading_rules(self) -> List[Dict[str, Any]]:
        try:
            return self._call_api(f"{self._service_url}/trading-info/public")
        except ServiceProxyError as e:
            raise GetTradingRuleError(f"Error while fetching public trading rules: {e}")

    def create_trading_rule(self, trading_rule: Dict[str, Any]) -> Dict[str, Any]:
        try:
            return self._call_api(f"{self._service_url}/trading-info", method="POST", json=trading_rule)
        except ServiceProxyError as e:
            raise CreateTradingRuleError(f"Error when creating trading rule: {e}")

    def update_trading_rule(self, trading_rule: Dict[str, Any]) -> Dict[str, Any]:
        try:
            return self._call_api(f"{self._service_url}/trading-info", method="PUT", json=trading_rule)
        except ServiceProxyError as e:
            raise UpdateTradingRuleError(f"Error when updating trading rule: {e}")

    def share_trading_rule(self, trading_rule: Dict[str, Any], recipient: str) -> Dict[str, Any]:
        try:
            return self._call_api(
                f"{self._service_url}/trading-info/share?destination={recipient}", method="POST", json=trading_rule
            )
        except ServiceProxyError as e:
            raise ShareTradingRuleError(f"Error when sharing trading rule: {e}")

    def get_prompts(self, prompt_ids: List[int]) -> List[Dict[str, Any]]:
        user_prompts = self._call_api(f"{self._service_url}/prompt", method="GET")
        public_prompts = self._call_api(f"{self._service_url}/prompt/public", method="GET")
        user_prompts.extend(public_prompts)
        user_prompts = {prompt["id"]: prompt for prompt in user_prompts}

        return [user_prompts[prompt_id] for prompt_id in prompt_ids]
