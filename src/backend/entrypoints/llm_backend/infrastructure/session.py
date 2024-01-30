import logging
import uuid
from copy import copy
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta
from enum import Enum, auto
from typing import Any, Dict, List, Optional, Tuple

import dill as pickle
import redis

from market_alerts.domain.services.steps import delete_optimization_study
from market_alerts.entrypoints.llm_backend.domain.exceptions import (
    SessionDoesNotExist,
    SessionExpiredError,
)
from market_alerts.infrastructure.mixins import SingletonMixin
from market_alerts.infrastructure.services.code import (
    are_all_lines_comments_or_empty,
    get_optimization_params_and_code_template,
)

logger = logging.getLogger(__name__)


class SessionManager(SingletonMixin):
    BASE_HASH_KEY = "sessions"

    def __init__(self, sync_redis_client: redis.client.Redis) -> None:
        self._redis_client = sync_redis_client

    def get(self, email: str, session_id: str) -> "Session":
        serialized_session = self._redis_client.hget(self._get_user_sessions_map_key(email), session_id)
        if serialized_session is not None:
            session = self._deserialize_session(serialized_session)
            if session.is_expired():
                self.delete(email, session_id)
                raise SessionExpiredError(f"User {email} tried to use session {session.session_id}, which has expired")
            return session
        raise SessionDoesNotExist(f"User {email} does not possess session with id `{session_id}`")

    def create(self, email: str, data: Optional[Dict[str, Any]] = None, expires_in: int = 3600) -> "Session":
        if data is None:
            data = {}

        session_id = uuid.uuid4().hex

        session = Session(session_id=session_id, data=data, expires_in=expires_in)
        self.save(email, session)
        return session

    def prolong(self, email: str, session_id: str) -> "Session":
        session = self.get(email, session_id)
        self.save(email, session)
        return session

    def save(self, email: str, session: "Session") -> None:
        session.prolong()
        self._redis_client.hset(self._get_user_sessions_map_key(email), session.session_id, session.to_pickle())

    def delete(self, email: str, session_id: str) -> None:
        self._redis_client.hdel(self._get_user_sessions_map_key(email), session_id)

    def delete_expired_sessions(self, optimization_storage_url: str) -> None:
        deleted_count = 0
        for key in self._redis_client.scan_iter(f"{self.BASE_HASH_KEY}:*"):
            user_sessions = self._redis_client.hgetall(key)
            for serialized_session_id, serialized_session in user_sessions.items():
                session = self._deserialize_session(serialized_session)
                if session.is_expired():
                    studies_names = session.get("optimization_studies_names", [])
                    for study_name in studies_names:
                        try:
                            delete_optimization_study(optimization_storage_url, study_name)
                        except KeyError:
                            # Study doesn't exist, perhaps it was revoked or already deleted
                            pass

                    email = self._deserialize_bytes(key).split(":")[1]
                    session_id = self._deserialize_bytes(serialized_session_id)
                    self.delete(email, session_id)
                    deleted_count += 1
        logger.info("Deleted %s expired sessions", deleted_count)

    def _get_user_sessions_map_key(self, email: str) -> str:
        return f"{self.BASE_HASH_KEY}:{email}"

    @staticmethod
    def _deserialize_session(serialized_session: bytes) -> "Session":
        return Session.from_pickle(serialized_session)

    @staticmethod
    def _deserialize_bytes(data: bytes) -> str:
        return data.decode(encoding="utf-8")


class Steps(Enum):
    FETCH_TICKERS = auto()
    SUBMIT_LLM_CHAT = auto()
    CALCULATE_INDICATORS = auto()
    PERFORM_BACKTESTING = auto()
    OPTIMIZE = auto()


_step_descriptions = {
    Steps.FETCH_TICKERS: "Tickers fetching",
    Steps.SUBMIT_LLM_CHAT: "LLM chat submission",
    Steps.CALCULATE_INDICATORS: "Indicators calculation",
    Steps.PERFORM_BACKTESTING: "Backtesting",
    Steps.OPTIMIZE: "Optimization",
}

_step_sequence = list(Steps)


@dataclass
class PipelineStatus:
    _steps: Tuple[Steps] = (Steps.FETCH_TICKERS, Steps.CALCULATE_INDICATORS)
    statuses: Dict[Steps, str] = field(init=False)

    def __post_init__(self):
        self.statuses: Dict[Steps, str] = {step: "" for step in self._steps}

    def to_dict(self) -> Dict[Steps, str]:
        data_dict = asdict(self)
        data_dict = {k: v for k, v in data_dict.items() if not k.startswith("_")}
        return data_dict


@dataclass
class FlowStatus:
    warnings: Dict[Steps, str] = field(default_factory=dict)
    errors: Dict[Steps, str] = field(default_factory=dict)
    step_fetch: Dict[str, Any] = field(default_factory=dict)
    step_submit_llm_chat: Dict[str, Any] = field(default_factory=dict)
    step_indicators: Dict[str, Any] = field(default_factory=dict)
    step_backtesting: Dict[str, Any] = field(default_factory=dict)
    step_optimization: Dict[str, Any] = field(default_factory=dict)

    is_llm_chat_history_cleared: bool = False
    is_indicators_block_present: bool = False
    is_trading_block_present: bool = False
    is_model_opened: bool = False
    opened_model_id: Optional[int] = None
    parsed_optimization_params: Dict[str, List[Any]] = field(default_factory=dict)

    _indicators_code_template: str = ""

    _pipeline_status: PipelineStatus = field(default_factory=PipelineStatus)

    _statuses: Dict[Steps, bool] = field(default_factory=lambda: {step: False for step in _step_sequence})
    _last_step_ever: Optional[Steps] = None
    _last_indicators_code: str = ""
    _last_trading_code: str = ""

    _indicators_never_executed_warning: str = f"Indicators block is present, but was never executed"

    def set_last_step(self, new_step: Steps, set_warnings: bool = True) -> None:
        if self._last_step_ever and new_step.value <= self._last_step_ever.value and set_warnings:
            for next_step in _step_sequence[new_step.value :]:
                if self.is_step_done(next_step):
                    if (
                        not (new_step == Steps.FETCH_TICKERS and next_step == Steps.SUBMIT_LLM_CHAT)
                        and not (
                            new_step == Steps.FETCH_TICKERS
                            and next_step == Steps.CALCULATE_INDICATORS
                            and not self.is_indicators_block_present
                        )
                        and not (
                            new_step == Steps.FETCH_TICKERS
                            and next_step == Steps.PERFORM_BACKTESTING
                            and not self.is_trading_block_present
                        )
                    ):
                        next_step_description = _step_descriptions[next_step]

                        if new_step == Steps.FETCH_TICKERS:
                            warning = f"Market data has been updated: re-run {next_step_description.lower()} to get new results."
                        else:
                            warning = f"{_step_descriptions[new_step]} re-executed: re-run {next_step_description.lower()} to get new results."

                        self.warnings[next_step] = warning

        if new_step in self.warnings:
            del self.warnings[new_step]

        if (
            new_step == Steps.CALCULATE_INDICATORS
            and self.warnings.get(Steps.PERFORM_BACKTESTING) == self._indicators_never_executed_warning
        ):
            del self.warnings[Steps.PERFORM_BACKTESTING]

        if new_step in self.errors:
            del self.errors[new_step]

        self._statuses[new_step] = True
        self._last_step_ever = (
            new_step if not self._last_step_ever or new_step.value > self._last_step_ever.value else self._last_step_ever
        )

    def promote_fetch_step(self, fetch_info: Optional[Dict[str, Any]] = None, fetch_error_message: Optional[str] = None) -> None:
        self.set_last_step(Steps.FETCH_TICKERS)

        if fetch_error_message is not None:
            self.warnings[Steps.FETCH_TICKERS] = fetch_error_message

        # Indicators data is cleared after every fetch step
        self._statuses[Steps.CALCULATE_INDICATORS] = False
        self.step_indicators["plots_meta"] = {}

        if fetch_info:
            self.step_fetch.update(fetch_info)

    def promote_submit_llm_chat_step(
        self, indicators_code: str, trading_code: str, info: Optional[Dict[str, Any]] = None
    ) -> None:
        if info:
            self.step_submit_llm_chat.update(info)
        self.unset_llm_chat_history_cleared()

        self.is_indicators_block_present = (
            True if indicators_code and not are_all_lines_comments_or_empty(indicators_code) else False
        )
        self.is_trading_block_present = True if trading_code and not are_all_lines_comments_or_empty(trading_code) else False

        # TODO: perhaps we need to extract parameters from trading block as well
        self.parsed_optimization_params, self._indicators_code_template = get_optimization_params_and_code_template(
            indicators_code
        )

        if self.is_indicators_block_present or self.is_trading_block_present:
            self.set_last_step(Steps.SUBMIT_LLM_CHAT, set_warnings=False)

        if self.is_indicators_block_present:
            if (
                self._last_indicators_code
                and self._last_indicators_code != indicators_code
                and self.is_step_done(Steps.CALCULATE_INDICATORS)
            ):
                self.warnings[
                    Steps.CALCULATE_INDICATORS
                ] = f"Indicators block changed: re-run {_step_descriptions[Steps.CALCULATE_INDICATORS].lower()} to get new results."
            if not self.is_step_done(Steps.CALCULATE_INDICATORS):
                self.warnings[Steps.PERFORM_BACKTESTING] = self._indicators_never_executed_warning
            self._last_indicators_code = indicators_code

        if self.is_trading_block_present:
            if (
                self._last_trading_code
                and self._last_trading_code != trading_code
                and self.is_step_done(Steps.PERFORM_BACKTESTING)
            ):
                self.warnings[
                    Steps.PERFORM_BACKTESTING
                ] = f"Trading block changed: re-run {_step_descriptions[Steps.PERFORM_BACKTESTING].lower()} to get new results."
            self._last_trading_code = trading_code

    @property
    def trading_code(self) -> str:
        return self._last_trading_code

    @property
    def indicators_code(self) -> str:
        return self._last_indicators_code

    @property
    def indicators_code_template(self) -> str:
        return self._indicators_code_template

    def get_interpolated_indicators_code_template(self, value_by_param: Dict[str, Any]) -> str:
        return self._indicators_code_template % self._get_indicators_interpolation_values(value_by_param)

    def _get_indicators_interpolation_values(self, value_by_param: Dict[str, Any]) -> tuple:
        return tuple(
            self._get_indicator_interpolation_value(value_by_param, param_name, param_type)
            for param_name, (_, param_type) in self.parsed_optimization_params.items()
        )

    @staticmethod
    def _get_indicator_interpolation_value(value_by_param: dict[str, Any], param_name: str, param_type: str) -> Any:
        return value_by_param[param_name] if param_type != "str" else f"'{value_by_param[param_name]}'"

    def promote_indicators_step(self, indicators_info: Optional[Dict[str, Any]] = None) -> None:
        self.set_last_step(Steps.CALCULATE_INDICATORS)
        if indicators_info:
            self.step_indicators.update(indicators_info)

    def promote_backtesting_step(self, backtesting_info: Optional[Dict[str, Any]] = None) -> None:
        self.set_last_step(Steps.PERFORM_BACKTESTING)
        if backtesting_info:
            self.step_backtesting.update(backtesting_info)

    def promote_optimization_step(self, optimization_info: Optional[Dict[str, Any]] = None) -> None:
        self.set_last_step(Steps.OPTIMIZE)
        if optimization_info:
            self.step_optimization.update(optimization_info)

    def add_error_for_step(self, step: Steps, error: str) -> None:
        self.errors[step] = error

    def is_step_done(self, step: Steps) -> bool:
        return self._statuses.get(step, False)

    def set_model_opened(self, model_id: int) -> None:
        self.is_model_opened = True
        self.opened_model_id = model_id

    def unset_model_opened(self) -> None:
        self.is_model_opened = False
        self.opened_model_id = None

    def get_pipeline_status(self) -> Dict[Steps, str]:
        return self._pipeline_status.to_dict()

    def reset_pipeline_status(self) -> None:
        self._pipeline_status = PipelineStatus()

    def pipeline_start_step(self, step: Steps) -> None:
        self._pipeline_status.statuses[step] = "STARTED"

    def pipeline_finish_step(self, step: Steps) -> None:
        self._pipeline_status.statuses[step] = "FINISHED"

    def set_llm_chat_history_cleared(self) -> None:
        self.is_llm_chat_history_cleared = True

    def unset_llm_chat_history_cleared(self) -> None:
        self.is_llm_chat_history_cleared = False

    def to_dict(self):
        data_dict = asdict(self)
        data_dict = {k: v for k, v in data_dict.items() if not k.startswith("_")}
        data_dict["step_fetch"]["done"] = self._statuses.get(Steps.FETCH_TICKERS, False)
        data_dict["step_submit_llm_chat"]["done"] = self._statuses.get(Steps.SUBMIT_LLM_CHAT, False)
        data_dict["step_indicators"]["done"] = self._statuses.get(Steps.CALCULATE_INDICATORS, False)
        data_dict["step_backtesting"]["done"] = self._statuses.get(Steps.PERFORM_BACKTESTING, False)
        data_dict["step_optimization"]["done"] = self._statuses.get(Steps.OPTIMIZE, False)

        for key, value in data_dict.items():
            if isinstance(value, dict):
                formatted_dict = {}
                for subkey, subvalue in value.items():
                    if isinstance(subkey, Enum):
                        subkey = subkey.value
                    formatted_dict[subkey] = subvalue
                data_dict[key] = formatted_dict
            elif isinstance(value, Enum):
                data_dict[key] = value.value

        return data_dict


@dataclass
class Session:
    session_id: str
    flow_status: FlowStatus = field(default_factory=FlowStatus)
    data: Dict[str, Any] = field(default_factory=dict)
    actions_history: List[Tuple[str, float]] = field(default_factory=list)
    expires_at: Optional[datetime] = None
    expires_in: int = 3600

    def __post_init__(self):
        if self.expires_at is None:
            self.prolong()

    def __getitem__(self, key):
        return self.data[key]

    def __setitem__(self, key, value):
        self.data[key] = value

    def __contains__(self, key):
        return key in self.data

    def get(self, key, default_value=None):
        return self.data.get(key, default_value)

    def setdefault(self, key, default_value=None):
        if key not in self.data:
            self.data[key] = default_value
        return self.data[key]

    def update(self, data: Optional[Dict[str, Any]] = None, **kwargs) -> None:
        if data is not None:
            self.data.update(data)
        self.data.update(kwargs)

    @classmethod
    def from_pickle(cls, value: bytes) -> "Session":
        return pickle.loads(value)

    def to_pickle(self) -> bytes:
        return pickle.dumps(self)

    def is_expired(self) -> bool:
        return datetime.now() > self.expires_at

    def prolong(self) -> None:
        self.expires_at = datetime.now() + timedelta(seconds=self.expires_in)

    def reset_flow_status(self) -> None:
        self.flow_status = FlowStatus()

    def add_action(self, task_id: str, timestamp: float):
        self.actions_history.append((task_id, timestamp))

    def get_slice(self, start: int, end: int) -> "Session":
        required_keys = ["time_line", "u_strs", "data_by_symbol"]
        if not all(key in self for key in required_keys):
            raise RuntimeError("Couldn't create session slice: " + ", ".join(required_keys) + " must be present")

        session_slice = copy(self)
        session_slice.data = copy(session_slice.data)
        session_slice["u_strs"] = session_slice["u_strs"].copy()
        session_slice["time_line"] = session_slice["time_line"][start:end]
        session_slice["data_by_symbol"] = {key: value[start:end] for key, value in session_slice["data_by_symbol"].items()}
        if "fx_rates" in session_slice:
            session_slice.fx_rates = copy(session_slice["fx_rates"])
            session_slice["fx_rates"] = {key: value[start:end] for key, value in session_slice["fx_rates"].items()}
        if "dividends_by_symbol" in session_slice:
            session_slice.dividends_by_symbol = copy(session_slice["dividends_by_symbol"])
            session_slice["dividends_by_symbol"] = {
                key: value[start:end] for key, value in session_slice["dividends_by_symbol"].items()
            }
        session_slice["start_date"] = str(session_slice["time_line"][0])
        session_slice["end_date"] = str(session_slice["time_line"][-1])
        return session_slice
