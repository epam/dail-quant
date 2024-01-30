import json
import time
from collections import ChainMap
from typing import Any, Dict, List, Optional

import dill as pickle
from celery import Task, current_app
from celery.exceptions import SoftTimeLimitExceeded
from celery.utils.log import get_task_logger
from optuna import Study
from optuna.study import StudyDirection
from optuna.trial import FrozenTrial
from redis import RedisError

from market_alerts.containers import (
    alerts_backend_proxy_singleton,
    optimization_samplers,
    optimization_target_funcs,
)
from market_alerts.domain.constants import ADDITIONAL_COLUMNS, PUBSUB_END_OF_DATA
from market_alerts.domain.exceptions import DataNotFoundError, LLMBadResponseError
from market_alerts.domain.services import (
    define_empty_indicators_step,
    define_useful_strings,
    get_actual_currency_fx_rates,
    get_combined_trading_statistics,
    get_sparse_dividends_for_each_tradable_symbol,
    indicator_chat,
    indicator_step,
    optimize,
    symbol_step,
    trading_step,
)
from market_alerts.domain.services.charts import get_lines_per_symbol_mapping
from market_alerts.domain.services.steps import (
    delete_optimization_study,
    get_optimization_results,
)
from market_alerts.entrypoints.llm_backend import middleware
from market_alerts.entrypoints.llm_backend.api.models.common import (
    FlowStatusResponseModel,
)
from market_alerts.entrypoints.llm_backend.api.models.llm import LLMResponseModel
from market_alerts.entrypoints.llm_backend.api.models.optimization import (
    OptimizationResponseModel,
    OptimizationResult,
)
from market_alerts.entrypoints.llm_backend.api.models.tickers import (
    TickersFetchInfoResponseModel,
)
from market_alerts.entrypoints.llm_backend.api.tasks import get_task_error_msg
from market_alerts.entrypoints.llm_backend.containers import (
    get_optimization_storage,
    session_manager_singleton,
    settings,
    sync_redis_client,
)
from market_alerts.entrypoints.llm_backend.infrastructure.access_management.context_vars import (
    user,
)
from market_alerts.entrypoints.llm_backend.infrastructure.services.ws import (
    OptimizationTaskWSMetaInfo,
    get_task_ws_meta_info,
)
from market_alerts.entrypoints.llm_backend.infrastructure.session import Session, Steps
from market_alerts.infrastructure.services.code import get_code_sections
from market_alerts.infrastructure.services.proxy.alerts_backend.exceptions import (
    LimitsDisabled,
)
from market_alerts.utils import convert_date, progress_and_time_generator, time_profile

logger = get_task_logger(__name__)


@current_app.task(name="fetch_tickers", bind=True)
def fetch_tickers(
    self,
    data_provider: str,
    datasets: List[str],
    periodicity: int,
    tradable_symbols_prompt: str,
    supplementary_symbols_prompt: str,
    economic_indicators: List[str],
    dividend_fields: List[str],
    time_range: int,
    is_chained: bool = False,
) -> Dict[str, Any]:
    current_user = user.get()

    session = session_manager_singleton.get(current_user.email, self.session_id)

    if is_chained:
        session.flow_status.pipeline_start_step(Steps.FETCH_TICKERS)
        session_manager_singleton.save(current_user.email, session)

    if (
        len(session["datasets_keys"]) != len(datasets)
        or data_provider != session["data_provider"]
        or len(dividend_fields) != len(session["dividend_fields"])
        or any([x != y for x, y in zip(session["datasets_keys"], datasets)])
    ):
        session["data_by_symbol"] = {}
        session["meta"] = {}

    session["data_provider"] = data_provider
    session["datasets_keys"] = datasets
    session["interval"] = periodicity
    session["tradable_symbols_prompt"] = tradable_symbols_prompt
    session["supplementary_symbols_prompt"] = supplementary_symbols_prompt
    session["economic_indicators"] = economic_indicators
    session["dividend_fields"] = dividend_fields
    session["time_period"] = time_range

    task_id = self.request.id
    try:
        progress_callback = lambda _: _safe_progress_publish(
            pubsub=sync_redis_client,
            channel=f"task-{task_id}-progress",
            message=json.dumps({}),
            error_flag=False,
        )
        (request_timestamp, fetched_symbols_meta, synth_formulas, error_message), execution_time = time_profile(
            symbol_step, session, progress_callback
        )
        payload = {"progress": PUBSUB_END_OF_DATA}
        _safe_progress_publish(
            pubsub=sync_redis_client,
            channel=f"task-{task_id}-progress",
            message=json.dumps(payload),
            error_flag=False,
        )

        define_useful_strings(session)
        define_empty_indicators_step(session)

        plots_meta = _build_prices_plots_meta(session)

        session.flow_status.promote_fetch_step(
            TickersFetchInfoResponseModel(
                data_provider=data_provider,
                datasets=datasets,
                periodicity=periodicity,
                time_range=time_range,
                fetched_symbols_meta=fetched_symbols_meta,
                plots_meta=plots_meta,
                synth_formulas=synth_formulas,
                request_timestamp=request_timestamp,
                execution_time=execution_time,
            ).model_dump(),
            error_message,
        )
        if is_chained:
            session.flow_status.pipeline_finish_step(Steps.FETCH_TICKERS)
        session_manager_singleton.save(current_user.email, session)

        return FlowStatusResponseModel(
            flow_status=session.flow_status.to_dict(),
        ).model_dump()
    # TODO: handle only specific exceptions in prod
    except (SoftTimeLimitExceeded, DataNotFoundError, Exception) as e:
        _handle_error(
            session_id=session.session_id,
            email=current_user.email,
            step=Steps.FETCH_TICKERS,
            exc_to_raise=e,
        )


def _build_prices_plots_meta(session) -> dict[str, dict[str, dict[str, list[list[dict[str, str]]]]]]:
    symbols_meta = {}
    plots_meta = {
        "symbols": symbols_meta,
        "start_date": convert_date(session["start_date"]),
        "end_date": convert_date(session["end_date"]),
    }

    lines_per_symbol = get_lines_per_symbol_mapping(
        {
            **session["data_by_symbol"],
            **session.get("data_by_synth", {}),
        }
    )

    for symbol, lines in lines_per_symbol.items():
        symbols_meta[symbol] = {"charts": []}
        for line in lines:
            symbols_meta[symbol]["charts"].append([{"name": line, "type": _determine_prices_plot_line_type(line)}])
        symbols_meta[symbol]["type"] = _determine_plot_type(symbol, session)

    return plots_meta


def _determine_prices_plot_line_type(line: str) -> str:
    if line in ADDITIONAL_COLUMNS:
        return "dividend"
    return "price"


def _determine_plot_type(symbol: str, session) -> str:
    if symbol in session["tradable_symbols"] or symbol in session["synth_formulas_to_trade"]:
        return "tradable"
    if symbol in session["supplementary_symbols"] or symbol in session["synth_formulas_not_to_trade"]:
        return "supplementary"
    elif symbol in session["economic_indicator_symbols"]:
        return "economic_indicator"
    raise RuntimeError(f"unexpected symbol: {symbol}")


@current_app.task(name="submit_llm_chat", bind=True)
def submit_llm_chat(
    self,
    llm_query: str,
    user_prompt_ids: List[int],
    engine: str,
) -> Dict[str, Any]:
    current_user = user.get()

    session = session_manager_singleton.get(current_user.email, self.session_id)

    session.setdefault("indicators_dialogue", []).append(llm_query)

    try:
        (llm_response, token_usage, request_timestamp), execution_time = time_profile(
            indicator_chat, session, user_prompt_ids, engine
        )

        try:
            alerts_backend_proxy_singleton.send_used_resources_info(
                token_usage["prompt_tokens"], token_usage["completion_tokens"], 0
            )
        except LimitsDisabled:
            logger.warning("Tried sending used resources info, but limits were disabled")

        indicators_code, trading_code = get_code_sections(llm_response)

        session.flow_status.promote_submit_llm_chat_step(
            indicators_code,
            trading_code,
            LLMResponseModel(
                flow_status=session.flow_status.to_dict(),
                llm_response=llm_response,
                engine=engine,
                token_usage=token_usage,
                request_timestamp=request_timestamp,
                execution_time=execution_time,
            ).model_dump(exclude={"flow_status"}),
        )
        session_manager_singleton.save(current_user.email, session)

        return FlowStatusResponseModel(
            flow_status=session.flow_status.to_dict(),
        ).model_dump()
    # TODO: handle only specific exceptions in prod
    except (SoftTimeLimitExceeded, Exception) as e:
        _handle_error(
            session_id=session.session_id,
            email=current_user.email,
            step=Steps.SUBMIT_LLM_CHAT,
            exc_to_raise=e,
        )


@current_app.task(name="calculate_indicators", bind=True)
def calculate_indicators(
    self,
    is_chained: bool = False,
) -> Dict[str, Any]:
    current_user = user.get()

    session = session_manager_singleton.get(current_user.email, self.session_id)

    if is_chained:
        session.flow_status.pipeline_start_step(Steps.CALCULATE_INDICATORS)
        session_manager_singleton.save(current_user.email, session)

    try:
        _, execution_time = time_profile(indicator_step, session)

        plots_meta = _build_indicators_plots_meta(session)

        session.flow_status.promote_indicators_step(
            {
                "execution_time": execution_time,
                "logs_present": True if session["indicators_code_log"] else False,
                "plots_meta": plots_meta,
            }
        )

        try:
            alerts_backend_proxy_singleton.send_used_resources_info(0, 0, execution_time)
        except LimitsDisabled:
            logger.warning("Tried sending used resources info, but limits were disabled")

        if is_chained:
            session.flow_status.pipeline_finish_step(Steps.CALCULATE_INDICATORS)
        session_manager_singleton.save(current_user.email, session)

        return FlowStatusResponseModel(
            flow_status=session.flow_status.to_dict(),
        ).model_dump()
    # TODO: handle only specific exceptions in prod
    except (SoftTimeLimitExceeded, LLMBadResponseError, Exception) as e:
        _handle_error(
            session_id=session.session_id,
            email=current_user.email,
            step=Steps.CALCULATE_INDICATORS,
            exc_to_raise=e,
            indicators_code_log=session.get("indicators_code_log", []),
        )


def _build_indicators_plots_meta(session):
    plots_meta = {}

    for symbol in session["main_roots"]:
        charts = []
        composite_chart = [{"name": "close", "type": "price"}]
        for ind in session["roots"][symbol]:
            composite_chart.append({"name": ind, "type": "indicator"})
        charts.append(composite_chart)
        for ind in session["main_roots"][symbol]:
            charts.append([{"name": ind, "type": "indicator"}])
        plots_meta[symbol] = {"charts": charts, "type": _determine_plot_type(symbol, session)}

    return plots_meta


@current_app.task(name="calculate_backtesting", bind=True)
def calculate_backtesting(
    self,
    actual_currency: str,
    bet_size: float,
    per_instrument_gross_limit: float,
    total_gross_limit: float,
    nop_limit: float,
    account_for_dividends: bool,
    trade_fill_price: str,
    execution_cost_bps: float,
) -> Dict[str, Any]:
    current_user = user.get()

    session = session_manager_singleton.get(current_user.email, self.session_id)

    session["actual_currency"] = actual_currency
    session["bet_size"] = bet_size
    session["per_instrument_gross_limit"] = per_instrument_gross_limit
    session["total_gross_limit"] = total_gross_limit
    session["nop_limit"] = nop_limit
    session["use_dividends_trading"] = account_for_dividends
    session["fill_trade_price"] = trade_fill_price
    session["execution_cost_bps"] = execution_cost_bps

    lclsglbls_before = session["lclsglbls"]

    try:
        elapsed_time = _run_backtesting_with_progress(
            session=session,
            account_for_dividends=account_for_dividends,
            progress_channel=f"task-{self.request.id}-progress",
        )

        try:
            alerts_backend_proxy_singleton.send_used_resources_info(0, 0, elapsed_time)
        except LimitsDisabled:
            logger.warning("Tried sending used resources info, but limits were disabled")

        plots_meta = _build_backtesting_plots_meta(session)

        session.flow_status.promote_backtesting_step(
            {
                "execution_time": elapsed_time,
                "logs_present": True if session["trading_code_log"] else False,
                "plots_meta": plots_meta,
            }
        )
        session["lclsglbls"] = lclsglbls_before
        session_manager_singleton.save(current_user.email, session)

        payload = {"progress": PUBSUB_END_OF_DATA}
        _safe_progress_publish(
            pubsub=sync_redis_client,
            channel=f"task-{self.request.id}-progress",
            message=json.dumps(payload),
            error_flag=False,
        )

        return FlowStatusResponseModel(
            flow_status=session.flow_status.to_dict(),
        ).model_dump()
    # TODO: handle only specific exceptions in prod
    except (SoftTimeLimitExceeded, LLMBadResponseError, Exception) as e:
        _handle_error(
            session_id=session.session_id,
            email=current_user.email,
            step=Steps.PERFORM_BACKTESTING,
            exc_to_raise=e,
            trading_code_log=session.get("trading_code_log", []),
        )


def _run_backtesting_with_progress(
    session: Session,
    account_for_dividends: bool,
    progress_channel: str,
) -> float:
    publish_error_encountered = False
    elapsed_time = 0.0

    for progress, elapsed_time, remaining_time in progress_and_time_generator()(
        trading_step,
        session,
        apply_dividends=account_for_dividends,
    ):
        payload = {"progress": progress, "elapsed_time": elapsed_time, "remaining_time": remaining_time}
        publish_error_encountered = _safe_progress_publish(
            pubsub=sync_redis_client,
            channel=progress_channel,
            message=json.dumps(payload),
            error_flag=publish_error_encountered,
        )

    return elapsed_time


def _build_backtesting_plots_meta(session):
    return {
        "symbols": list(
            filter(
                lambda s: _determine_plot_type(s, session) == "tradable",
                ChainMap(session["data_by_symbol"], session.get("data_by_synth", {})),
            )
        ),
        "start_date": convert_date(session["start_date"]),
        "end_date": convert_date(session["end_date"]),
    }


@current_app.task(name="run_optimization_prepare", bind=True)
def run_optimization_prepare(
    self,
    actual_currency: str,
    bet_size: float,
    per_instrument_gross_limit: float,
    total_gross_limit: float,
    nop_limit: float,
    account_for_dividends: bool,
    trade_fill_price: str,
    execution_cost_bps: float,
    n_trials: int,
    train_size: float,
    params: list[dict[str, Any]],
    minimize: bool,
    maximize: bool,
    sampler: str,
    target_func: str,
    studies_names: list[str],
) -> None:
    current_user = user.get()

    session = session_manager_singleton.get(current_user.email, self.session_id)

    session["actual_currency"] = actual_currency
    session["bet_size"] = bet_size
    session["per_instrument_gross_limit"] = per_instrument_gross_limit
    session["total_gross_limit"] = total_gross_limit
    session["nop_limit"] = nop_limit
    session["use_dividends_trading"] = account_for_dividends
    session["fill_trade_price"] = trade_fill_price
    session["execution_cost_bps"] = execution_cost_bps

    session["optimization_trials"] = n_trials
    session["optimization_train_size"] = train_size
    session["optimization_params"] = params
    session["optimization_minimize"] = minimize
    session["optimization_maximize"] = maximize
    session["optimization_sampler"] = sampler
    session["optimization_target_func"] = target_func

    if old_studies_names := session.get("optimization_studies_names"):
        for study_name in old_studies_names:
            try:
                delete_optimization_study(get_optimization_storage(pool_size=3, max_overflow=3), study_name)
            except KeyError:
                # Study doesn't exist, perhaps it was revoked
                pass

    session["optimization_studies_names"] = studies_names

    session["range_by_param"] = {p["name"]: p["values"] for p in params}

    try:
        session["fx_rates"] = get_actual_currency_fx_rates(session, actual_currency)
        if account_for_dividends:
            session["dividends_by_symbol"] = get_sparse_dividends_for_each_tradable_symbol(session)

        session_manager_singleton.save(current_user.email, session)

    # TODO: handle only specific exceptions in prod
    except (SoftTimeLimitExceeded, Exception) as e:
        _handle_error(
            session_id=session.session_id,
            email=current_user.email,
            step=Steps.OPTIMIZE,
            exc_to_raise=e,
        )
        # TODO: need to handle study removal if pipeline dropped with exception
        # try:
        #     if minimization_study:
        #         delete_study(study_name=minimization_study.study_name, storage=...)
        #     if maximization_study:
        #         delete_study(study_name=maximization_study.study_name, storage=...)
        # except Exception as e:
        #     logger.warning(f"Couldn't remove optuna studies from DB due to error: %s", e)


@current_app.task(name="run_optimization", bind=True)
def run_optimization(
    self,
    study_name: str,
    target_func: str,
    sampler: str,
    account_for_dividends: bool,
    n_trials: int,
    train_size: float,
    progress_channel: str,
    pipeline_id: str,
) -> None:
    current_user = user.get()

    session = session_manager_singleton.get(current_user.email, self.session_id)

    try:
        _run_optimization_with_progress(
            session=session,
            study_name=study_name,
            target_func=target_func,
            sampler=sampler,
            account_for_dividends=account_for_dividends,
            n_trials=n_trials,
            train_size=train_size,
            progress_channel=progress_channel,
            pipeline_id=pipeline_id,
        )

        # session_manager_singleton.save(current_user.email, session)
    # TODO: handle only specific exceptions in prod
    except (SoftTimeLimitExceeded, LLMBadResponseError, Exception) as e:
        _handle_error(
            session_id=session.session_id,
            email=current_user.email,
            step=Steps.OPTIMIZE,
            exc_to_raise=e,
        )


def _run_optimization_with_progress(
    session: Session,
    study_name: str,
    target_func: str,
    sampler: str,
    account_for_dividends: bool,
    n_trials: int,
    train_size: float,
    progress_channel: str,
    pipeline_id: str,
) -> None:
    optimize(
        session,
        target_function=optimization_target_funcs[target_func]["value"],
        storage=get_optimization_storage(pool_size=3, max_overflow=3),
        sampler=optimization_samplers[sampler]["value"](),
        study_name=study_name,
        study_direction=StudyDirection.MAXIMIZE,
        study_load_if_exists=True,
        trial_callbacks=[OptimizationTrialCallback(sync_redis_client, progress_channel, pipeline_id)],
        n_trials=n_trials,
        train_size=train_size,
        apply_dividends=account_for_dividends,
        is_trades_stats_needed=optimization_target_funcs[target_func]["is_trades_stats_needed"],
    )


class OptimizationTrialCallback:
    def __init__(self, pubsub, channel: str, pipeline_id: str) -> None:
        self._pubsub = pubsub
        self._channel = channel
        self._pipeline_id = pipeline_id
        self._publish_error_encountered = False

    def __call__(self, study: Study, trial: FrozenTrial) -> None:
        message = {
            "trial_in_sample": trial.value,
            "trial_out_of_sample": trial.user_attrs["test_value"],
            "trial_params": trial.params,
            "direction": "minimization" if study.direction == StudyDirection.MINIMIZE else "maximization",
            "duration": trial.duration.total_seconds(),
        }

        try:
            alerts_backend_proxy_singleton.send_used_resources_info(0, 0, message["duration"])
        except LimitsDisabled:
            logger.debug("Tried sending used resources info, but limits were disabled")

        self._publish_error_encountered = _safe_progress_publish(
            self._pubsub, self._channel, json.dumps(message), self._publish_error_encountered
        )

        task_info = get_task_ws_meta_info(self._pipeline_id)
        if isinstance(task_info, OptimizationTaskWSMetaInfo):
            if task_info.stop_flag:
                study.stop()
                logger.info("Stopped '%s' study", study.study_name)
        else:
            raise RuntimeError(
                f"optimization trial callback expected {OptimizationTaskWSMetaInfo.__name__}, but got {type(task_info).__name__}"
            )


@current_app.task(name="run_optimization_chord", bind=True)
def run_optimization_chord(self, pipeline_start_time: int, studies_names: list[str]):
    current_user = user.get()

    session = session_manager_singleton.get(current_user.email, self.session_id)

    try:
        optimization_results = [
            get_optimization_results(settings.optimization.storage_url, study_name) for study_name in studies_names
        ]

        minimization_result = _get_optimization_result(StudyDirection.MINIMIZE, optimization_results)
        maximization_result = _get_optimization_result(StudyDirection.MAXIMIZE, optimization_results)

        session.flow_status.promote_optimization_step(
            {
                # TODO: safe only if running processes are within the same machine/pod, fine for one celery pod
                "execution_time": time.time() - pipeline_start_time,
                **OptimizationResponseModel(
                    minimization=minimization_result,
                    maximization=maximization_result,
                    sampler=session.get("optimization_sampler"),
                    target_func=session.get("optimization_target_func"),
                ).model_dump(),
            }
        )

        session_manager_singleton.save(current_user.email, session)

        payload = {"progress": PUBSUB_END_OF_DATA}
        _safe_progress_publish(
            pubsub=sync_redis_client,
            channel=f"task-{self.request.id}-progress",
            message=json.dumps(payload),
            error_flag=False,
        )

        return FlowStatusResponseModel(
            flow_status=session.flow_status.to_dict(),
        ).model_dump()
    # TODO: handle only specific exceptions in prod
    except (SoftTimeLimitExceeded, LLMBadResponseError, Exception) as e:
        _handle_error(
            session_id=session.session_id,
            email=current_user.email,
            step=Steps.OPTIMIZE,
            exc_to_raise=e,
        )


def _get_optimization_result(study_direction: StudyDirection, all_results) -> Optional[OptimizationResult]:
    try:
        best_params, _, trials = next(filter(lambda r: r[1].direction == study_direction, all_results))
    except StopIteration:
        return None
    return OptimizationResult(
        best_params=best_params,
        trials=trials,
    )


@current_app.task(name="run_after_optimization_prepare", bind=True)
def run_after_optimization_prepare(
    self,
    actual_currency: str,
    bet_size: float,
    per_instrument_gross_limit: float,
    total_gross_limit: float,
    nop_limit: float,
    account_for_dividends: bool,
    trade_fill_price: str,
    execution_cost_bps: float,
):
    current_user = user.get()

    session = session_manager_singleton.get(current_user.email, self.session_id)

    session["actual_currency"] = actual_currency
    session["bet_size"] = bet_size
    session["per_instrument_gross_limit"] = per_instrument_gross_limit
    session["total_gross_limit"] = total_gross_limit
    session["nop_limit"] = nop_limit
    session["use_dividends_trading"] = account_for_dividends
    session["fill_trade_price"] = trade_fill_price
    session["execution_cost_bps"] = execution_cost_bps

    try:
        session_manager_singleton.save(current_user.email, session)
    # TODO: handle only specific exceptions in prod
    except (SoftTimeLimitExceeded, Exception) as e:
        _handle_error(
            session_id=session.session_id,
            email=current_user.email,
            step=Steps.OPTIMIZE,
            exc_to_raise=e,
        )


@current_app.task(name="run_after_optimization", bind=True)
def run_after_optimization(
    self, params: list[dict[str, Any]], account_for_dividends: bool, progress_channel: str
) -> tuple[dict[str, Any], dict[str, Any]]:
    current_user = user.get()

    session = session_manager_singleton.get(current_user.email, self.session_id)

    constructed_llm_response = """
```python
%s
```

```python
%s
```
""" % (
        session.flow_status.get_interpolated_indicators_code_template(ChainMap(*params)),
        session.flow_status.trading_code,
    )
    session["indicators_dialogue"][-1] = constructed_llm_response

    try:
        indicator_step(session)
        _run_backtesting_with_progress(
            session=session,
            account_for_dividends=account_for_dividends,
            progress_channel=progress_channel,
        )

        return pickle.dumps((session["trading_stats_by_symbol"], session["strategy_stats"]))
    # TODO: handle only specific exceptions in prod
    except (SoftTimeLimitExceeded, LLMBadResponseError, Exception) as e:
        _handle_error(
            session_id=session.session_id,
            email=current_user.email,
            step=Steps.OPTIMIZE,
            exc_to_raise=e,
        )


@current_app.task(name="run_after_optimization_chord", bind=True)
def run_after_optimization_chord(self, results, pipeline_start_time: int, account_for_dividends: bool):
    current_user = user.get()

    session = session_manager_singleton.get(current_user.email, self.session_id)

    try:
        list_trading_stats_by_symbol, list_strategy_stats = [], []
        for result in results:
            trading_stats_by_symbol, strategy_stats = pickle.loads(result)
            list_trading_stats_by_symbol.append(trading_stats_by_symbol)
            list_strategy_stats.append(strategy_stats)

        trading_statistics = get_combined_trading_statistics(
            session, list_trading_stats_by_symbol, list_strategy_stats, apply_dividends=account_for_dividends
        )

        session.update(trading_statistics)

        session.flow_status.promote_backtesting_step(
            {
                # TODO: safe only if running processes are within the same machine/pod, fine for one celery pod
                "execution_time": time.time()
                - pipeline_start_time,
            }
        )

        session_manager_singleton.save(current_user.email, session)

        payload = {"progress": PUBSUB_END_OF_DATA}
        _safe_progress_publish(
            pubsub=sync_redis_client,
            channel=f"task-{self.request.id}-progress",
            message=json.dumps(payload),
            error_flag=False,
        )

        return FlowStatusResponseModel(
            flow_status=session.flow_status.to_dict(),
        ).model_dump()
    # TODO: handle only specific exceptions in prod
    except (SoftTimeLimitExceeded, LLMBadResponseError, Exception) as e:
        _handle_error(
            session_id=session.session_id,
            email=current_user.email,
            step=Steps.OPTIMIZE,
            exc_to_raise=e,
        )


@current_app.task(name="chained_load_model_into_session", bind=True)
def chained_load_model_into_session(
    self,
    model_id: Optional[int],
    is_public: bool,
    indicators_dialogue: List[str],
    data_provider: str,
    datasets: List[str],
    periodicity: int,
    tradable_symbols_prompt: str,
    supplementary_symbols_prompt: str,
    economic_indicators: List[str],
    dividend_fields: List[str],
    time_range: int,
    strategy_title: str,
    strategy_description: str,
    actual_currency: str,
    bet_size: float,
    per_instrument_gross_limit: float,
    total_gross_limit: float,
    nop_limit: float,
    account_for_dividends: bool,
    trade_fill_price: str,
    execution_cost_bps: float,
    optimization_trials: int,
    optimization_train_size: float,
    optimization_params: List[dict[str, Any]],
    optimization_minimize: bool,
    optimization_maximize: bool,
    optimization_sampler: str,
    optimization_target_func: str,
) -> None:
    current_user = user.get()

    session = session_manager_singleton.get(current_user.email, self.session_id)

    session.reset_flow_status()
    if not is_public and model_id is not None:
        session.flow_status.set_model_opened(model_id)

    session["indicators_dialogue"] = indicators_dialogue

    session["data_provider"] = data_provider
    session["datasets_keys"] = datasets
    session["interval"] = periodicity
    session["tradable_symbols_prompt"] = tradable_symbols_prompt
    session["supplementary_symbols_prompt"] = supplementary_symbols_prompt
    session["economic_indicators"] = economic_indicators
    session["dividend_fields"] = dividend_fields
    session["time_period"] = time_range
    session["strategy_title"] = strategy_title
    session["strategy_description"] = strategy_description
    session["actual_currency"] = actual_currency
    session["bet_size"] = bet_size
    session["per_instrument_gross_limit"] = per_instrument_gross_limit
    session["total_gross_limit"] = total_gross_limit
    session["nop_limit"] = nop_limit
    session["use_dividends_trading"] = account_for_dividends
    session["fill_trade_price"] = trade_fill_price
    session["execution_cost_bps"] = execution_cost_bps

    session["optimization_trials"] = optimization_trials
    session["optimization_train_size"] = optimization_train_size
    session["optimization_params"] = optimization_params
    session["optimization_minimize"] = optimization_minimize
    session["optimization_maximize"] = optimization_maximize
    session["optimization_sampler"] = optimization_sampler
    session["optimization_target_func"] = optimization_target_func

    session_manager_singleton.save(current_user.email, session)


@current_app.task(name="chained_submit_llm_chat", bind=True)
def chained_submit_llm_chat(
    self,
    indicators_code: str,
    trading_code: str,
) -> None:
    middleware.check_limits_middleware()

    current_user = user.get()

    session = session_manager_singleton.get(current_user.email, self.session_id)

    session.flow_status.promote_submit_llm_chat_step(indicators_code, trading_code)

    session_manager_singleton.save(current_user.email, session)


@current_app.task(name="clear_expired_sessions", base=Task)
def clear_expired_sessions() -> None:
    session_manager_singleton.delete_expired_sessions(settings.optimization.storage_url)


def _handle_error(session_id: str, email: str, step: Steps, exc_to_raise: Exception, **additional_session_values):
    session = session_manager_singleton.get(email, session_id)
    for key, value in additional_session_values.items():
        session[key] = value
    session.flow_status.add_error_for_step(step, get_task_error_msg(exc_to_raise))
    session_manager_singleton.save(email, session)
    raise exc_to_raise


def _safe_progress_publish(pubsub, channel: str, message: str, error_flag: bool) -> bool:
    try:
        pubsub.publish(channel, message)
        if error_flag:
            logger.info(f"Redis is back up. Resuming progress updates to '{channel}'")
            return False
    except RedisError as e:
        if not error_flag:
            logger.warning(f"Some problem occurred while pushing progress updates to '{channel}' Redis channel: {e}")
            return True
    return error_flag
