import time
import uuid
from typing import Any

import optuna
from celery import chain, chord
from fastapi import APIRouter, Body, Depends, Query
from optuna.study import StudyDirection

from market_alerts.containers import optimization_samplers
from market_alerts.entrypoints.llm_backend.api.models.optimization import (
    AfterOptimizationBacktestingRequestModel,
    AfterOptimizationSetAsDefaultRequestModel,
    OptimizationCalendarResponseModel,
    OptimizationRequestModel,
    validate_params,
)
from market_alerts.entrypoints.llm_backend.api.models.session import (
    SessionInfoResponseModel,
)
from market_alerts.entrypoints.llm_backend.api.utils import get_session
from market_alerts.entrypoints.llm_backend.containers import (
    celery_app,
    get_optimization_storage,
    session_manager_singleton,
    settings,
)
from market_alerts.entrypoints.llm_backend.domain.exceptions import (
    IndicatorsBlockIsNotPresentError,
    LLMChatNotSubmittedError,
    TradingBlockIsNotPresentError,
)
from market_alerts.entrypoints.llm_backend.infrastructure.access_management.context_vars import (
    user,
)
from market_alerts.entrypoints.llm_backend.infrastructure.services.ws.task_meta_info import (
    save_task_ws_meta_info,
)
from market_alerts.entrypoints.llm_backend.infrastructure.session import Session, Steps
from market_alerts.entrypoints.llm_backend.infrastructure.utils import (
    create_task_headers,
)
from market_alerts.infrastructure.services.code import update_code_sections
from market_alerts.utils import convert_date

optimization_router = APIRouter(prefix="/optimization")


@optimization_router.post("/run", tags=["Optimization"])
def run_optimization(
    request_model: OptimizationRequestModel = Body(...),
    session: Session = Depends(get_session),
):
    if not session.flow_status.is_step_done(Steps.SUBMIT_LLM_CHAT):
        raise LLMChatNotSubmittedError
    if not session.flow_status.is_indicators_block_present:
        raise IndicatorsBlockIsNotPresentError
    if not session.flow_status.is_trading_block_present:
        raise TradingBlockIsNotPresentError

    validate_params(request_model.params, session.flow_status.parsed_optimization_params)

    current_user = user.get()

    studies_names = []

    sampler = optimization_samplers[request_model.sampler]["value"]()
    is_sampler_random = optimization_samplers[request_model.sampler].get("is_random", False)

    storage = get_optimization_storage(pool_size=3, max_overflow=2)

    if request_model.maximize:
        maximization_study = optuna.create_study(
            storage=storage,
            sampler=sampler,
            study_name=f"{current_user.email}-maximization-study-{request_model.sampler}-{request_model.target_func}-{uuid.uuid4().hex}",
            direction=StudyDirection.MAXIMIZE,
        )
        studies_names.append(maximization_study.study_name)
    if not is_sampler_random and request_model.minimize:
        minimization_study = optuna.create_study(
            storage=storage,
            sampler=sampler,
            study_name=f"{current_user.email}-minimization-study-{request_model.sampler}-{request_model.target_func}-{uuid.uuid4().hex}",
            direction=StudyDirection.MINIMIZE,
        )
        studies_names.append(minimization_study.study_name)

    optimization_prepare_task = celery_app.signature(
        "run_optimization_prepare",
        kwargs={
            "actual_currency": request_model.actual_currency,
            "bet_size": request_model.bet_size,
            "per_instrument_gross_limit": request_model.per_instrument_gross_limit,
            "total_gross_limit": request_model.total_gross_limit,
            "nop_limit": request_model.nop_limit,
            "account_for_dividends": request_model.account_for_dividends,
            "trade_fill_price": request_model.trade_fill_price,
            "execution_cost_bps": request_model.execution_cost_bps,
            "n_trials": request_model.n_trials,
            "train_size": request_model.train_size,
            "params": [p.model_dump() for p in request_model.params],
            "minimize": request_model.minimize,
            "maximize": request_model.maximize,
            "sampler": request_model.sampler,
            "target_func": request_model.target_func,
            "studies_names": studies_names,
        },
        queue="alerts_default",
        routing_key="alerts_default",
        headers=create_task_headers(session.session_id),
        task_id=str(uuid.uuid4()),
    )

    pipeline_id = str(uuid.uuid4())
    pipeline_start_time = time.time()

    def get_optimization_signature(study_name: str, n_trials: int):
        return celery_app.signature(
            "run_optimization",
            kwargs={
                "study_name": study_name,
                "target_func": request_model.target_func,
                "sampler": request_model.sampler,
                "account_for_dividends": request_model.account_for_dividends,
                "n_trials": n_trials,
                "train_size": request_model.train_size,
                "progress_channel": f"task-{pipeline_id}-progress",
                "pipeline_id": pipeline_id,
            },
            queue="alerts_default",
            routing_key="alerts_default",
            headers=create_task_headers(session.session_id),
            task_id=str(uuid.uuid4()),
        )

    optimization_chord_task = celery_app.signature(
        "run_optimization_chord",
        kwargs={
            "studies_names": studies_names,
            "pipeline_start_time": pipeline_start_time,
        },
        queue="alerts_default",
        routing_key="alerts_default",
        headers=create_task_headers(session.session_id),
        task_id=pipeline_id,
        immutable=True,
    )

    n_trials_total = request_model.n_trials * len(studies_names)

    tasks_per_study = settings.optimization.tasks_per_study
    if len(studies_names) == 1:
        tasks_per_study *= 2

    trials_per_task = distribute_trials_per_task(n_trials_total, studies_names, tasks_per_study)

    tasks = [get_optimization_signature(*distribution) for distribution in trials_per_task]
    optimization_chord = chord(header=tasks, body=optimization_chord_task)
    optimization_chord.set_immutable(True)

    pipeline = chain(
        optimization_prepare_task,
        optimization_chord,
    )

    pipeline_task_ids = [optimization_prepare_task.id] + [task.id for task in tasks] + [pipeline_id]

    save_task_ws_meta_info(
        pipeline_id,
        task_ids=pipeline_task_ids,
        studies_names=studies_names,
        work_units=n_trials_total,
        start_time=pipeline_start_time,
    )

    for task_id in pipeline_task_ids:
        session.add_action(task_id, pipeline_start_time)
    session_manager_singleton.save(current_user.email, session)

    soft_time_limit = 10 * 60
    pipeline.apply_async(soft_time_limit=soft_time_limit, time_limit=soft_time_limit + 20, task_id=pipeline_id)

    return {
        "task_id": pipeline_id,
    }


def distribute_trials_per_task(n_trials: int, studies: list[str], tasks_per_study: int) -> list[tuple[str, int]]:
    if tasks_per_study > n_trials:
        raise RuntimeError("tasks_per_study was greater than n_trials")

    num_studies = len(studies)

    quotient, remainder = divmod(n_trials, num_studies)

    trials_per_study = [quotient + 1 if i < remainder else quotient for i in range(num_studies)]

    results = []

    for study_index, study in enumerate(studies):
        trials_in_study = trials_per_study[study_index]

        quotient, remainder = divmod(trials_in_study, tasks_per_study)

        trials_per_task_in_study = [quotient + 1 if i < remainder else quotient for i in range(tasks_per_study)]

        results.extend((study, t) for t in trials_per_task_in_study if t > 0)

    return results


@optimization_router.post("/after", tags=["Optimization"])
def run_after_optimization(
    request_model: AfterOptimizationBacktestingRequestModel = Body(...),
    session: Session = Depends(get_session),
):
    if not session.flow_status.is_step_done(Steps.SUBMIT_LLM_CHAT):
        raise LLMChatNotSubmittedError
    if not session.flow_status.is_indicators_block_present:
        raise IndicatorsBlockIsNotPresentError
    if not session.flow_status.is_trading_block_present:
        raise TradingBlockIsNotPresentError

    for params in request_model.params:
        validate_params(params, session.flow_status.parsed_optimization_params)

    current_user = user.get()

    pipeline_id = str(uuid.uuid4())
    pipeline_start_time = time.time()

    after_optimization_prepare_task = celery_app.signature(
        "run_after_optimization_prepare",
        kwargs={
            "actual_currency": request_model.actual_currency,
            "bet_size": request_model.bet_size,
            "per_instrument_gross_limit": request_model.per_instrument_gross_limit,
            "total_gross_limit": request_model.total_gross_limit,
            "nop_limit": request_model.nop_limit,
            "account_for_dividends": request_model.account_for_dividends,
            "trade_fill_price": request_model.trade_fill_price,
            "execution_cost_bps": request_model.execution_cost_bps,
        },
        queue="alerts_default",
        routing_key="alerts_default",
        headers=create_task_headers(session.session_id),
        task_id=str(uuid.uuid4()),
    )

    def get_after_optimization_signature(params_: list[dict[str, Any]]):
        return celery_app.signature(
            "run_after_optimization",
            kwargs={
                "params": params_,
                "account_for_dividends": request_model.account_for_dividends,
                "progress_channel": f"task-{pipeline_id}-progress",
            },
            queue="alerts_default",
            routing_key="alerts_default",
            headers=create_task_headers(session.session_id),
            task_id=str(uuid.uuid4()),
        )

    after_optimization_chord_task = celery_app.signature(
        "run_after_optimization_chord",
        kwargs={
            "pipeline_start_time": pipeline_start_time,
            "account_for_dividends": request_model.account_for_dividends,
        },
        queue="alerts_default",
        routing_key="alerts_default",
        headers=create_task_headers(session.session_id),
        task_id=pipeline_id,
    )

    tasks = [get_after_optimization_signature([{p.name: p.value} for p in params]) for params in request_model.params]
    after_optimization_chord = chord(header=tasks, body=after_optimization_chord_task)
    after_optimization_chord.set_immutable(True)

    pipeline = chain(
        after_optimization_prepare_task,
        after_optimization_chord,
    )

    pipeline_task_ids = [after_optimization_prepare_task.id] + [task.id for task in tasks] + [pipeline_id]
    work_units = 100 * len(request_model.params)

    save_task_ws_meta_info(
        pipeline_id,
        task_ids=pipeline_task_ids,
        work_units=work_units,
        start_time=pipeline_start_time,
    )

    for task_id in pipeline_task_ids:
        session.add_action(task_id, pipeline_start_time)
    session_manager_singleton.save(current_user.email, session)

    soft_time_limit = 10 * 60
    pipeline.apply_async(soft_time_limit=soft_time_limit, time_limit=soft_time_limit + 20, task_id=pipeline_id)

    return {
        "task_id": pipeline_id,
    }


@optimization_router.post("/set_as_default", response_model=SessionInfoResponseModel, tags=["Optimization"])
def set_as_default(
    request_model: AfterOptimizationSetAsDefaultRequestModel = Body(...),
    session: Session = Depends(get_session),
):
    if not session.flow_status.is_step_done(Steps.SUBMIT_LLM_CHAT):
        raise LLMChatNotSubmittedError
    if not session.flow_status.is_indicators_block_present:
        raise IndicatorsBlockIsNotPresentError
    if not session.flow_status.is_trading_block_present:
        raise TradingBlockIsNotPresentError

    validate_params(request_model.params, session.flow_status.parsed_optimization_params)

    params_dict = {param.name: param.value for param in request_model.params}

    new_indicators_code = session.flow_status.get_interpolated_indicators_code_template(params_dict)

    session["indicators_dialogue"][-1] = update_code_sections(
        session["indicators_dialogue"][-1], new_indicators_code, session.flow_status.trading_code
    )

    session.flow_status.promote_submit_llm_chat_step(new_indicators_code, session.flow_status.trading_code)

    return SessionInfoResponseModel(
        sessionId=session.session_id,
        flow_status=session.flow_status.to_dict(),
        datasets=session.get("datasets_keys"),
        time_range=session.get("time_period"),
        periodicity=session.get("interval"),
        account_for_dividends=session.get("use_dividends_trading"),
        trade_fill_price=session.get("fill_trade_price"),
        **session.data,
    )


@optimization_router.get("/cutoff_date", response_model=OptimizationCalendarResponseModel, tags=["Optimization"])
def get_cut_off_date(
    train_size: float = Query(..., ge=0.01, le=1.0),
    session: Session = Depends(get_session),
):
    if not session.flow_status.is_step_done(Steps.SUBMIT_LLM_CHAT):
        raise LLMChatNotSubmittedError
    if not session.flow_status.is_indicators_block_present:
        raise IndicatorsBlockIsNotPresentError
    if not session.flow_status.is_trading_block_present:
        raise TradingBlockIsNotPresentError

    session_slice = session.get_slice(0, round(train_size * len(session["time_line"])))

    date_keys = ["cutoff_date", "start_date", "end_date"]
    date_vals = [session_slice["end_date"], session["start_date"], session["end_date"]]

    dates = {date_key: convert_date(date_val) for date_key, date_val in zip(date_keys, date_vals)}

    return OptimizationCalendarResponseModel(**dates)
