import logging
from copy import copy
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import optuna
import pandas as pd
from optuna import Study
from optuna.samplers import BaseSampler
from optuna.storages import BaseStorage
from optuna.study import StudyDirection
from optuna.trial import FrozenTrial

from market_alerts.containers import data_periodicities, data_providers
from market_alerts.domain.services.steps import indicator_step, trading_step
from market_alerts.domain.services.steps.utils import get_fx_rate, get_sparse_dividends

logger = logging.getLogger(__name__)


def save_new_keys_in_origin_session(session_dict_origin, session_dict):
    if "fx_rates" in session_dict:
        session_dict_origin.setdefault("fx_rates", dict())
        new_keys = set(session_dict["fx_rates"].keys()) - set(session_dict_origin["fx_rates"].keys())
        if new_keys:
            provider = data_providers[session_dict_origin["data_provider"]]
            for fx_rate_symbol in new_keys:
                left_curr, right_curr = fx_rate_symbol.split("/")
                if left_curr == right_curr:
                    session_dict_origin["fx_rates"][fx_rate_symbol] = pd.DataFrame(
                        data=1.0, columns=["open", "high", "low", "close"], index=session_dict_origin["time_line"]
                    )
                else:
                    session_dict_origin["fx_rates"][fx_rate_symbol] = get_fx_rate(
                        session_dict_origin["time_line"],
                        fx_rate_symbol,
                        session_dict_origin["start_date"],
                        session_dict_origin["end_date"],
                        data_periodicities[session_dict_origin["interval"]]["value"],
                        provider,
                    )

    if "dividends_by_symbol" in session_dict:
        session_dict_origin.setdefault("dividends_by_symbol", dict())
        new_keys = set(session_dict["dividends_by_symbol"].keys()) - set(session_dict_origin["dividends_by_symbol"].keys())
        if new_keys:
            provider = data_providers[session_dict_origin["data_provider"]]
            for symbol in new_keys:
                session_dict_origin["dividends_by_symbol"][symbol] = get_sparse_dividends(
                    time_line=session_dict_origin["time_line"],
                    provider=provider,
                    symbol=symbol,
                    div_end_date=session_dict_origin["end_date"],
                    div_start_date=session_dict_origin["start_date"],
                    true_symbols=session_dict_origin["true_symbols"],
                )


def objective(trial, session, target_function, apply_dividends: bool, train_size: float, is_trades_stats_needed: bool) -> Any:
    #     session_origin = session
    session = copy(session)
    session.data = copy(session.data)
    session["u_strs"] = session["u_strs"].copy()
    value_by_param = dict()
    for key, (_, type_) in session.flow_status.parsed_optimization_params.items():
        if type_ == "int":
            value_by_param[key] = trial.suggest_int(key, session["range_by_param"][key][0], session["range_by_param"][key][1])
        elif type_ == "float":
            value_by_param[key] = trial.suggest_float(key, session["range_by_param"][key][0], session["range_by_param"][key][1])
        else:
            value_by_param[key] = trial.suggest_categorical(key, session["range_by_param"][key])
    new_llm_response = """
```python
%s
```

```python
%s
```
""" % (
        session.flow_status.get_interpolated_indicators_code_template(value_by_param),
        session.flow_status.trading_code,
    )
    session["indicators_dialogue"][-1] = new_llm_response

    if train_size < 1.0:
        n_train = round(train_size * len(session["time_line"]))
        n_overlap = 250
        session_train = session.get_slice(0, n_train)
        session_test = session.get_slice(max(0, n_train - n_overlap), len(session["time_line"]))

        logger.debug("Running in-sample...")

        indicator_step(session_train)
        for _ in trading_step(session_train, apply_dividends=apply_dividends, is_trades_stats_needed=is_trades_stats_needed):
            pass

        logger.debug("Running out-of-sample...")

        indicator_step(session_test)
        for _ in trading_step(
            session_test, apply_dividends=apply_dividends, start_idx=n_overlap, is_trades_stats_needed=is_trades_stats_needed
        ):
            pass

        res = target_function(session_test)

        if np.issubdtype(type(res), np.integer):
            res = int(res)
        trial.set_user_attr("test_value", res)

        return target_function(session_train)
    else:
        session = session.get_slice(0, len(session["time_line"]))
        indicator_step(session)
        for _ in trading_step(session, apply_dividends=apply_dividends, is_trades_stats_needed=is_trades_stats_needed):
            pass

        trial.set_user_attr("test_value", 0.0)

        return target_function(session)


def optimize(
    session,
    target_function,
    storage: str | BaseStorage,
    sampler: BaseSampler,
    study_name: str,
    study_direction: Optional[StudyDirection] = None,
    study_load_if_exists: bool = False,
    trial_callbacks: Optional[list[Callable[[Study, FrozenTrial], None]]] = None,
    n_trials: int = 5,
    train_size: float = 1.0,
    apply_dividends: bool = False,
    is_trades_stats_needed: bool = True,
) -> None:
    study = optuna.create_study(
        storage=storage, sampler=sampler, study_name=study_name, direction=study_direction, load_if_exists=study_load_if_exists
    )
    study.optimize(
        lambda trial: objective(
            trial,
            session,
            target_function,
            apply_dividends=apply_dividends,
            train_size=train_size,
            is_trades_stats_needed=is_trades_stats_needed,
        ),
        n_trials=n_trials,
        callbacks=trial_callbacks,
    )


def get_optimization_results(
    storage: Optional[str | BaseStorage],
    study_name: str,
) -> Tuple[Dict[str, Any], Study, List[Tuple[int, float, float, Dict[str, Any], float]]]:
    study = optuna.load_study(study_name=study_name, storage=storage)
    return (
        study.best_params,
        study,
        sorted(
            [
                (trial.number + 1, trial.value, trial.user_attrs["test_value"], trial.params, trial.duration.total_seconds())
                for trial in study.get_trials()
            ],
            key=lambda x: x[0],
        ),
    )


def delete_optimization_study(storage: Optional[str | BaseStorage], study_name: str) -> None:
    optuna.delete_study(study_name=study_name, storage=storage)
    logger.info("Study %s deleted successfully", study_name)
