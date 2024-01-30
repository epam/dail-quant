from enum import Enum
from typing import Any, Optional, Union

from fastapi import Query
from pydantic import BaseModel, Field, field_validator
from pydantic_core.core_schema import FieldValidationInfo

from market_alerts.domain.dtos import (
    optimization_samplers_dtos,
    optimization_target_funcs_dtos,
)
from market_alerts.entrypoints.llm_backend.api.models.llm import BacktestingRequestModel
from market_alerts.entrypoints.llm_backend.domain.exceptions import (
    InvalidOptimizationParamError,
)


class ParamType(str, Enum):
    string = "str"
    integer = "int"
    float = "float"


class ParamRange(BaseModel):
    name: str
    values: list


class Param(BaseModel):
    name: str
    value: Union[int, float, str]


class OptimizationRequestModel(BacktestingRequestModel):
    n_trials: int = Field(100, ge=1, le=1000)
    train_size: float = Field(1.0, ge=0.01, le=1.0)
    minimize: bool = Field(True)
    maximize: bool = Field(True)
    params: list[ParamRange]
    sampler: str = optimization_samplers_dtos[0].query_param
    target_func: str = optimization_target_funcs_dtos[0].query_param

    @field_validator("maximize")
    def check_minimize_or_maximize(cls, v: bool, info: FieldValidationInfo):
        if "minimize" in info.data:
            if not (info.data["minimize"] or v):
                raise ValueError("Either minimize or maximize must be True")
        return v

    @field_validator("sampler")
    def check_sampler(cls, v: str):
        allowed_values = [dto.query_param for dto in optimization_samplers_dtos]
        if v not in allowed_values:
            raise ValueError(
                f"Sampler value '{v}' was provided, which is not among the allowed values: {', '.join(allowed_values)}"
            )
        return v

    @field_validator("target_func")
    def check_target_func(cls, v: str):
        allowed_values = [dto.query_param for dto in optimization_target_funcs_dtos]
        if v not in allowed_values:
            raise ValueError(
                f"Target func value '{v}' was provided, which is not among the allowed values: {', '.join(allowed_values)}"
            )
        return v


class OptimizationResult(BaseModel):
    best_params: dict[str, Any]
    trials: list[tuple[int, float, float, dict[str, Any], float]]


class OptimizationResponseModel(BaseModel):
    minimization: Optional[OptimizationResult]
    maximization: Optional[OptimizationResult]
    sampler: Optional[str]
    target_func: Optional[str]


class AfterOptimizationSetAsDefaultRequestModel(BaseModel):
    params: list[Param]


class AfterOptimizationBacktestingRequestModel(BacktestingRequestModel):
    params: list[list[Param]]


def validate_params(
    params: Union[list[Param], list[ParamRange]],
    optimization_params: dict[str, list[Any]],
) -> None:
    param_names = {param.name for param in params}

    if not set(optimization_params.keys()).issubset(param_names):
        missing_keys = set(optimization_params.keys()) - param_names
        raise InvalidOptimizationParamError(f"Missing params: {', '.join(missing_keys)}")

    for param in params:
        range_mode = isinstance(param, ParamRange)

        param_name = param.name

        _validate_param_name(param_name, optimization_params)

        _, param_type = optimization_params[param_name]

        values = param.values if range_mode else [param.value]

        if param_type == ParamType.string:
            if not all(isinstance(v, str) for v in values):
                raise InvalidOptimizationParamError(
                    f"'{param_name}' param has type 'str', all values must have typalerte 'str' as well"
                )
            if range_mode and (len(values) < 1 or len(values) > 10):
                raise InvalidOptimizationParamError(
                    f"'{param_name}' param has type 'str', the values amount must be between 1 and 10"
                )
        elif param_type == ParamType.integer:
            if not all(isinstance(v, int) for v in values):
                raise InvalidOptimizationParamError(
                    f"'{param_name}' param has type 'int', all values must have type 'int' as well"
                )
            if range_mode and len(values) != 2:
                raise InvalidOptimizationParamError(f"'{param_name}' param has type 'int', the values amount must be 2")
        elif param_type == ParamType.float:
            if not all(isinstance(v, (int, float)) for v in values):
                raise InvalidOptimizationParamError(
                    f"'{param_name}' param has type 'float', all values must have type 'float' as well"
                )
            if range_mode and len(values) != 2:
                raise InvalidOptimizationParamError(f"'{param_name}' param has type 'float', the values amount must be 2")


def _validate_param_name(param_name, optimization_params: dict[str, list[Any]]):
    if param_name not in optimization_params:
        raise InvalidOptimizationParamError(f"'{param_name}' param wasn't met in the code")


class OptimizationResultRequestModel(BaseModel):
    studies_names: list[str] = Field(Query([]))


class OptimizationCalendarResponseModel(BaseModel):
    cutoff_date: str
    start_date: str
    end_date: str
