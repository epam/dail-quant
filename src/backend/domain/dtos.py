from dataclasses import dataclass

from market_alerts.containers import (
    data_periodicities,
    data_providers,
    data_timeranges,
    optimization_samplers,
    optimization_target_funcs,
)
from market_alerts.containers import trade_fill_prices as tfp_container
from market_alerts.domain.data_providers.constants import DATASETS_META


@dataclass
class DTO:
    label: str
    query_param: str
    description: str
    default_checked: bool

    def __init__(self, **kwargs):
        self.label = kwargs.get("label", "")
        self.query_param = kwargs.get("query_param", "")
        self.description = kwargs.get("description", "")
        self.default_checked = kwargs.get("default_checked", False)


@dataclass
class OptimizationSamplerDTO(DTO):
    is_random: bool

    def __init__(self, **kwargs):
        self.is_random = kwargs.get("is_random", False)
        super(OptimizationSamplerDTO, self).__init__(**kwargs)


data_providers_dtos = [
    DTO(label=provider.PROVIDER_NAME, query_param=backend_key, default_checked=provider.DEFAULT_CHECKED)
    for backend_key, provider in data_providers.items()
]

data_periodicities_dtos = [DTO(query_param=backend_key, **dp) for backend_key, dp in data_periodicities.items()]

data_time_ranges_dtos = [DTO(query_param=backend_key, **dtr) for backend_key, dtr in data_timeranges.items()]

datasets_dtos = [DTO(query_param=backend_key, **ds) for backend_key, ds in DATASETS_META.items()]

trade_fill_prices_dtos = [
    DTO(label=tfp.FILL_PRICE_NAME, query_param=tfp.BACKEND_KEY, default_checked=tfp.IS_DEFAULT)
    for tfp in tfp_container.trade_fill_prices
]

optimization_samplers_dtos = [
    OptimizationSamplerDTO(query_param=backend_key, **samp) for backend_key, samp in optimization_samplers.items()
]

optimization_target_funcs_dtos = [DTO(query_param=backend_key, **tf) for backend_key, tf in optimization_target_funcs.items()]
