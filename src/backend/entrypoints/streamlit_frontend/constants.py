from market_alerts.containers import data_periodicities, data_providers, datasets

DATA_PROVIDERS_NAMES_TO_BACKEND_KEY = {provider.PROVIDER_NAME: backend_key for backend_key, provider in data_providers.items()}

DATA_PERIODICITIES_VALUES_TO_BACKEND_KEY = {
    periodicity["value"]: backend_key for backend_key, periodicity in data_periodicities.items()
}

DATASETS_OPTIONS = [
    dict(name=dataset.DATASET_NAME, description=dataset.DATASET_DESCRIPTION, default_checked=dataset.IS_DEFAULT)
    for dataset in datasets.values()
]

PAGINATION_MIN_PAGE_SIZE = 10

PAGE_SIZE_OPTIONS = [10, 20, 30, 50, 100]

DEFAULT_PAGE_SIZE = 100
