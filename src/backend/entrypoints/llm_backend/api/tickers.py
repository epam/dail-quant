import time
import uuid

from fastapi import APIRouter, Depends

from market_alerts.containers import data_providers
from market_alerts.domain.data_providers.constants import WORK_UNITS_PER_DATASET
from market_alerts.domain.services.steps.utils import parse_tickers
from market_alerts.entrypoints.llm_backend.api.models.tickers import (
    TickersFetchRequestModel,
    TickersLookupRequestModel,
)
from market_alerts.entrypoints.llm_backend.api.utils import get_session
from market_alerts.entrypoints.llm_backend.containers import (
    celery_app,
    session_manager_singleton,
)
from market_alerts.entrypoints.llm_backend.infrastructure.access_management.context_vars import (
    user,
)
from market_alerts.entrypoints.llm_backend.infrastructure.services.ws.task_meta_info import (
    save_task_ws_meta_info,
)
from market_alerts.entrypoints.llm_backend.infrastructure.session import Session
from market_alerts.entrypoints.llm_backend.infrastructure.utils import (
    create_task_headers,
)

tickers_router = APIRouter(prefix="/tickers")


@tickers_router.get("/fetch", tags=["Tickers"])
def fetch_tickers(request_model: TickersFetchRequestModel = Depends(), session: Session = Depends(get_session)):
    current_user = user.get()

    task = celery_app.signature(
        "fetch_tickers",
        kwargs={
            "data_provider": request_model.data_provider,
            "datasets": request_model.datasets,
            "periodicity": request_model.periodicity,
            "tradable_symbols_prompt": request_model.tradable_symbols_prompt,
            "supplementary_symbols_prompt": request_model.supplementary_symbols_prompt,
            "economic_indicators": request_model.economic_indicators,
            "dividend_fields": request_model.dividend_fields,
            "time_range": request_model.time_range,
        },
        queue="alerts_default",
        routing_key="alerts_default",
        headers=create_task_headers(session.session_id),
    )

    all_symbols, _, _ = parse_tickers(request_model.tradable_symbols_prompt, request_model.data_provider)

    num_symbols = sum(len(v) for v in all_symbols.values())

    work_units = num_symbols * sum(WORK_UNITS_PER_DATASET.get(ds, 0) for ds in request_model.datasets) + len(
        request_model.economic_indicators
    )

    task_id = str(uuid.uuid4())
    task_scheduled_timestamp = time.time()

    save_task_ws_meta_info(
        task_id,
        work_units=work_units,
        start_time=task_scheduled_timestamp,
    )

    session.add_action(task_id, task_scheduled_timestamp)
    session_manager_singleton.save(current_user.email, session)

    soft_time_limit = 10 * 60
    task.apply_async(soft_time_limit=soft_time_limit, time_limit=soft_time_limit + 20, task_id=task_id)

    return {
        "task_id": task_id,
    }


@tickers_router.get("/lookup", tags=["Tickers"])
def lookup_tickers(request_model: TickersLookupRequestModel = Depends()):
    data_provider = data_providers[request_model.data_provider]
    return data_provider.search_ticker(request_model.tickers_query)
