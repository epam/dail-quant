from fastapi import APIRouter, Body, Depends

from market_alerts.entrypoints.llm_backend.api.models.common import StrategyTitleModel
from market_alerts.entrypoints.llm_backend.api.models.jupyter import (
    JupyterLinkResponseModel,
)
from market_alerts.entrypoints.llm_backend.api.utils import get_session
from market_alerts.entrypoints.llm_backend.containers import jupyter_service_singleton
from market_alerts.entrypoints.llm_backend.domain.exceptions import DataNotFetchedError
from market_alerts.entrypoints.llm_backend.infrastructure.access_management.context_vars import (
    user,
)
from market_alerts.entrypoints.llm_backend.infrastructure.session import Session, Steps
from market_alerts.utils import session_state_getter

jupyter_router = APIRouter(prefix="/jupyter")


@jupyter_router.post("/create_notebook", response_model=JupyterLinkResponseModel, tags=["Jupyter"])
def create_and_open_notebook(
    request_model: StrategyTitleModel = Body(None),
    session: Session = Depends(get_session),
):
    if not session.flow_status.is_step_done(Steps.FETCH_TICKERS):
        raise DataNotFetchedError

    current_user = user.get()

    jupyter_service_singleton.create_user(current_user.email)
    jupyter_service_singleton.start_user_server(current_user.email)
    jupyter_service_singleton.wait_until_user_server_ready(current_user.email, timeout=60)

    data = session_state_getter(session)

    new_notebook_filename = request_model.strategy_title if request_model is not None else None

    return JupyterLinkResponseModel(
        link=jupyter_service_singleton.put_user_notebook(current_user.email, data, new_notebook_filename)
    )
