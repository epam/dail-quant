import asyncio
import logging
import time

from optuna.study import StudyDirection

from market_alerts.domain.services.steps import get_optimization_results
from market_alerts.entrypoints.llm_backend.containers import settings

logger = logging.getLogger(__name__)


class BaseTaskProgressHandler:
    def __init__(self, start_time: float, websocket):
        self._start_time = start_time
        self._websocket = websocket
        self._items_received = 0

    async def pre_process(self):
        pass

    async def post_process(self, message_data):
        self._items_received += 1
        await self._websocket.send_json(self.get_message_with_progress_data(message_data))

    def get_message_with_progress_data(self, message_data):
        return message_data


class WorkUnitBasedProgressHandler(BaseTaskProgressHandler):
    def __init__(self, work_units: int, *args, **kwargs):
        self._work_units = work_units
        super().__init__(*args, **kwargs)

    def get_message_with_progress_data(self, message_data):
        total_duration = time.time() - self._start_time
        average_time_per_task = total_duration / self._items_received
        remaining_time = average_time_per_task * (self._work_units - self._items_received)
        message_data["elapsed_time"] = total_duration
        message_data["remaining_time"] = remaining_time
        message_data["progress"] = self._items_received / self._work_units * 100
        return message_data


class OptimizationTaskProgressHandler(WorkUnitBasedProgressHandler):
    def __init__(self, studies_names: list[str], *args, **kwargs):
        self._studies_names = studies_names
        super().__init__(*args, **kwargs)

    async def pre_process(self):
        for study_name in self._studies_names:
            try:
                _, study, optimization_results = await asyncio.to_thread(
                    get_optimization_results, settings.optimization.storage_url, study_name
                )
            except Exception as e:
                logger.debug(
                    "Some error occurred while trying to read completed trials results before sending them "
                    "through websocket: [%s] %s",
                    e.__class__.__name__,
                    e,
                )
                continue

            for _, trial_in_sample, trial_out_of_sample, trial_params, _ in optimization_results:
                message = {
                    "trial_in_sample": trial_in_sample,
                    "trial_out_of_sample": trial_out_of_sample,
                    "trial_params": trial_params,
                    "direction": "minimization" if study.direction == StudyDirection.MINIMIZE else "maximization",
                }
                self._items_received += len(optimization_results)
                await self._websocket.send_json(self.get_message_with_progress_data(message))
