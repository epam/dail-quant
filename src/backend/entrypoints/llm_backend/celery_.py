from datetime import timedelta

from celery import Celery, Task
from celery.app import trace

from market_alerts.domain.exceptions import AuthError
from market_alerts.entrypoints.llm_backend import Settings
from market_alerts.entrypoints.llm_backend.infrastructure.access_management.context_vars import (
    UserValueObject,
    user,
)

trace.LOG_SUCCESS = """\
Task %(name)s[%(id)s] succeeded in %(runtime)ss\
"""

auth_settings = Settings().authentication


class BaseTask(Task):
    def __call__(self, *args, **kwargs):
        self._set_current_user_context()
        return super().__call__(*args, **kwargs)

    def _set_current_user_context(self):
        if not auth_settings.enabled:
            return

        try:
            current_user = self.request.current_user
            user.set(UserValueObject(**current_user))
        except AttributeError:
            raise AuthError(f"Authorization is enabled, but current_user is not provided inside '{self.name}' task")

        try:
            self.session_id = self.request.sessionId
        except AttributeError:
            raise AuthError(f"Authorization is enabled, but sessionId is not provided inside '{self.name}' task")


def make_celery(
    broker_url: str,
    result_backend: str,
    task_acks_late: bool,
    task_reject_on_worker_lost: bool,
    task_track_started: bool,
) -> Celery:
    celery_app = Celery(
        "app_celery",
        broker=broker_url,
        backend=result_backend,
        task_acks_late=task_acks_late,
        task_reject_on_worker_lost=task_reject_on_worker_lost,
        task_track_started=task_track_started,
        broker_connection_retry_on_startup=True,
        task_cls=BaseTask,
        include=("market_alerts.entrypoints.llm_backend.tasks",),
    )
    celery_app.conf.result_expires = timedelta(hours=1)
    celery_app.conf.result_extended = True
    return celery_app
