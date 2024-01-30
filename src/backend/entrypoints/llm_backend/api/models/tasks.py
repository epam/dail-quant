from pydantic import BaseModel


class TaskIDRequestModel(BaseModel):
    task_id: str
