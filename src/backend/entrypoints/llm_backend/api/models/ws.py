from pydantic import BaseModel


class WSAuthTicketResponseModel(BaseModel):
    ticket_id: str
    task_id: str
    creation_timestamp: str
