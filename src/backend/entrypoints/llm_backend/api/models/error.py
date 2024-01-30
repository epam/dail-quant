from pydantic import BaseModel


class ErrorModel(BaseModel):
    code: str
    message: str
