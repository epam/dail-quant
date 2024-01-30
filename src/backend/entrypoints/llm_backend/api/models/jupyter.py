from pydantic import BaseModel


class JupyterLinkResponseModel(BaseModel):
    link: str
