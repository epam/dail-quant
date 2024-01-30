import contextvars
from dataclasses import dataclass


@dataclass
class UserValueObject:
    email: str
    token: str
    is_admin: bool


user: contextvars.ContextVar[UserValueObject] = contextvars.ContextVar("user")
