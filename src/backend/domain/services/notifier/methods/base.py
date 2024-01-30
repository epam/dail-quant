from abc import ABC, abstractmethod


class NotificationMethod(ABC):
    METHOD_NAME = None

    @abstractmethod
    def execute(self) -> None:
        pass
