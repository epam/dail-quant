from abc import ABC, abstractmethod
from typing import Any, Dict


class BaseModelExportArtifactBuilder(ABC):
    @abstractmethod
    def build_artifact(self, data: Dict[str, Any]) -> Any:
        pass
