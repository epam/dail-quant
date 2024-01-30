import json
import tempfile
from typing import Any, Dict

from .base import BaseModelExportArtifactBuilder


class JSONFileModelExportArtifactBuilder(BaseModelExportArtifactBuilder):
    JSON_INDENT = 4

    def build_artifact(self, data: Dict[str, Any]) -> str:
        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as temp_file:
            temp_file.write(json.dumps(data, indent=self.JSON_INDENT))
            return temp_file.name
