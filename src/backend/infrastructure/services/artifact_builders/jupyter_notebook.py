import json
import os
from typing import Any, Dict

import nbformat
from nbformat import NotebookNode

from market_alerts.containers import trade_fill_prices
from market_alerts.domain.constants import RESOURCES_PATH
from market_alerts.domain.exceptions import JupyterError, LLMBadResponseError
from market_alerts.infrastructure.services.code import (
    are_all_lines_comments_or_empty,
    get_code_sections,
)
from market_alerts.utils import ms_to_string

from .base import BaseModelExportArtifactBuilder


class JupyterNotebookModelExportArtifactBuilder(BaseModelExportArtifactBuilder):
    FETCH_DATA_SETTINGS_CELL_IDX = 3
    INDICATORS_SECTION_START_CELL_IDX = 6
    INDICATORS_SETTINGS_CELL_IDX = 7
    BACKTESTING_SECTION_START_CELL_IDX = 10
    BACKTESTING_SETTINGS_CELL_IDX = 11
    NOTEBOOK_TEMPLATE_FILE_NAME = "notebook_template.ipynb"

    def build_artifact(self, data: Dict[str, Any]) -> Dict[str, Any]:
        with open(os.path.join(RESOURCES_PATH, self.NOTEBOOK_TEMPLATE_FILE_NAME), "r") as f:
            template_notebook = nbformat.read(f, as_version=4)

        self._format_fetch_data_cell(template_notebook, data)

        try:
            code_sections = get_code_sections(data["indicators_logic"], tabs=1)
            data["indicators_code"] = code_sections[0]
            self._format_indicators_settings_cell(template_notebook, data)
            data["trading_code"] = code_sections[1]
            if are_all_lines_comments_or_empty(data["trading_code"]):
                raise ValueError
            self._format_backtesting_settings_cell(template_notebook, data)
        except LLMBadResponseError:
            template_notebook.cells = template_notebook.cells[: self.INDICATORS_SECTION_START_CELL_IDX]
        except (IndexError, ValueError):
            template_notebook.cells = template_notebook.cells[: self.BACKTESTING_SECTION_START_CELL_IDX]

        return {
            "content": json.loads(nbformat.writes(template_notebook)),
            "format": "json",
            "type": "notebook",
        }

    def _format_fetch_data_cell(self, template_notebook: NotebookNode, data: Dict[str, Any]) -> None:
        cell = template_notebook.cells[self.FETCH_DATA_SETTINGS_CELL_IDX]
        data["fill_trade_price"] = trade_fill_prices.get_trade_price_by_backend_key(data["fill_trade_price"]).BACKEND_KEY
        data["time_period"] = ms_to_string(data["time_period"])
        data["interval"] = ms_to_string(data["interval"])
        self._format_cell(cell, data)

    def _format_indicators_settings_cell(self, template_notebook: NotebookNode, data: Dict[str, Any]) -> None:
        cell = template_notebook.cells[self.INDICATORS_SETTINGS_CELL_IDX]
        self._format_cell(cell, data)

    def _format_backtesting_settings_cell(self, template_notebook: NotebookNode, data: Dict[str, Any]) -> None:
        cell = template_notebook.cells[self.BACKTESTING_SETTINGS_CELL_IDX]
        self._format_cell(cell, data)

    @staticmethod
    def _format_cell(cell, data: Dict[str, Any]) -> None:
        if cell.cell_type == "code":
            cell.source = cell.source.format(**data)
        else:
            raise JupyterError("Error when interpolating jupyter notebook template")
