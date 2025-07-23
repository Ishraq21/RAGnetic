import logging
import nbformat
from typing import Any, Dict, List, Type
from pydantic.v1 import BaseModel, Field

logger = logging.getLogger(__name__)

class NotebookParserInput(BaseModel):
    """Input schema for the NotebookParserTool."""
    notebook_content: str = Field(..., description="A string containing the raw JSON from a .ipynb file.")
    extract: str = Field("code", description="What to extract: 'code' for code cells, 'markdown' for text cells.")

class NotebookParserTool:
    """A tool for parsing Jupyter Notebooks (.ipynb) and extracting cell content."""
    name: str = "notebook_parser_tool"
    description: str = "Parses the JSON content of a Jupyter Notebook to extract either all code cells or all markdown cells."
    args_schema: Type[BaseModel] = NotebookParserInput

    def get_input_schema(self) -> Dict[str, Any]:
        return self.args_schema.schema()

    def run(self, notebook_content: str, extract: str = "code", **kwargs: Any) -> List[str]:
        logger.info(f"Parsing notebook to extract '{extract}' cells.")
        try:
            notebook = nbformat.reads(notebook_content, as_version=4)
            extracted_cells = []
            for cell in notebook.cells:
                if extract == "code" and cell.cell_type == 'code':
                    extracted_cells.append(cell.source)
                elif extract == "markdown" and cell.cell_type == 'markdown':
                    extracted_cells.append(cell.source)
            return extracted_cells
        except Exception as e:
            logger.error(f"Failed to parse notebook content: {e}", exc_info=True)
            return [f"Error: Failed to parse notebook content: {e}"]