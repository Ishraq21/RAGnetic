import logging
from typing import Any, Dict, List, Type
from pydantic.v1 import BaseModel, Field
from sql_metadata import Parser

logger = logging.getLogger(__name__)

class SQLParserInput(BaseModel):
    """Input schema for the SQLParserTool."""
    sql_query: str = Field(..., description="A string containing the SQL query to be analyzed.")

class SQLParserTool:
    """A tool for parsing SQL queries to extract metadata like table and column names."""
    name: str = "sql_parser_tool"
    description: str = (
        "Parses a SQL query string to identify all tables and columns involved."
    )
    args_schema: Type[BaseModel] = SQLParserInput

    def get_input_schema(self) -> Dict[str, Any]:
        """Returns the JSON schema for the tool's input."""
        return self.args_schema.schema()

    def run(self, sql_query: str, **kwargs: Any) -> Dict[str, List[str]]:
        """Parses the SQL query and extracts table and column information."""
        logger.info(f"Parsing SQL query...")
        try:
            parser = Parser(sql_query)
            return {
                "tables_read": parser.tables,
                "tables_written": parser.tables_aliases, # More complex logic may be needed for inserts/updates
                "columns": parser.columns,
            }
        except Exception as e:
            logger.error(f"Failed to parse SQL query: {e}", exc_info=True)
            return {"error": f"Failed to parse SQL query: {e}"}