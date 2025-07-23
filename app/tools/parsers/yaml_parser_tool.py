import logging
import yaml
from typing import Any, Dict, List, Type
from pydantic.v1 import BaseModel, Field

logger = logging.getLogger(__name__)

class YAMLParserInput(BaseModel):
    """Input schema for the YAMLParserTool."""
    yaml_string: str = Field(..., description="A string containing the YAML content to be parsed.")
    query: str = Field(..., description="A query to extract specific data from the YAML, using dot notation (e.g., 'metadata.name').")

class YAMLParserTool:
    """A tool for parsing YAML strings and extracting specific values."""
    name: str = "yaml_parser_tool"
    description: str = "Parses a YAML string and extracts a value based on a dot-notation query."
    args_schema: Type[BaseModel] = YAMLParserInput

    def get_input_schema(self) -> Dict[str, Any]:
        return self.args_schema.schema()

    def run(self, yaml_string: str, query: str, **kwargs: Any) -> Any:
        logger.info(f"Parsing YAML to find data at path: '{query}'")
        try:
            data = yaml.safe_load(yaml_string)
            keys = query.split('.')
            value = data
            for key in keys:
                if isinstance(value, dict):
                    value = value.get(key)
                else:
                    return {"error": f"Path '{query}' not found in YAML."}
            return value
        except Exception as e:
            logger.error(f"Failed to parse YAML: {e}", exc_info=True)
            return {"error": f"Failed to parse YAML: {e}"}