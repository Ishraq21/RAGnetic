import logging
import os
from typing import Any, Dict, List, Type
from pydantic.v1 import BaseModel, Field
from lark import Lark, Transformer, v_args

logger = logging.getLogger(__name__)


# Load the grammar from the file. Lark handles the parsing of the grammar itself.
_grammar_path = os.path.join(os.path.dirname(__file__), 'python.lark')
with open(_grammar_path) as f:
    _python_grammar = f.read()


class _FunctionFinder(Transformer):
    """A Lark Transformer that walks the parse tree and collects function names."""

    def __init__(self):
        self.functions = []

    @v_args(inline=True)
    def function_def(self, name):
        # This method is automatically called by Lark when it finds a 'function_def' rule
        self.functions.append(name.value)
        return name


class CodeParserInput(BaseModel):
    """Input schema for the CodeParserTool."""
    code_string: str = Field(..., description="A string containing the source code to be analyzed.")
    language: str = Field("python", description="The programming language of the code (must be 'python').")


class CodeParserTool:
    """A tool for parsing source code to extract structural elements like function definitions."""
    name: str = "code_parser_tool"
    description: str = "Parses a string of Python source code to find and list all function definitions."
    args_schema: Type[BaseModel] = CodeParserInput

    def get_input_schema(self) -> Dict[str, Any]:
        """Returns the JSON schema for the tool's input."""
        return self.args_schema.schema()

    def run(self, code_string: str, language: str = "python", **kwargs: Any) -> List[str]:
        """Parses the source code and extracts function names using Lark."""
        if language != 'python':
            return [f"Error: Language '{language}' is not supported by this tool."]

        logger.info(f"Parsing {language} code to find function definitions...")
        try:
            # Create a Lark parser instance with our grammar
            parser = Lark(_python_grammar, start='program')
            tree = parser.parse(code_string)

            # Use our custom transformer to walk the tree and find the functions
            finder = _FunctionFinder()
            finder.transform(tree)

            logger.info(f"Found {len(finder.functions)} function definitions.")
            return finder.functions

        except Exception as e:
            logger.error(f"Failed to parse code with Lark: {e}", exc_info=True)
            return [f"Error: Failed to parse code: {e}"]
