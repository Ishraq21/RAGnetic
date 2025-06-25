from langchain_community.tools import ArxivQueryRun
from langchain_core.tools import BaseTool
from typing import List

def get_arxiv_tool() -> List[BaseTool]:
    """Creates and returns the ArXiv search tool."""
    return [ArxivQueryRun()]