from typing import List
from langchain_community.agent_toolkits import SQLDatabaseToolkit
from langchain_community.utilities import SQLDatabase
from langchain_openai import ChatOpenAI
from sqlalchemy import create_engine
from langchain_core.tools import BaseTool


def create_sql_toolkit(db_connection_string: str) -> List[BaseTool]:
    """Creates and returns the tools from the SQLDatabaseToolkit."""
    engine = create_engine(db_connection_string)
    db = SQLDatabase(engine)

    # The toolkit needs an LLM to function (e.g., for the QueryCheckerTool)
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

    toolkit = SQLDatabaseToolkit(db=db, llm=llm)
    return toolkit.get_tools()