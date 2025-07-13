import logging
from typing import List, Optional # Added Optional for ModelParams
from langchain_community.agent_toolkits import SQLDatabaseToolkit
from langchain_community.utilities import SQLDatabase
from langchain_openai import ChatOpenAI # Still needed for type hinting, but LLM initialized via get_llm_model
from sqlalchemy import create_engine, text
from sqlalchemy.exc import SQLAlchemyError
from langchain_core.tools import BaseTool

from app.core.config import get_llm_model
from app.schemas.agent import ModelParams # Added import for ModelParams

logger = logging.getLogger(__name__)

def create_sql_toolkit(
    db_connection_string: str,
    llm_model_name: str,
    llm_model_params: Optional[ModelParams] = None
) -> List[BaseTool]:
    """
    Creates and returns the tools from the SQLDatabaseToolkit,
    with database connection string validation and robust error handling.
    The LLM used by the toolkit is now configurable.
    """
    try:
        if not db_connection_string:
            logger.error("Validation Error: Database connection string is required for SQL toolkit.")
            return []

        supported_dialects = ["sqlite://", "postgresql://", "mysql://", "oracle://", "mssql+pyodbc://", "mongodb://"]
        if not any(db_connection_string.startswith(d) for d in supported_dialects):
            logger.error(f"Validation Error: Unsupported or invalid database connection string dialect for SQL toolkit: {db_connection_string[:50]}...")
            raise ValueError("Unsupported or invalid database connection string dialect.")

        logger.info(f"Attempting to create SQLAlchemy engine for database: {db_connection_string[:20]}...")
        engine = create_engine(db_connection_string)

        with engine.connect() as connection:
            connection.execute(text("SELECT 1"))
        logger.info("Successfully established test connection to the database.")

        db = SQLDatabase(engine)

        # --- LLM Initialization Consistency (now uses passed parameters) ---
        llm = get_llm_model(
            model_name=llm_model_name,
            model_params=llm_model_params
        )

        if llm is None:
            logger.error(f"Failed to initialize LLM '{llm_model_name}' for SQLDatabaseToolkit. Cannot create toolkit.")
            return []

        logger.info(f"Initializing SQLDatabaseToolkit with LLM: {llm_model_name}.")
        toolkit = SQLDatabaseToolkit(db=db, llm=llm)
        tools = toolkit.get_tools()
        logger.info(f"Successfully created {len(tools)} SQL tools.")
        return tools

    except SQLAlchemyError as e:
        logger.error(f"Database error during SQL toolkit creation: {e}", exc_info=True)
        return []
    except ValueError as e:
        logger.error(f"SQL Toolkit Validation or LLM Error: {e}")
        return []
    except Exception as e:
        logger.error(f"An unexpected error occurred during SQL toolkit creation: {e}", exc_info=True)
        return []