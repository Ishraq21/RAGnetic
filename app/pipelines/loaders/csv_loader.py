import pandas as pd
import logging
import os
from pathlib import Path
from typing import List
from langchain_core.documents import Document
import asyncio

# NEW: Import get_path_settings from centralized config
from app.core.config import get_path_settings

logger = logging.getLogger(__name__)

# --- Centralized Path Configuration ---
_PATH_SETTINGS = get_path_settings()
_PROJECT_ROOT_FROM_CONFIG = _PATH_SETTINGS["PROJECT_ROOT"] # Store project root if needed
_ALLOWED_DATA_DIRS_RESOLVED = _PATH_SETTINGS["ALLOWED_DATA_DIRS"] # Store resolved allowed dirs
logger.info(f"Loaded allowed data directories for CSV loader from central config: {[str(d) for d in _ALLOWED_DATA_DIRS_RESOLVED]}")
# --- End Centralized Configuration ---


def _is_path_safe_and_within_allowed_dirs(input_path: str) -> Path:
    """
    Validates if the input_path resolves to a location within the configured
    allowed data directories. Raises ValueError if unsafe.
    Returns the resolved absolute Path if safe.
    """
    resolved_path = Path(input_path).resolve()

    is_safe = False
    for allowed_dir in _ALLOWED_DATA_DIRS_RESOLVED: # This variable now comes from central config
        if resolved_path.is_relative_to(allowed_dir):
            is_safe = True
            break

    if not is_safe:
        raise ValueError(f"Attempted to access path outside allowed directories: {resolved_path}")

    return resolved_path


async def load(file_path: str) -> List[Document]:
    """
    Loads a CSV file and creates a well-formatted Document for each row,
    with path safety validation and standardized error logging.
    Now supports asynchronous loading.
    """
    docs = []
    try:
        # First, validate the file_path itself
        safe_file_path = _is_path_safe_and_within_allowed_dirs(file_path)

        if not safe_file_path.exists():
            logger.error(f"Error: CSV file not found at {safe_file_path}")
            return []
        if not safe_file_path.is_file():
            logger.error(f"Error: Provided path '{safe_file_path}' is not a file.")
            return []
        if safe_file_path.suffix.lower() not in ['.csv']:
            logger.error(f"Error: Provided file '{safe_file_path}' is not a .csv file.")
            return []

        logger.info(f"Attempting to load CSV file: {safe_file_path}")

        # MODIFIED: Run pd.read_csv in a separate thread because it's blocking I/O
        def _read_csv_blocking():
            return pd.read_csv(safe_file_path)

        df = await asyncio.to_thread(_read_csv_blocking)

        for index, row in df.iterrows():
            # Use the first column's value as a title, e.g., the customer's name or ID
            title_col = df.columns[0]
            title_val = row[title_col]

            # Format each column-value pair on a new line
            row_details = "\n".join([f"- {str(col).replace('_', ' ').strip()}: {val}" for col, val in row.items()])

            page_content = f"Record for {title_col} '{title_val}':\n{row_details}"

            doc = Document(
                page_content=page_content,
                metadata={
                    "source": str(safe_file_path.resolve()),
                    "source_type": "csv",
                    "row_number": index + 1
                }
            )
            docs.append(doc)

        logger.info(f"Loaded {len(docs)} rows from {safe_file_path.name}")
        return docs
    except ValueError as e:
        logger.error(f"Security or validation error during CSV file loading: {e}")
        return []
    except pd.errors.EmptyDataError:
        logger.warning(f"CSV file {file_path} is empty. No data loaded.")
        return []
    except pd.errors.ParserError as e:
        logger.error(f"Error parsing CSV file {file_path}: {e}. Check file format.", exc_info=True)
        return []
    except Exception as e:
        logger.error(f"Error loading CSV file {file_path}: {e}", exc_info=True)
        return []