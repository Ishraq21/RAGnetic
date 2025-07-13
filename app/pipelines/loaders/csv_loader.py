import pandas as pd
import logging # Added import
import os
from pathlib import Path # Added import
from typing import List
from langchain_core.documents import Document

logger = logging.getLogger(__name__) # Added logger initialization

# --- Configuration for Allowed Data Directories (copied for consistency) ---
# IMPORTANT: Adjust these paths to correctly reflect where your 'data' and 'agents_data'
# directories are located relative to your project's root when the application runs.
# os.getcwd() assumes the script is run from the project root.
_PROJECT_ROOT = Path(os.getcwd()) # This should be your RAGnetic project's base directory
_ALLOWED_DATA_DIRS = [
    _PROJECT_ROOT / "data",
    _PROJECT_ROOT / "agents_data" # If agent configs or related files can be loaded via 'local' source type
    # Add any other directories that are explicitly allowed for local data sources
]
# Resolve all allowed directories to their absolute, canonical form once at startup
_ALLOWED_DATA_DIRS_RESOLVED = [d.resolve() for d in _ALLOWED_DATA_DIRS]
logger.info(f"Configured allowed data directories for CSV loader: {[str(d) for d in _ALLOWED_DATA_DIRS_RESOLVED]}")
# --- End Configuration ---

def _is_path_safe_and_within_allowed_dirs(input_path: str) -> Path:
    """
    Validates if the input_path resolves to a location within the configured
    allowed data directories. Raises ValueError if unsafe.
    Returns the resolved absolute Path if safe.
    """
    resolved_path = Path(input_path).resolve() # Resolve '..' and get absolute path

    is_safe = False
    for allowed_dir in _ALLOWED_DATA_DIRS_RESOLVED:
        if resolved_path.is_relative_to(allowed_dir):
            is_safe = True
            break

    if not is_safe:
        raise ValueError(f"Attempted to access path outside allowed directories: {resolved_path}")

    return resolved_path

def load(file_path: str) -> List[Document]:
    """
    Loads a CSV file and creates a well-formatted Document for each row,
    with path safety validation and standardized error logging.
    """
    docs = []
    try:
        # First, validate the file_path itself
        safe_file_path = _is_path_safe_and_within_allowed_dirs(file_path)

        # Validate file existence, type, and if it's a file
        if not safe_file_path.exists():
            logger.error(f"Error: CSV file not found at {safe_file_path}") # Changed from print()
            return []
        if not safe_file_path.is_file():
            logger.error(f"Error: Provided path '{safe_file_path}' is not a file.")
            return []
        if safe_file_path.suffix.lower() not in ['.csv']:
            logger.error(f"Error: Provided file '{safe_file_path}' is not a .csv file.")
            return []

        logger.info(f"Attempting to load CSV file: {safe_file_path}")
        df = pd.read_csv(safe_file_path) # Use safe_file_path

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
                    "source": str(safe_file_path.resolve()), # Use resolved safe path
                    "source_type": "csv",
                    "row_number": index + 1
                }
            )
            docs.append(doc)

        logger.info(f"Loaded {len(docs)} rows from {safe_file_path.name}") # Changed from print()
        return docs
    except ValueError as e: # Catches validation errors from _is_path_safe_and_within_allowed_dirs or file checks
        logger.error(f"Security or validation error during CSV file loading: {e}")
        return []
    except pd.errors.EmptyDataError:
        logger.warning(f"CSV file {file_path} is empty. No data loaded.")
        return []
    except pd.errors.ParserError as e:
        logger.error(f"Error parsing CSV file {file_path}: {e}. Check file format.", exc_info=True)
        return []
    except Exception as e:
        logger.error(f"Error loading CSV file {file_path}: {e}", exc_info=True) # Changed from print() and added exc_info
        return []