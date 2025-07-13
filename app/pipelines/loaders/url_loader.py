import logging # Added import
import trafilatura
from typing import List
from langchain_core.documents import Document
from urllib.parse import urlparse # Added import for URL scheme validation

logger = logging.getLogger(__name__) # Added logger initialization

def load(url: str) -> List[Document]:
    """
    Loads a webpage and creates a single Document from its main content,
    with URL scheme validation and standardized error logging.
    """
    try:
        # --- Input Validation: URL Scheme ---
        parsed_url = urlparse(url)
        if parsed_url.scheme not in ['http', 'https']:
            logger.error(f"Attempted to load from unsupported URL scheme: {parsed_url.scheme} in {url}. Only 'http' and 'https' are allowed for security reasons.")
            raise ValueError("Unsupported URL scheme for security.")
        if not parsed_url.netloc: # Basic check for domain/host presence
            logger.error(f"Invalid URL format: missing domain/host in {url}.")
            raise ValueError("Invalid URL format.")

        logger.info(f"Attempting to download and extract content from URL: {url}")
        # trafilatura.fetch_url handles network errors internally, often returning None or raising exceptions
        downloaded = trafilatura.fetch_url(url)

        if not downloaded:
            logger.warning(f"No content downloaded from {url}. It might be empty, inaccessible, or an error occurred during fetch.")
            return []

        text = trafilatura.extract(downloaded)

        if text:
            doc = Document(
                page_content=text,
                metadata={
                    "source": url,
                    "source_type": "url"
                }
            )
            logger.info(f"Successfully extracted content from {url}.")
            return [doc]
        else:
            logger.warning(f"Content extraction from {url} resulted in empty text. Skipping document creation.")
            return []
    except trafilatura.errors.NonSuccessfulRequest as e: # Catch trafilatura specific errors if needed
        logger.error(f"Trafilatura failed to fetch URL {url} due to non-successful request: {e}", exc_info=True)
        return []
    except ValueError as e: # Catches validation errors we raise
        logger.error(f"Validation error for URL {url}: {e}")
        return []
    except Exception as e: # Catch any other unexpected errors
        logger.error(f"An unexpected error occurred while loading URL {url}: {e}", exc_info=True)
        return []