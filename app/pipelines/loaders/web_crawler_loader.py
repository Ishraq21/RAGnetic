import logging # Added import
import trafilatura
from typing import List
from langchain_core.documents import Document
from langchain_community.document_loaders.recursive_url_loader import RecursiveUrlLoader
from bs4 import BeautifulSoup as Soup # Imported for RecursiveUrlLoader if it's used with it
from urllib.parse import urlparse # Added import for URL scheme validation

logger = logging.getLogger(__name__) # Added logger initialization

def _trafilatura_extractor(html: str) -> str:
    """A custom extractor function that uses trafilatura to get main text content."""
    # Fallback to empty string if extraction fails
    # Add logging for extraction failures if trafilatura.extract raises exceptions or returns None for valid input
    try:
        extracted_text = trafilatura.extract(html)
        return extracted_text or ""
    except Exception as e:
        logger.warning(f"Error during Trafilatura HTML extraction: {e}", exc_info=True)
        return ""


def load(url: str, max_depth: int = 2) -> List[Document]:
    """
    Crawls a website starting from the given URL up to a max_depth,
    ingesting the content of each page found, with URL validation and proper logging.
    """
    try:
        # --- Input Validation: URL Scheme ---
        if not url:
            logger.error("Error: A starting URL is required for the web crawler.") # Changed from print()
            return []

        parsed_url = urlparse(url)
        if parsed_url.scheme not in ['http', 'https']:
            logger.error(f"Attempted to crawl unsupported URL scheme: {parsed_url.scheme} in {url}. Only 'http' and 'https' are allowed for security reasons.")
            raise ValueError("Unsupported URL scheme for web crawling.")
        if not parsed_url.netloc: # Basic check for domain/host presence
            logger.error(f"Invalid URL format for crawling: missing domain/host in {url}.")
            raise ValueError("Invalid URL format for crawling.")

        logger.info(f"Starting web crawl from URL: {url} with max_depth: {max_depth}")

        # Initialize the loader with our custom text extractor and settings
        loader = RecursiveUrlLoader(
            url=url,
            max_depth=max_depth,
            extractor=_trafilatura_extractor,
            prevent_outside=True,  # IMPORTANT: Prevents crawling external sites
            use_async=True,  # Uses asyncio for faster crawling
            timeout=60,  # Timeout for each request
            check_response_status=True
        )

        docs = loader.load()

        # Add our consistent source_type metadata to each document
        for doc in docs:
            doc.metadata['source_type'] = 'web_crawler'

        logger.info(f"Crawled and loaded {len(docs)} pages starting from {url}") # Changed from print()
        return docs

    except ValueError as e: # Catch validation errors we raise
        logger.error(f"Validation error for web crawling URL {url}: {e}")
        return []
    except Exception as e:
        logger.error(f"An error occurred while crawling website {url}: {e}", exc_info=True) # Changed from print()
        return []