from langchain_community.document_loaders.recursive_url_loader import RecursiveUrlLoader
from bs4 import BeautifulSoup as Soup
from typing import List
from langchain_core.documents import Document
import trafilatura


def _trafilatura_extractor(html: str) -> str:
    """A custom extractor function that uses trafilatura to get main text content."""
    # Fallback to empty string if extraction fails
    return trafilatura.extract(html) or ""


def load(url: str, max_depth: int = 2) -> List[Document]:
    """
    Crawls a website starting from the given URL up to a max_depth,
    ingesting the content of each page found.
    """
    if not url:
        print("Error: A starting URL is required for the web crawler.")
        return []

    try:
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

        print(f"Crawled and loaded {len(docs)} pages starting from {url}")
        return docs

    except Exception as e:
        print(f"An error occurred while crawling website {url}: {e}")
        return []