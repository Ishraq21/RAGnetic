import trafilatura
from typing import List
from langchain_core.documents import Document

def load(url: str) -> List[Document]:
    """
    Loads a webpage and creates a single Document from its main content.
    """
    downloaded = trafilatura.fetch_url(url)
    text = trafilatura.extract(downloaded) if downloaded else ""

    if text:
        doc = Document(
            page_content=text,
            metadata={
                "source": url,
                "source_type": "url"
            }
        )
        return [doc]
    return []