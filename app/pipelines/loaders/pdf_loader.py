import fitz  # This is the PyMuPDF library
from langchain_core.documents import Document
from typing import List
import os


def load(file_path: str) -> List[Document]:
    """
    Loads a PDF file and creates a Document for each page.
    """
    if not os.path.exists(file_path):
        print(f"Error: File not found at {file_path}")
        return []

    pdf_document = fitz.open(file_path)
    docs = []
    for page_num, page in enumerate(pdf_document):
        text = page.get_text()
        if text:  # Only create a document if there's text on the page
            doc = Document(
                page_content=text,
                metadata={
                    "source": os.path.abspath(file_path),
                    "source_type": "pdf",
                    "page_number": page_num + 1,
                }
            )
            docs.append(doc)

    print(f"Loaded {len(docs)} pages from {os.path.basename(file_path)}")
    return docs