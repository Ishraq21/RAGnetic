import docx
from langchain_core.documents import Document
from typing import List
import os

def load(file_path: str) -> List[Document]:
    """
    Loads a .docx file and creates a single Document from its content.
    """
    if not os.path.exists(file_path):
        print(f"Error: File not found at {file_path}")
        return []

    try:
        document = docx.Document(file_path)
        full_text = "\n".join([para.text for para in document.paragraphs])

        if full_text.strip():
            doc = Document(
                page_content=full_text,
                metadata={
                    "source": os.path.abspath(file_path),
                    "source_type": "docx"
                }
            )
            print(f"Loaded content from {os.path.basename(file_path)}")
            return [doc]
        return []
    except Exception as e:
        print(f"Error loading .docx file {file_path}: {e}")
        return []