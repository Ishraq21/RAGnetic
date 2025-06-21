from pathlib import Path
from typing import List
from langchain_core.documents import Document

def load(folder_path: str) -> List[Document]:
    """
    Loads all files from a local directory and creates a Document for each.
    """

    p = Path(folder_path)
    if not p.is_dir():
        print(f"Error: Provided path '{folder_path}' is not a valid directory.")
        return []

    docs = []
    for file in Path(folder_path).rglob("*.*"):
        if file.is_dir():
            continue
        try:
            with open(file, "r", encoding="utf-8") as f:
                text = f.read()
                doc = Document(
                    page_content=text,
                    metadata={
                        "source": str(file.resolve()),
                        "source_type": "local_directory"
                    }
                )
                docs.append(doc)
        except Exception as e:
            print(f"Error reading {file}: {e}")
    return docs