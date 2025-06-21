from langchain_community.document_loaders import GitLoader
from typing import List
from langchain_core.documents import Document
import os


def load(repo_path: str) -> List[Document]:
    """
    Loads all files from a local Git repository path.
    Each file is treated as a single Document.
    """
    if not os.path.isdir(repo_path):
        print(f"Error: Provided path '{repo_path}' is not a valid directory.")
        return []

    # Define a filter to include only certain file extensions
    # and exclude common non-code files.
    def file_filter(file_path):
        supported_extensions = {".py", ".js", ".ts", ".java", ".c", ".cpp", ".h", ".cs", ".go", ".rs", ".php", ".rb",
                                ".swift", ".kt", ".scala", ".md", ".json", ".yml", ".yaml", ".html", ".css"}
        return Path(file_path).suffix in supported_extensions

    try:
        # Use LangChain's GitLoader to handle the repository logic
        # It automatically respects .gitignore and is efficient.
        loader = GitLoader(
            repo_path=repo_path,
            file_filter=file_filter
        )

        docs = loader.load()

        # Add our custom metadata to each loaded document
        for doc in docs:
            doc.metadata['source_type'] = 'code_repository'

        print(f"Loaded {len(docs)} code files from {repo_path}")
        return docs

    except Exception as e:
        print(f"Error loading git repository at {repo_path}: {e}")
        return []


# We need pathlib for the file_filter
from pathlib import Path