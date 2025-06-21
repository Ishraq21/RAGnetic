from langchain_community.document_loaders import GitLoader
from typing import List
from langchain_core.documents import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter, Language
import os
from pathlib import Path


def load(repo_path: str) -> List[Document]:
    """
    Loads files from a local Git repository path and splits them into
    syntax-aware chunks based on their programming language.
    """
    if not os.path.isdir(repo_path):
        print(f"Error: Provided path '{repo_path}' is not a valid directory.")
        return []

    # A mapping from file extension to LangChain Language enum
    EXTENSION_MAP = {
        ".py": Language.PYTHON,
        ".js": Language.JS,
        ".ts": Language.TS,
        ".java": Language.JAVA,
        ".c": Language.C,
        ".cpp": Language.CPP,
        ".h": Language.CPP,
        ".cs": Language.CSHARP,
        ".go": Language.GO,
        ".rs": Language.RUST,
        ".php": Language.PHP,
        ".rb": Language.RUBY,
        ".swift": Language.SWIFT,
        ".kt": Language.KOTLIN,
        ".scala": Language.SCALA,
        ".md": Language.MARKDOWN,
        ".html": Language.HTML,
    }

    def file_filter(file_path):
        return Path(file_path).suffix in EXTENSION_MAP

    try:
        # Step 1: Load the raw documents from the Git repository
        loader = GitLoader(
            repo_path=repo_path,
            file_filter=file_filter
        )
        raw_docs = loader.load()

        # Step 2: Split the raw documents into syntax-aware chunks
        chunked_docs = []
        for doc in raw_docs:
            file_extension = Path(doc.metadata["file_path"]).suffix
            language = EXTENSION_MAP.get(file_extension)

            if language:
                # Use a language-specific splitter
                splitter = RecursiveCharacterTextSplitter.from_language(
                    language=language, chunk_size=1000, chunk_overlap=100
                )
                chunks = splitter.split_documents([doc])

                # Add our custom metadata to each new chunk
                for chunk in chunks:
                    chunk.metadata['source_type'] = 'code_repository'
                    # The original 'source' (full file path) is preserved

                chunked_docs.extend(chunks)

        print(f"Loaded and chunked {len(raw_docs)} files into {len(chunked_docs)} code chunks from {repo_path}")
        return chunked_docs

    except Exception as e:
        print(f"Error loading git repository at {repo_path}: {e}")
        return []