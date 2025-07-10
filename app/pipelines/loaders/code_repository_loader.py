import os
import tempfile
import logging
import subprocess
from typing import List, Optional
from pathlib import Path

from git import Repo
from langchain_community.document_loaders import GitLoader
from langchain_core.documents import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter, Language

logger = logging.getLogger(__name__)


def load(repo_path: str) -> List[Document]:
    """
    Smart loader that handles both remote GitHub URLs and local file paths.
    - URLs get cloned to a temp dir, default branch auto-detected.
    - Local paths are processed in-place (uses whatever branch is checked out).
    """
    if repo_path.startswith(("http://", "https://")):
        with tempfile.TemporaryDirectory() as temp_dir:
            try:
                logger.info(f"Cloning {repo_path} → {temp_dir}")
                repo = Repo.clone_from(repo_path, to_path=temp_dir)
                default_branch = _get_default_branch(temp_dir)
                if not default_branch:
                    logger.error(f"Can't determine default branch for {repo_path}")
                    return []
                logger.info(f"Default branch detected: '{default_branch}'")
                return _load_and_chunk(temp_dir, branch=default_branch)

            except Exception as e:
                logger.error(f"Clone/process failed for {repo_path}: {e}", exc_info=True)
                return []
    else:
        # Local filesystem path
        return _load_and_chunk(repo_path, branch=None)


def _get_default_branch(repo_path: str) -> Optional[str]:
    """
    Run `git symbolic-ref refs/remotes/origin/HEAD` to discover the default branch name.
    Returns e.g. 'main' or 'master', or None on error.
    """
    try:
        result = subprocess.run(
            ["git", "symbolic-ref", "refs/remotes/origin/HEAD"],
            cwd=repo_path,
            capture_output=True,
            text=True,
            check=True
        )
        # 'refs/remotes/origin/main' → ['refs', 'remotes', 'origin', 'main']
        return result.stdout.strip().split("/")[-1]
    except Exception as e:
        logger.error(f"Failed to determine default branch in {repo_path}: {e}")
        return None


def _load_and_chunk(repo_path: str, branch: Optional[str]) -> List[Document]:
    """
    Load via GitLoader (if branch provided, checkout that branch first),
    then split all files matching EXTENSION_MAP into code-language chunks.
    """
    if not os.path.isdir(repo_path):
        logger.error(f"'{repo_path}' is not a directory.")
        return []

    EXTENSION_MAP = {
        ".py": Language.PYTHON, ".js": Language.JS, ".ts": Language.TS,
        ".java": Language.JAVA, ".c": Language.C, ".cpp": Language.CPP,
        ".h": Language.CPP, ".cs": Language.CSHARP, ".go": Language.GO,
        ".rs": Language.RUST, ".php": Language.PHP, ".rb": Language.RUBY,
        ".swift": Language.SWIFT, ".kt": Language.KOTLIN, ".scala": Language.SCALA,
        ".md": Language.MARKDOWN, ".html": Language.HTML,
    }

    def file_filter(path: str) -> bool:
        return Path(path).suffix in EXTENSION_MAP

    try:
        loader_kwargs = {"repo_path": repo_path, "file_filter": file_filter}
        if branch:
            loader_kwargs["branch"] = branch

        loader = GitLoader(**loader_kwargs)
        raw_docs = loader.load()
        chunked = []

        for doc in raw_docs:
            ext = Path(doc.metadata["file_path"]).suffix
            lang = EXTENSION_MAP.get(ext)
            if not lang:
                continue

            splitter = RecursiveCharacterTextSplitter.from_language(
                language=lang, chunk_size=1000, chunk_overlap=100
            )
            for chunk in splitter.split_documents([doc]):
                chunk.metadata["source_type"] = "code_repository"
                chunked.append(chunk)

        logger.info(f"Chunked {len(raw_docs)} files into {len(chunked)} code chunks.")
        return chunked

    except Exception as e:
        logger.error(f"Error loading '{repo_path}': {e}", exc_info=True)
        return []
