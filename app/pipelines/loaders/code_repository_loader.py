import os
import tempfile
import logging
import subprocess
from typing import List, Optional
from pathlib import Path
from urllib.parse import urlparse
import asyncio  # NEW: Added import for asynchronous operations

from git import Repo, InvalidGitRepositoryError, NoSuchPathError, GitCommandError
from langchain_community.document_loaders import GitLoader
from langchain_core.documents import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter, Language

logger = logging.getLogger(__name__)

# --- Configuration for Allowed Data Directories ---
# IMPORTANT: Adjust these paths to correctly reflect where your 'data' and 'agents_data'
# directories are located relative to your project's root when the application runs.
# os.getcwd() assumes the script is run from the project root.
_PROJECT_ROOT = Path(os.getcwd())
# Define a specific, allowed temporary directory for repository clones
_CLONE_TEMP_DIR = _PROJECT_ROOT / ".ragnetic_temp_clones"
_ALLOWED_DATA_DIRS = [
    _PROJECT_ROOT / "data",
    _PROJECT_ROOT / "agents_data",
    _CLONE_TEMP_DIR
]
# Resolve all allowed directories to their absolute, canonical form once at startup
_ALLOWED_DATA_DIRS_RESOLVED = [d.resolve() for d in _ALLOWED_DATA_DIRS]
logger.info(
    f"Configured allowed data directories for code repository loader: {[str(d) for d in _ALLOWED_DATA_DIRS_RESOLVED]}")


# --- End Configuration ---

def _is_path_safe_and_within_allowed_dirs(input_path: str) -> Path:
    """
    Validates if the input_path resolves to a location within the configured
    allowed data directories. Raises ValueError if unsafe.
    Returns the resolved absolute Path if safe.
    """
    resolved_path = Path(input_path).resolve()

    is_safe = False
    for allowed_dir in _ALLOWED_DATA_DIRS_RESOLVED:
        if resolved_path.is_relative_to(allowed_dir):
            is_safe = True
            break

    if not is_safe:
        raise ValueError(f"Attempted to access path outside allowed directories: {resolved_path}")

    return resolved_path


async def load(repo_path: str) -> List[Document]:  # MODIFIED: async def
    """
    Smart loader that handles both remote GitHub URLs and local file paths.
    Includes input validation for both types and robust error handling.
    Now supports asynchronous loading.
    """
    try:
        if not repo_path:
            logger.error("Validation Error: Repository path (URL or local) is required.")
            return []

        if repo_path.startswith(("http://", "https://", "git@")):
            # --- Remote URL Validation ---
            if repo_path.startswith(("http://", "https://")):
                parsed_url = urlparse(repo_path)
                if parsed_url.scheme not in ['http', 'https']:
                    logger.error(f"Unsupported URL scheme for code repository: {parsed_url.scheme} in {repo_path}.")
                    raise ValueError("Only 'http', 'https', or 'git@' URLs are allowed for security.")
                if not parsed_url.netloc:
                    logger.error(f"Invalid URL format for code repository: missing domain/host in {repo_path}.")
                    raise ValueError("Invalid URL format.")

            os.makedirs(_CLONE_TEMP_DIR, exist_ok=True)

            with tempfile.TemporaryDirectory(dir=str(_CLONE_TEMP_DIR)) as temp_dir:
                try:
                    safe_temp_dir = _is_path_safe_and_within_allowed_dirs(temp_dir)

                    logger.info(f"Cloning {repo_path} → {safe_temp_dir}")
                    # MODIFIED: Run Repo.clone_from in a separate thread
                    repo = await asyncio.to_thread(Repo.clone_from, repo_path, to_path=str(safe_temp_dir))

                    # MODIFIED: Await the async _get_default_branch
                    default_branch = await _get_default_branch(str(safe_temp_dir))
                    if not default_branch:
                        logger.error(f"Can't determine default branch for {repo_path}. Skipping.")
                        return []
                    logger.info(f"Default branch detected: '{default_branch}' for {repo_path}.")

                    # MODIFIED: Await the async _load_and_chunk
                    return await _load_and_chunk(str(safe_temp_dir), branch=default_branch)

                except (GitCommandError, InvalidGitRepositoryError, NoSuchPathError) as git_e:
                    logger.error(f"Git operation failed for {repo_path}: {git_e}", exc_info=True)
                    return []
                except Exception as e:
                    logger.error(f"An unexpected error during cloning/processing {repo_path}: {e}", exc_info=True)
                    return []
        else:
            # --- Local filesystem path Validation ---
            logger.info(f"Processing local code repository at: {repo_path}")
            safe_local_repo_path = _is_path_safe_and_within_allowed_dirs(repo_path)
            # MODIFIED: Await the async _load_and_chunk
            return await _load_and_chunk(str(safe_local_repo_path), branch=None)

    except ValueError as e:
        logger.error(f"Code repository loader validation error: {e}")
        return []
    except Exception as e:
        logger.error(f"An unexpected error occurred in code_repository_loader: {e}", exc_info=True)
        return []


async def _get_default_branch(repo_path: str) -> Optional[str]:  # MODIFIED: async def
    """
    Run `git symbolic-ref refs/remotes/origin/HEAD` to discover the default branch name.
    Now supports asynchronous execution.
    """
    try:
        # MODIFIED: Run subprocess.run in a separate thread
        result = await asyncio.to_thread(subprocess.run,
                                         ["git", "symbolic-ref", "refs/remotes/origin/HEAD"],
                                         cwd=repo_path,
                                         capture_output=True,
                                         text=True,
                                         check=True
                                         )
        return result.stdout.strip().split("/")[-1]
    except subprocess.CalledProcessError as sub_e:
        logger.warning(f"Git command failed to determine default branch in {repo_path}: {sub_e.stderr.strip()}",
                       exc_info=True)
        return None
    except Exception as e:
        logger.error(f"Failed to determine default branch in {repo_path}: {e}", exc_info=True)
        return None


async def _load_and_chunk(repo_path: str, branch: Optional[str]) -> List[Document]:  # MODIFIED: async def
    """
    Load via GitLoader (if branch provided, checkout that branch first),
    then split all files matching EXTENSION_MAP into code-language chunks.
    Assumes repo_path is already validated as safe.
    Now supports asynchronous loading.
    """
    try:
        if not Path(repo_path).is_dir():
            logger.error(f"'{repo_path}' is not a valid directory for loading.")
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

        logger.info(f"Initializing GitLoader for {repo_path} (branch: {branch if branch else 'current'}).")
        loader_kwargs = {"repo_path": repo_path, "file_filter": file_filter}
        if branch:
            loader_kwargs["branch"] = branch

        loader = GitLoader(**loader_kwargs)
        # MODIFIED: Run loader.load() in a separate thread
        raw_docs = await asyncio.to_thread(loader.load)
        chunked = []

        for doc in raw_docs:
            ext = Path(doc.metadata.get("file_path", "")).suffix
            lang = EXTENSION_MAP.get(ext)
            if not lang:
                logger.debug(f"Skipping file {doc.metadata.get('file_path')} due to unsupported extension {ext}.")
                continue

            # RecursiveCharacterTextSplitter.from_language and split_documents are CPU-bound
            splitter = RecursiveCharacterTextSplitter.from_language(
                language=lang, chunk_size=1000, chunk_overlap=100
            )
            for chunk in splitter.split_documents([doc]):
                chunk.metadata["source_type"] = "code_repository"
                chunked.append(chunk)

        logger.info(f"Chunked {len(raw_docs)} files into {len(chunked)} code chunks for '{repo_path}'.")
        return chunked

    except (InvalidGitRepositoryError, NoSuchPathError) as git_loader_e:
        logger.error(f"GitLoader error for '{repo_path}': {git_loader_e}. Ensure path is a valid git repo.",
                     exc_info=True)
        return []
    except Exception as e:
        logger.error(f"Error loading or chunking '{repo_path}': {e}", exc_info=True)
        return []