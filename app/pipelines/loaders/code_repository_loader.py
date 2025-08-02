import os
import tempfile
import logging
import subprocess
from typing import List, Optional
from pathlib import Path
from urllib.parse import urlparse
import asyncio
from datetime import datetime

from app.core.config import get_path_settings
from app.schemas.agent import AgentConfig, DataSource
from app.pipelines.data_policy_utils import apply_data_policies
from app.pipelines.metadata_utils import generate_base_metadata

from git import Repo, InvalidGitRepositoryError, NoSuchPathError, GitCommandError
from langchain_community.document_loaders import GitLoader
from langchain_core.documents import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter, Language

logger = logging.getLogger(__name__)

# --- Centralized Path Configuration ---
_PATH_SETTINGS = get_path_settings()
_PROJECT_ROOT_FROM_CONFIG = _PATH_SETTINGS["PROJECT_ROOT"]
_ALLOWED_DATA_DIRS_RESOLVED = _PATH_SETTINGS["ALLOWED_DATA_DIRS"]
logger.debug(
    f"Loaded allowed data directories for code repository loader from central config: {[str(d) for d in _ALLOWED_DATA_DIRS_RESOLVED]}")

_CLONE_TEMP_DIR = _PROJECT_ROOT_FROM_CONFIG / ".ragnetic_temp_clones"


def _is_path_safe_and_within_allowed_dirs(input_path: str) -> Path:
    resolved_path = Path(input_path).resolve()

    is_safe = False
    for allowed_dir in _ALLOWED_DATA_DIRS_RESOLVED:
        if resolved_path.is_relative_to(allowed_dir):
            is_safe = True
            break

    if not is_safe:
        raise ValueError(f"Attempted to access path outside allowed directories: {resolved_path}")

    return resolved_path


async def load(repo_path: str, agent_config: Optional[AgentConfig] = None, source: Optional[DataSource] = None) -> List[Document]:
    try:
        if not repo_path:
            logger.error("Validation Error: Repository path (URL or local) is required.")
            return []

        if repo_path.startswith(("http://", "https://", "git@")):
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
                    repo = await asyncio.to_thread(Repo.clone_from, repo_path, to_path=str(safe_temp_dir))

                    default_branch = await _get_default_branch(str(safe_temp_dir))
                    if not default_branch:
                        logger.error(f"Can't determine default branch for {repo_path}. Skipping.")
                        return []
                    return await _load_and_chunk(str(safe_temp_dir), branch=default_branch, agent_config=agent_config,
                                                 source=source)

                except (GitCommandError, InvalidGitRepositoryError, NoSuchPathError) as git_e:
                    logger.error(f"Git operation failed for {repo_path}: {git_e}", exc_info=True)
                    return []
                except Exception as e:
                    logger.error(f"An unexpected error during cloning/processing {repo_path}: {e}", exc_info=True)
                    return []
        else:
            logger.info(f"Processing local code repository at: {repo_path}")
            safe_local_repo_path = _is_path_safe_and_within_allowed_dirs(repo_path)
            return await _load_and_chunk(str(safe_local_repo_path), branch=None, agent_config=agent_config,
                                         source=source)

    except ValueError as e:
        logger.error(f"Code repository loader validation error: {e}")
        return []
    except Exception as e:
        logger.error(f"An unexpected error occurred in code_repository_loader: {e}", exc_info=True)
        return []


async def _get_default_branch(repo_path: str) -> Optional[str]:
    try:
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


async def _load_and_chunk(repo_path: str, branch: Optional[str], agent_config: Optional[AgentConfig],
                          source: Optional[DataSource]) -> List[Document]:
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
            ".yml": Language.YAML, ".yaml": Language.YAML,
            ".json": Language.JSON
        }

        def file_filter(path: str) -> bool:
            return Path(path).suffix in EXTENSION_MAP

        logger.info(f"Initializing GitLoader for {repo_path} (branch: {branch if branch else 'current'}).")
        loader_kwargs = {"repo_path": repo_path, "file_filter": file_filter}
        if branch:
            loader_kwargs["branch"] = branch

        loader = GitLoader(**loader_kwargs)
        raw_docs = await asyncio.to_thread(loader.load)

        processed_docs = []
        for doc in raw_docs:
            processed_text = doc.page_content
            document_blocked = False

            if agent_config and agent_config.data_policies:
                logger.debug(
                    f"Applying data policies to file '{doc.metadata.get('file_path', doc.metadata.get('source'))}'...")
                processed_text, document_blocked = apply_data_policies(doc.page_content, agent_config.data_policies, policy_context="file")

            if document_blocked:
                logger.warning(
                    f"File '{doc.metadata.get('file_path', doc.metadata.get('source'))}' was completely blocked by a data policy and will not be processed.")
                continue

            doc.page_content = processed_text

            source_context = doc.metadata.get('file_path', repo_path)
            metadata = generate_base_metadata(source, source_context=source_context, source_type="code_repository")
            # Add code repository-specific keys
            metadata["source_path"] = str(Path(repo_path).resolve())
            metadata["file_name"] = Path(doc.metadata.get("file_path", "")).name
            metadata["file_path_in_repo"] = doc.metadata.get("file_path", "N/A")
            metadata["file_type"] = Path(doc.metadata.get("file_path", "")).suffix.lower()
            metadata["repo_url"] = source.url if source and source.url else "N/A"
            metadata["repo_branch"] = branch if branch else "default"

            doc.metadata = {**doc.metadata, **metadata}
            processed_docs.append(doc)

        chunked = []
        for doc in processed_docs:
            ext = Path(doc.metadata.get("file_path", "")).suffix
            lang = EXTENSION_MAP.get(ext)
            if not lang:
                logger.debug(
                    f"Skipping file {doc.metadata.get('file_path')} due to unsupported extension {ext} for chunking.")
                continue

            if not doc.page_content.strip():
                logger.debug(
                    f"File {doc.metadata.get('file_path')} has no content after policy application. Skipping chunking.")
                continue

            splitter = RecursiveCharacterTextSplitter.from_language(
                language=lang, chunk_size=1000, chunk_overlap=100
            )
            for chunk in splitter.split_documents([doc]):
                # Preserve all rich metadata from the document onto the chunk
                chunk.metadata = {**chunk.metadata, **doc.metadata}
                chunked.append(chunk)

        logger.info(
            f"Chunked {len(processed_docs)} processed files into {len(chunked)} code chunks for '{repo_path}' with enriched metadata.")
        return chunked

    except (InvalidGitRepositoryError, NoSuchPathError) as git_loader_e:
        logger.error(f"GitLoader error for '{repo_path}': {git_loader_e}. Ensure path is a valid git repo.",
                     exc_info=True)
        return []
    except Exception as e:
        logger.error(f"Error loading or chunking '{repo_path}': {e}", exc_info=True)
        return []