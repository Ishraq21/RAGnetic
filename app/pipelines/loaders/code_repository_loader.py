import os
import tempfile
import logging
import subprocess
import re  # Added for PII redaction
from typing import List, Optional
from pathlib import Path
from urllib.parse import urlparse
import asyncio

from app.core.config import get_path_settings
from app.schemas.agent import AgentConfig, DataPolicy

from git import Repo, InvalidGitRepositoryError, NoSuchPathError, GitCommandError
from langchain_community.document_loaders import GitLoader
from langchain_core.documents import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter, Language

logger = logging.getLogger(__name__)

# --- Centralized Path Configuration ---
_PATH_SETTINGS = get_path_settings()
_PROJECT_ROOT_FROM_CONFIG = _PATH_SETTINGS["PROJECT_ROOT"]  # Store project root for clarity
_ALLOWED_DATA_DIRS_RESOLVED = _PATH_SETTINGS["ALLOWED_DATA_DIRS"]  # Store resolved allowed dirs

# NEW: Define _CLONE_TEMP_DIR relative to the centralized PROJECT_ROOT
_CLONE_TEMP_DIR = _PROJECT_ROOT_FROM_CONFIG / ".ragnetic_temp_clones"
logger.info(
    f"Loaded allowed data directories for code repository loader from central config: {[str(d) for d in _ALLOWED_DATA_DIRS_RESOLVED]}")


# --- End Centralized Configuration ---


def _is_path_safe_and_within_allowed_dirs(input_path: str) -> Path:
    """
    Validates if the input_path resolves to a location within the configured
    allowed data directories. Raises ValueError if unsafe.
    Returns the resolved absolute Path if safe.
    """
    resolved_path = Path(input_path).resolve()

    is_safe = False
    for allowed_dir in _ALLOWED_DATA_DIRS_RESOLVED:  # This variable now comes from central config
        if resolved_path.is_relative_to(allowed_dir):
            is_safe = True
            break

    if not is_safe:
        raise ValueError(f"Attempted to access path outside allowed directories: {resolved_path}")

    return resolved_path


def _apply_data_policies(text: str, policies: List[DataPolicy]) -> tuple[str, bool]:
    """
    Applies data policies (redaction/filtering) to the text content.
    Returns the processed text and a boolean indicating if the document (file) was blocked.
    """
    processed_text = text
    document_blocked = False

    for policy in policies:
        if policy.type == 'pii_redaction' and policy.pii_config:
            pii_config = policy.pii_config
            for pii_type in pii_config.types:
                pattern = None
                if pii_type == 'email':
                    pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
                elif pii_type == 'phone':
                    # Common phone number formats (adjust or enhance regex as needed for international formats)
                    pattern = r'\b(?:\+\d{1,3}[-. ]?)?\(?\d{3}\)?[-. ]?\d{3}[-. ]?\d{4}\b'
                elif pii_type == 'ssn':
                    pattern = r'\b\d{3}[- ]?\d{2}[- ]?\d{4}\b'
                elif pii_type == 'credit_card':
                    # Basic credit card pattern (major issuers, e.g., Visa, Mastercard, Amex, Discover)
                    pattern = r'\b(?:4\d{3}[- ]?\d{4}[- ]?\d{4}[- ]?\d{4}|5[1-5]\d{2}[- ]?\d{4}[- ]?\d{4}[- ]?\d{4}|3[47]\d{13}|6(?:011|5\d{2})[- ]?\d{4}[- ]?\d{4}[- ]?\d{4})\b'
                elif pii_type == 'name':
                    # Name redaction is complex and context-dependent.
                    logger.warning(
                        f"PII type '{pii_type}' (name) is complex and not fully implemented for regex-based redaction. Skipping for now.")
                    continue

                if pattern:
                    replacement = pii_config.redaction_placeholder or (pii_config.redaction_char * 8)  # Generic length
                    if pii_type == 'credit_card': replacement = pii_config.redaction_placeholder or (
                                pii_config.redaction_char * 16)  # Specific length for CC
                    processed_text = re.sub(pattern, replacement, processed_text)
                    logger.debug(f"Applied {pii_type} redaction policy. Replaced with: {replacement}")

        elif policy.type == 'keyword_filter' and policy.keyword_filter_config:
            kw_config = policy.keyword_filter_config
            for keyword in kw_config.keywords:
                if keyword in processed_text:
                    if kw_config.action == 'redact':
                        replacement = kw_config.redaction_placeholder or (kw_config.redaction_char * len(keyword))
                        processed_text = processed_text.replace(keyword, replacement)
                        logger.debug(f"Applied keyword redaction for '{keyword}'. Replaced with: {replacement}")
                    elif kw_config.action == 'block_chunk':
                        # At this stage (file level), we can't block just a chunk directly.
                        # We'll redact and log a warning for now, indicating this file contains content
                        # that should ideally be split and then blocked at chunking.
                        replacement = kw_config.redaction_placeholder or (kw_config.redaction_char * len(keyword))
                        processed_text = processed_text.replace(keyword, replacement)
                        logger.warning(
                            f"Keyword '{keyword}' found. This file contains content that should ideally be blocked at chunk level. Currently redacting.")
                    elif kw_config.action == 'block_document':  # In code loader, 'document' refers to a 'file'
                        logger.warning(
                            f"Keyword '{keyword}' found. File is marked for blocking by policy. Content will be discarded.")
                        document_blocked = True
                        return "", document_blocked  # Return empty content and blocked flag

    return processed_text, document_blocked


async def load(repo_path: str, agent_config: Optional[AgentConfig] = None) -> List[Document]:  # Added agent_config
    """
    Smart loader that handles both remote GitHub URLs and local file paths.
    Includes input validation for both types, data policy application, and robust error handling.
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
                    repo = await asyncio.to_thread(Repo.clone_from, repo_path, to_path=str(safe_temp_dir))

                    default_branch = await _get_default_branch(str(safe_temp_dir))
                    if not default_branch:
                        logger.error(f"Can't determine default branch for {repo_path}. Skipping.")
                        return []
                    # Pass agent_config to _load_and_chunk
                    return await _load_and_chunk(str(safe_temp_dir), branch=default_branch, agent_config=agent_config)

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
            # Pass agent_config to _load_and_chunk
            return await _load_and_chunk(str(safe_local_repo_path), branch=None, agent_config=agent_config)

    except ValueError as e:
        logger.error(f"Code repository loader validation error: {e}")
        return []
    except Exception as e:
        logger.error(f"An unexpected error occurred in code_repository_loader: {e}", exc_info=True)
        return []


async def _get_default_branch(repo_path: str) -> Optional[str]:
    """
    Run `git symbolic-ref refs/remotes/origin/HEAD` to discover the default branch name.
    Now supports asynchronous execution.
    """
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


async def _load_and_chunk(repo_path: str, branch: Optional[str], agent_config: Optional[AgentConfig]) -> List[
    Document]:  # Added agent_config
    """
    Load via GitLoader (if branch provided, checkout that branch first),
    then split all files matching EXTENSION_MAP into code-language chunks.
    Assumes repo_path is already validated as safe.
    Applies data policies before chunking.
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
            ".yml": Language.YAML, ".yaml": Language.YAML,  # Added for IaC/config files often in repos
            ".json": Language.JSON  # Added for config/data files
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

            # Apply data policies if provided in agent_config
            if agent_config and agent_config.data_policies:
                logger.debug(
                    f"Applying data policies to file '{doc.metadata.get('file_path', doc.metadata.get('source'))}'...")
                processed_text, document_blocked = _apply_data_policies(doc.page_content, agent_config.data_policies)

            if document_blocked:
                logger.warning(
                    f"File '{doc.metadata.get('file_path', doc.metadata.get('source'))}' was completely blocked by a data policy and will not be processed.")
                continue  # Skip this document if it's blocked

            # Update the document's page_content with the processed text
            doc.page_content = processed_text
            processed_docs.append(doc)

        chunked = []
        for doc in processed_docs:  # Iterate over processed documents
            ext = Path(doc.metadata.get("file_path", "")).suffix
            lang = EXTENSION_MAP.get(ext)
            if not lang:
                logger.debug(
                    f"Skipping file {doc.metadata.get('file_path')} due to unsupported extension {ext} for chunking.")
                continue

            # Ensure there is content left after policy application before chunking
            if not doc.page_content.strip():
                logger.debug(
                    f"File {doc.metadata.get('file_path')} has no content after policy application. Skipping chunking.")
                continue

            splitter = RecursiveCharacterTextSplitter.from_language(
                language=lang, chunk_size=1000, chunk_overlap=100
            )
            for chunk in splitter.split_documents([doc]):
                chunk.metadata["source_type"] = "code_repository"
                # Add file_name and file_path to metadata for consistency
                if "file_path" in doc.metadata:
                    chunk.metadata["file_path"] = doc.metadata["file_path"]
                    chunk.metadata["file_name"] = Path(doc.metadata["file_path"]).name
                elif "source" in doc.metadata:  # Fallback if GitLoader uses 'source' for full path
                    chunk.metadata["file_path"] = doc.metadata["source"]
                    chunk.metadata["file_name"] = Path(doc.metadata["source"]).name

                chunked.append(chunk)

        logger.info(f"Chunked {len(processed_docs)} processed files into {len(chunked)} code chunks for '{repo_path}'.")
        return chunked

    except (InvalidGitRepositoryError, NoSuchPathError) as git_loader_e:
        logger.error(f"GitLoader error for '{repo_path}': {git_loader_e}. Ensure path is a valid git repo.",
                     exc_info=True)
        return []
    except Exception as e:
        logger.error(f"Error loading or chunking '{repo_path}': {e}", exc_info=True)
        return []