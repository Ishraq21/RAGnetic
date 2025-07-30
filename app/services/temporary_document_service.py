# app/services/temporary_document_service.py

import os
import shutil
import uuid
import logging
import mimetypes
from pathlib import Path
from typing import Dict, Any, List, Optional
import asyncio

from fastapi import UploadFile

from app.core.config import get_path_settings
from app.core.embed_config import get_embedding_model
from app.core.parsing_utils import parse_document_to_chunks
from app.agents.config_manager import AgentConfig

logger = logging.getLogger(__name__)

# --- Configuration Paths ---
_APP_PATHS = get_path_settings()
_TEMP_CHAT_UPLOADS_DIR = _APP_PATHS["TEMP_CLONES_DIR"] / "chat_uploads"
_TEMP_VECTORSTORE_DIR = _APP_PATHS["VECTORSTORE_DIR"] / "temp_chat_data"

# Ensure directories exist
_TEMP_CHAT_UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
_TEMP_VECTORSTORE_DIR.mkdir(parents=True, exist_ok=True)


class TemporaryDocumentUploadResult:
    """Result object for a successful temporary document upload."""

    def __init__(self, file_name: str, file_size: int, temp_doc_id: str):
        self.file_name = file_name
        self.file_size = file_size
        self.temp_doc_id = temp_doc_id


class TemporaryDocumentService:
    def __init__(self, agent_config: AgentConfig):
        if not hasattr(agent_config, 'embedding_model') or not agent_config.embedding_model:
            raise ValueError("AgentConfig must specify an 'embedding_model' for TemporaryDocumentService.")
        self.embedding_model = get_embedding_model(agent_config.embedding_model)
        logger.info(f"TemporaryDocumentService initialized with embedding model: {agent_config.embedding_model}")

    async def process_and_store_temp_document(
            self,
            file: UploadFile,
            user_id: int,
            thread_id: str
    ) -> TemporaryDocumentUploadResult:
        """
        Processes an uploaded file, stores it temporarily, parses, chunks,
        embeds, and stores embeddings in a temporary vector store.
        """
        temp_doc_id = str(uuid.uuid4())
        user_thread_dir = _TEMP_CHAT_UPLOADS_DIR / str(user_id) / thread_id
        user_thread_dir.mkdir(parents=True, exist_ok=True)

        temp_file_name_with_id = f"{temp_doc_id}_{file.filename}"
        temp_file_path = user_thread_dir / temp_file_name_with_id

        logger.info(f"Uploading temporary file {file.filename} to {temp_file_path}")

        try:
            with open(temp_file_path, "wb") as buffer:
                while content := await file.read(1024 * 1024):
                    buffer.write(content)
            file_size = temp_file_path.stat().st_size
        except Exception as e:
            logger.error(f"Failed to save temporary file {file.filename}: {e}")
            raise ValueError(f"Failed to save uploaded file: {e}")

        if file_size == 0:
            os.remove(temp_file_path)
            raise ValueError("Uploaded file is empty.")

        mime_type, _ = mimetypes.guess_type(temp_file_path)
        if mime_type:
            file_type = mime_type.split('/')[0]
        else:
            file_type = temp_file_path.suffix.lstrip('.').lower()

        MAX_FILE_SIZE_BYTES = 25 * 1024 * 1024
        SUPPORTED_EXTENSIONS = ('.pdf', '.docx', '.txt', '.csv', '.json', '.yaml', '.yml', '.hcl', '.tf', '.ipynb',
                                '.md', '.log')

        if file_size > MAX_FILE_SIZE_BYTES:
            os.remove(temp_file_path)
            raise ValueError(f"File size exceeds {MAX_FILE_SIZE_BYTES / (1024 * 1024)}MB limit.")
        if temp_file_path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            os.remove(temp_file_path)
            raise ValueError(
                f"Unsupported file type: {temp_file_path.suffix}. Supported types are: {', '.join(SUPPORTED_EXTENSIONS)}")

        try:
            chunks = await parse_document_to_chunks(temp_file_path)

            documents_to_embed = []
            for i, chunk in enumerate(chunks):
                chunk.metadata.update({
                    "user_id": str(user_id),
                    "thread_id": thread_id,
                    "temp_doc_id": temp_doc_id,
                    "doc_name": file.filename,
                    "chunk_index": i,
                    "original_file_path": str(temp_file_path.relative_to(_APP_PATHS["PROJECT_ROOT"]))
                })
                documents_to_embed.append(chunk)

            if not documents_to_embed:
                os.remove(temp_file_path)
                raise ValueError("Could not extract any content from the uploaded file.")

        except Exception as e:
            logger.error(f"Failed to parse or chunk document {file.filename}: {e}")
            os.remove(temp_file_path)
            raise ValueError(f"Failed to process document content: {e}")

        try:
            from langchain_community.vectorstores import FAISS

            temp_vector_store_path = _TEMP_VECTORSTORE_DIR / temp_doc_id
            temp_vector_store_path.mkdir(parents=True, exist_ok=True)

            temp_db = await asyncio.to_thread(FAISS.from_documents, documents_to_embed, self.embedding_model)
            await asyncio.to_thread(temp_db.save_local, str(temp_vector_store_path))

            logger.info(f"Embeddings for {file.filename} saved to temporary vector store at {temp_vector_store_path}")

        except Exception as e:
            logger.error(f"Failed to embed or store document {file.filename} in vector store: {e}")
            if temp_file_path.exists():
                os.remove(temp_file_path)
            if temp_vector_store_path.exists():
                shutil.rmtree(temp_vector_store_path, ignore_errors=True)
            raise ValueError(f"Failed to create embeddings for document: {e}")

        return TemporaryDocumentUploadResult(
            file_name=file.filename,
            file_size=file_size,
            temp_doc_id=temp_doc_id
        )

    @staticmethod
    def cleanup_temp_document(temp_doc_id: str):
        """Removes a temporary document and its associated vector store data."""
        logger.info(f"Attempting to clean up temporary document: {temp_doc_id}")

        file_removed = False
        for user_dir in _TEMP_CHAT_UPLOADS_DIR.iterdir():
            if user_dir.is_dir():
                for thread_dir in user_dir.iterdir():
                    if thread_dir.is_dir():
                        for file_in_dir in thread_dir.iterdir():
                            if file_in_dir.is_file() and file_in_dir.name.startswith(f"{temp_doc_id}_"):
                                os.remove(file_in_dir)
                                logger.info(f"Removed temporary document file: {file_in_dir}")
                                file_removed = True
                                break
                        if file_removed:
                            break
                if file_removed:
                    break

        if not file_removed:
            logger.warning(f"Temporary document file not found for cleanup (ID: {temp_doc_id}).")

        temp_vector_store_path = _TEMP_VECTORSTORE_DIR / temp_doc_id

        if temp_vector_store_path.exists():
            shutil.rmtree(temp_vector_store_path, ignore_errors=True)
            logger.info(f"Removed temporary vector store for: {temp_doc_id}")
        else:
            logger.warning(f"Temporary vector store not found for cleanup: {temp_vector_store_path}")

    @staticmethod
    def cleanup_user_thread_temp_documents(user_id: int, thread_id: str):
        """
        Cleans up all temporary documents for a given user and thread by iterating their files.
        """
        user_thread_upload_dir = _TEMP_CHAT_UPLOADS_DIR / str(user_id) / thread_id

        logger.info(f"Cleaning up temporary documents for user {user_id}, thread {thread_id}")

        if user_thread_upload_dir.exists():
            files_in_dir = list(user_thread_upload_dir.iterdir())
            for file_path in files_in_dir:
                if file_path.is_file() and file_path.name.startswith(
                        f"{str(uuid.UUID(file_path.name.split('_')[0]))}_"):
                    try:
                        extracted_temp_doc_id = file_path.name.split('_', 1)[0]
                        TemporaryDocumentService.cleanup_temp_document(extracted_temp_doc_id)
                    except ValueError as e:
                        logger.warning(
                            f"Skipping file {file_path.name} during thread cleanup due to unexpected naming: {e}")
                else:
                    logger.warning(
                        f"Skipping non-file or non-temp-doc-named item during thread cleanup: {file_path.name}")

            shutil.rmtree(user_thread_upload_dir, ignore_errors=True)
            logger.info(f"Removed temporary upload directory: {user_thread_upload_dir}")
        else:
            logger.warning(f"Temporary upload directory not found for cleanup: {user_thread_upload_dir}")