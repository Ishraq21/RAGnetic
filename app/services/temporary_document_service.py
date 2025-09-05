# app/services/temporary_document_service.py

import os
import shutil
import uuid
import logging
from pathlib import Path
from typing import List, Dict, Any
import asyncio
import filetype

from fastapi import UploadFile
from sqlalchemy import select, desc
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.exc import IntegrityError

from app.core.config import get_path_settings
from app.core.embed_config import get_embedding_model
from app.core.parsing_utils import parse_document_to_chunks
from app.agents.config_manager import AgentConfig
from app.db import get_async_db_session
from app.db.dao import create_document_chunk, create_temp_document, delete_temp_document_data
from app.db.models import temporary_documents_table
from app.schemas.agent import DocumentMetadata

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
        if not getattr(agent_config, 'embedding_model', None):
            raise ValueError("AgentConfig must specify an 'embedding_model' for TemporaryDocumentService.")
        self.embedding_model = get_embedding_model(agent_config.embedding_model)
        logger.info(
            f"TemporaryDocumentService initialized with embedding model: {agent_config.embedding_model}"
        )

    async def process_and_store_temp_document(
            self,
            file: UploadFile,
            user_id: int,
            thread_id: str,
            db: AsyncSession
    ) -> TemporaryDocumentUploadResult:
        """
        Processes an uploaded file, stores it temporarily, parses, chunks,
        persists each chunk to the DB (so it gets a chunk_id), embeds,
        and stores embeddings in a temporary FAISS store.
        """
        try:
            temp_doc_record = await create_temp_document(
                db=db,
                user_id=user_id,
                thread_id=thread_id,
                original_name=file.filename,
                file_size=file.size if file.size is not None else 0,
            )
            temp_doc_id = temp_doc_record['temp_doc_id']
            temp_doc_db_id = temp_doc_record['id']
            logger.info(f"Created temp doc record {temp_doc_id} in DB.")
        except IntegrityError:
            await db.rollback()
            raise ValueError("Failed to create unique temp document record.")
        except Exception as e:
            await db.rollback()
            logger.error(f"Failed to create temp document DB record: {e}", exc_info=True)
            raise ValueError("Failed to create temp document record.")

        temp_file_path = None
        vs_path = None
        try:
            # 2) Save the uploaded file to a temp directory
            upload_dir = _TEMP_CHAT_UPLOADS_DIR / str(user_id) / thread_id
            upload_dir.mkdir(parents=True, exist_ok=True)

            temp_file_name = f"{temp_doc_id}_{file.filename}"
            temp_file_path = upload_dir / temp_file_name
            logger.info(f"Uploading temporary file {file.filename} to {temp_file_path}")

            file.file.seek(0)
            file_content = await file.read()

            with open(temp_file_path, "wb") as buf:
                buf.write(file_content)

            file_size = temp_file_path.stat().st_size

            if file_size == 0:
                raise ValueError("Uploaded file is empty.")

            # --- Advanced File Validation using pure Python 'filetype' library ---
            MAX_SIZE = 25 * 1024 * 1024
            ALLOWED_EXTENSIONS = {
                '.pdf', '.docx', '.txt', '.csv', '.json', '.yaml', '.yml',
                '.hcl', '.tf', '.ipynb', '.md', '.log', '.html'
            }
            ALLOWED_MIMETYPES = {
                'application/pdf', 'application/vnd.openxmlformats-officedocument.wordprocessingml.document',
                'text/plain', 'text/csv', 'application/json', 'text/x-yaml',
                'application/x-yaml', 'text/x-hcl', 'text/x-terraform',
                'application/x-ipynb+json', 'text/markdown', 'text/html'
            }

            if file_size > MAX_SIZE:
                raise ValueError(f"File exceeds size limit of {MAX_SIZE // (1024 * 1024)}MB.")

            if temp_file_path.suffix.lower() not in ALLOWED_EXTENSIONS:
                raise ValueError(
                    f"Unsupported file extension: {temp_file_path.suffix}. "
                    f"Allowed: {', '.join(sorted(ALLOWED_EXTENSIONS))}"
                )

            try:
                kind = filetype.guess(str(temp_file_path))
                file_mime_type = kind.mime if kind else 'text/plain'

                if file_mime_type not in ALLOWED_MIMETYPES:
                    raise ValueError(
                        f"Unsupported file content type: {file_mime_type}. "
                        f"Allowed: {', '.join(sorted(ALLOWED_MIMETYPES))}"
                    )
                logger.info(f"File {file.filename} passed MIME type validation: {file_mime_type}")
            except Exception as e:
                logger.error(f"MIME type check failed for {file.filename}: {e}", exc_info=True)
                raise ValueError(f"Failed to validate file content: {e}")

            # --- END NEW LOGIC ---

            # 3) Parse into chunks
            try:
                chunks = await parse_document_to_chunks(temp_file_path)
            except Exception as e:
                logger.error(f"Parsing failed for {file.filename}: {e}", exc_info=True)
                raise ValueError(f"Failed to parse document: {e}")

            if not chunks:
                raise ValueError("No content extracted from document.")

            # 4) Persist each chunk to DB to get chunk_id
            documents_to_embed: List = []
            unique_document_name = f"temp::{temp_doc_id}::{file.filename}"
            for idx, chunk in enumerate(chunks):
                chunk.metadata.update({
                    "user_id": str(user_id),
                    "thread_id": thread_id,
                    "temp_doc_id": temp_doc_id,
                    "doc_name": file.filename,  # Keep original filename in metadata
                    "chunk_index": idx,
                    "original_file_path": str(temp_file_path.relative_to(_APP_PATHS["PROJECT_ROOT"])),
                })
                db_id = await create_document_chunk(
                    db=db,
                    document_name=unique_document_name,  # Use the new unique name
                    chunk_index=idx,
                    content=chunk.page_content,
                    page_number=chunk.metadata.get("page_number"),
                    row_number=chunk.metadata.get("row_number"),
                    temp_document_id=temp_doc_db_id,
                )
                chunk.id = str(db_id)
                chunk.metadata["chunk_id"] = db_id
                documents_to_embed.append(chunk)

            # 5) Build and save temporary FAISS index
            try:
                from langchain_community.vectorstores import FAISS

                vs_path = _TEMP_VECTORSTORE_DIR / temp_doc_id
                vs_path.mkdir(parents=True, exist_ok=True)

                faiss_store = await asyncio.to_thread(
                    FAISS.from_documents,
                    documents_to_embed,
                    self.embedding_model
                )
                await asyncio.to_thread(faiss_store.save_local, str(vs_path))
                logger.info(
                    f"Temporary FAISS store saved at {vs_path} for doc {file.filename}"
                )
            except Exception as e:
                logger.error(f"FAISS embedding failed for {file.filename}: {e}", exc_info=True)
                raise ValueError(f"Failed to embed document: {e}")

            await db.commit()

            return TemporaryDocumentUploadResult(
                file_name=file.filename,
                file_size=file.size if file.size is not None else file_size,
                temp_doc_id=temp_doc_id
            )

        except Exception as e:
            logger.error(f"Critical failure during temp document processing for {temp_doc_id}: {e}", exc_info=True)
            await delete_temp_document_data(db, temp_doc_id)
            if temp_file_path and temp_file_path.exists():
                temp_file_path.unlink()
            if vs_path and vs_path.exists():
                shutil.rmtree(vs_path, ignore_errors=True)
            raise e


    @staticmethod
    def cleanup_fs(doc_data: Dict[str, Any]):
        """
        Static method to remove a single document's associated vector store data and physical file.
        This method is called by the cleanup tasks and requires no class instance state.
        """
        temp_doc_id = doc_data['temp_doc_id']
        original_name = doc_data['original_name']
        user_id = doc_data['user_id']
        thread_id = doc_data['thread_id']

        logger.info(f"Attempting to clean up filesystem for temporary document: {temp_doc_id}")

        file_path = _TEMP_CHAT_UPLOADS_DIR / str(user_id) / thread_id / f"{temp_doc_id}_{original_name}"
        if file_path.exists():
            try:
                os.remove(file_path)
                logger.info(f"Removed temporary document file: {file_path}")
            except OSError as e:
                logger.warning(f"Failed to remove file {file_path}: {e}")
        else:
            logger.warning(f"Temporary document file not found for cleanup: {file_path}")

        temp_vector_store_path = _TEMP_VECTORSTORE_DIR / temp_doc_id
        if temp_vector_store_path.exists():
            try:
                shutil.rmtree(temp_vector_store_path, ignore_errors=False)
                logger.info(f"Removed temporary vector store directory: {temp_vector_store_path}")
            except OSError as e:
                logger.error(f"Failed to remove vector store directory {temp_vector_store_path}: {e}")
        else:
            logger.warning(f"Temporary vector store directory not found for cleanup: {temp_vector_store_path}")

    async def cleanup_user_thread_temp_documents(self, user_id: int, thread_id: str, db: AsyncSession):
        """
        Cleans up all temporary documents for a given user and thread.
        This is a transactional operation that deletes files and database records.
        """
        logger.info(f"Starting cleanup for temporary documents of user {user_id}, thread {thread_id}")

        stmt = select(temporary_documents_table).where(
            temporary_documents_table.c.user_id == user_id,
            temporary_documents_table.c.thread_id == thread_id,
            temporary_documents_table.c.cleaned_up == False
        )
        temp_docs_to_clean = (await db.execute(stmt)).mappings().all()

        if not temp_docs_to_clean:
            logger.info(f"No temporary documents found for user {user_id} and thread {thread_id}. Cleanup finished.")
            return

        logger.info(f"Found {len(temp_docs_to_clean)} documents to clean.")

        for doc in temp_docs_to_clean:
            doc_dict = dict(doc)
            logger.info(f"Processing cleanup for temp doc: {doc_dict['temp_doc_id']}")
            self.cleanup_fs(doc_dict)
            await delete_temp_document_data(db, doc_dict['temp_doc_id'])

        user_thread_upload_dir = _TEMP_CHAT_UPLOADS_DIR / str(user_id) / thread_id
        if user_thread_upload_dir.exists():
            shutil.rmtree(user_thread_upload_dir, ignore_errors=True)
            logger.info(f"Removed empty temporary upload directory: {user_thread_upload_dir}")

        logger.info(f"Finished cleanup for user {user_id} and thread {thread_id}.")

    def get_latest_by_filename(self, file_name: str) -> Dict[str, Any]:
        """
        Look up the most recent temp_doc by original filename.
        Safe in both sync + threaded contexts (e.g., LangChain tools).
        Returns enriched metadata including file path for FileService.
        """

        async def _fetch():
            try:
                async with get_async_db_session() as db:
                    stmt = (
                        select(temporary_documents_table)
                        .where(temporary_documents_table.c.original_name == file_name)
                        .order_by(desc(temporary_documents_table.c.created_at))
                        .limit(1)
                    )
                    result = await db.execute(stmt)
                    row = result.mappings().first()
                    if not row:
                        return None

                    # Build original path (where process_and_store_temp_document saved the file)
                    user_id = row["user_id"]
                    thread_id = row["thread_id"]
                    temp_doc_id = row["temp_doc_id"]
                    original_name = row["original_name"]
                    file_path = (
                            _TEMP_CHAT_UPLOADS_DIR / str(user_id) / thread_id / f"{temp_doc_id}_{original_name}"
                    )

                    return {
                        "temp_doc_id": temp_doc_id,
                        "original_name": original_name,
                        "user_id": user_id,
                        "thread_id": thread_id,
                        "file_path": str(file_path),
                    }
            except Exception as e:
                logger.error(f"Database query failed for file {file_name}: {e}", exc_info=True)
                return None

        try:
            loop = asyncio.get_running_loop()
            # If we're in an async context, we need to create a task
            import concurrent.futures
            import threading
            
            # Run in a separate thread to avoid event loop conflicts
            with concurrent.futures.ThreadPoolExecutor() as executor:
                future = executor.submit(lambda: asyncio.run(_fetch()))
                return future.result(timeout=30)  # 30 second timeout
        except RuntimeError:
            # No event loop running, safe to use asyncio.run()
            return asyncio.run(_fetch())
        except Exception as e:
            logger.error(f"Failed to fetch latest document for {file_name}: {e}", exc_info=True)
            return None

#22