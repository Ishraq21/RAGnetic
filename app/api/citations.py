# app/api/citations.py
import asyncio
import logging
from typing import Optional, Dict, Any, List
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException, status, Query
from sqlalchemy.ext.asyncio import AsyncSession
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

from app.db import get_db
from app.core.security import get_http_api_key, get_current_user_from_api_key
from app.schemas.security import User  # Assuming User is the correct model for current_user
from app.core.config import get_path_settings
from app.core.embed_config import get_embedding_model  # To load FAISS with correct embeddings

logger = logging.getLogger("ragnetic")

router = APIRouter(prefix="/api/v1/chat", tags=["Chat"])

# Define paths for temporary storage
_APP_PATHS = get_path_settings()
_TEMP_CHAT_UPLOADS_DIR = _APP_PATHS["TEMP_CLONES_DIR"] / "chat_uploads"
_TEMP_VECTORSTORE_DIR = _APP_PATHS["VECTORSTORE_DIR"] / "temp_chat_data"


@router.get("/citation-snippet")
async def get_citation_snippet(
        temp_doc_id: str = Query(..., description="The unique ID of the temporary document."),
        doc_name: str = Query(..., description="The original name of the document."),
        page: Optional[int] = Query(None, description="The page number of the snippet (if applicable)."),
        current_user: User = Depends(get_current_user_from_api_key),
        db: AsyncSession = Depends(get_db)  # db session might be useful for future metadata lookup
) -> Dict[str, str]:
    """
    Retrieves a text snippet for a given citation from a temporary document.
    This endpoint is used by the frontend to display source context on citation hover/click.
    """
    user_id = current_user.id
    logger.info(
        f"User {user_id} requesting citation snippet for temp_doc_id: {temp_doc_id}, doc_name: {doc_name}, page: {page}")

    # Step 1: Locate the FAISS vector store for the specific temp_doc_id
    temp_vector_store_path = _TEMP_VECTORSTORE_DIR / temp_doc_id

    if not temp_vector_store_path.exists():
        logger.warning(f"Temporary vector store not found for temp_doc_id: {temp_doc_id}")
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Citation source not found or has been purged."
        )

    try:
        embeddings = get_embedding_model()  # Get the default embedding model to load FAISS

        # Load the specific FAISS index for this temporary document
        vectorstore = await asyncio.to_thread(
            FAISS.load_local,
            str(temp_vector_store_path),
            embeddings,
            allow_dangerous_deserialization=True  # Necessary if documents contain non-primitive types
        )
        logger.debug(f"Loaded FAISS store for {temp_doc_id}.")

        # Step 2: Search for the relevant chunk(s) within this temporary document's index
        # We need to find chunks that match doc_name and page, and belong to this user/thread
        # The metadata in the FAISS index should contain 'doc_name', 'page_number', 'user_id', 'thread_id', 'temp_doc_id'

        # Construct a query to find the specific chunk.
        # A simple approach is to retrieve all documents from this temp_doc_id's index
        # and then filter by metadata. A more efficient way would be to query the vector store
        # with metadata filters if the vector store supports it (FAISS.similarity_search_by_vector
        # or FAISS.similarity_search with a dummy query, then filter).

        # For FAISS, direct metadata filtering during retrieval isn't as straightforward as
        # with other vector DBs. A common pattern is to retrieve broadly and then filter in-memory.
        # However, since we're loading a *specific* temp_doc_id's index, all docs in it belong
        # to that temp_doc_id. We just need to find the right page/chunk.

        # A dummy query to retrieve some documents, then filter by metadata
        # Or, if you stored the original file, read from there directly.
        # For simplicity, let's assume we can retrieve by a unique identifier or filter.

        # Best approach: If you stored original files, read from there.
        # Alternative: Retrieve all docs from this specific FAISS index and filter.

        # Option A: Read from original temporary file (more accurate for exact snippet)
        # This requires knowing the original file's path.
        # The original file path is stored in chunk metadata: 'original_file_path'
        # We need to retrieve the original file path from one of the chunks' metadata.

        # To get the original file path, we'd ideally query the DB or have a mapping.
        # For now, let's assume we can reconstruct it or find it via a simple search in the temp upload dir.

        # A more robust way would be to store a mapping of temp_doc_id -> original_file_path in a lightweight DB table
        # or in Redis, or retrieve it from the chat_messages_table.meta.

        # For this endpoint, let's simplify and assume we can retrieve the relevant chunk
        # directly from the FAISS index by searching for a "dummy" query that should
        # return all documents, then filter by page number.

        # Retrieve a broad set of documents from this specific temp_doc_id's index
        # A common way to get all documents from a FAISS index is to query with a vector
        # that is "average" or "zero" and retrieve a very large 'k'.

        # For a more direct way: if you stored the raw file, read from that.
        # Let's try to get the content from the original file first, as it's more reliable for snippets.

        # First, try to find the original file path from the temp upload directory structure
        # This is a bit hacky, but works with the current `TemporaryDocumentService` structure.
        original_file_location = None
        for user_dir in _TEMP_CHAT_UPLOADS_DIR.iterdir():
            if user_dir.is_dir():
                for thread_dir in user_dir.iterdir():
                    if thread_dir.is_dir():
                        # Iterate through files in the thread directory
                        for file_in_dir in thread_dir.iterdir():
                            # Check if the filename contains the temp_doc_id (assuming it's part of the filename or a convention)
                            # Or, if you stored a direct mapping, use that.
                            # For now, we'll rely on the FAISS metadata to get the original_file_path.
                            pass  # We'll get the path from the document metadata below

        target_snippet_content = "Snippet not found."

        # Retrieve documents from the FAISS index and filter by page
        # A dummy query to get all documents in the index (or a large enough subset)
        all_docs_in_temp_index = vectorstore.similarity_search("dummy query", k=1000)  # Retrieve a large number

        found_doc_for_snippet = None
        for doc in all_docs_in_temp_index:
            if doc.metadata.get('doc_name') == doc_name and \
                    (page is None or doc.metadata.get('page_number') == page) and \
                    doc.metadata.get('temp_doc_id') == temp_doc_id:
                found_doc_for_snippet = doc
                break

        if found_doc_for_snippet:
            target_snippet_content = found_doc_for_snippet.page_content
            logger.info(f"Found matching chunk for snippet: {doc_name} (Page: {page}) from temp_doc_id: {temp_doc_id}")
        else:
            logger.warning(
                f"No exact chunk found in FAISS for {doc_name} (Page: {page}) from temp_doc_id: {temp_doc_id}. Attempting to read from original file if path available in metadata.")
            # Fallback: Try to read from the original file path if available in any chunk's metadata
            # This is more complex as it requires parsing the original file again.
            # For a quick snippet, the FAISS chunk is usually sufficient.

            # If you want to read from the original file, you'd need the full path.
            # Let's get the path from any document in the index, assuming they all point to the same original file.
            if all_docs_in_temp_index and 'original_file_path' in all_docs_in_temp_index[0].metadata:
                original_file_path_str = all_docs_in_temp_index[0].metadata['original_file_path']
                original_full_path = _APP_PATHS["PROJECT_ROOT"] / Path(original_file_path_str)

                if original_full_path.exists():
                    try:
                        # This is a simplified read. For PDFs/DOCX, you'd need the parser again.
                        # For now, just read text files.
                        if original_full_path.suffix.lower() == '.txt':
                            with open(original_full_path, 'r', encoding='utf-8') as f:
                                all_file_content = f.read()
                                # Attempt to extract a relevant line/paragraph around the page number
                                if page is not None:
                                    lines = all_file_content.splitlines()
                                    # Simple heuristic: get content around the estimated page start
                                    # This is very basic and won't work for all documents/page definitions
                                    lines_per_page = 50  # Estimate
                                    start_line = max(0, (page - 1) * lines_per_page)
                                    end_line = min(len(lines), start_line + lines_per_page)
                                    target_snippet_content = "\n".join(lines[start_line:end_line])
                                else:
                                    target_snippet_content = all_file_content[:500] + "..."  # First 500 chars
                                logger.info(f"Read snippet from original file: {original_full_path.name}")
                        else:
                            target_snippet_content = f"Preview not available for {original_full_path.suffix} from original file. Showing chunk content if found."
                            if found_doc_for_snippet:  # Fallback to chunk content if original file parsing is complex
                                target_snippet_content = found_doc_for_snippet.page_content
                            else:
                                target_snippet_content = "Snippet not found in FAISS or original file."

                    except Exception as e:
                        logger.error(f"Error reading original file {original_full_path.name}: {e}")
                        target_snippet_content = "Error reading original file for snippet."
                else:
                    logger.warning(f"Original file path {original_full_path} not found for snippet retrieval.")
                    target_snippet_content = "Original file not found for snippet."
            else:
                logger.warning("Original file path not found in metadata for snippet retrieval.")
                target_snippet_content = "Original file path missing from metadata for snippet."


    except Exception as e:
        logger.error(f"Error retrieving citation snippet from FAISS for {temp_doc_id}: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve snippet: {e}"
        )

    return {"snippet": target_snippet_content}