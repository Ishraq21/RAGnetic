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


# app/api/citations.py

@router.get("/citation-snippet")
async def get_citation_snippet(
        temp_doc_id: str = Query(..., description="The unique ID of the temporary document."),
        doc_name: str = Query(..., description="The original name of the document."),
        page: Optional[int] = Query(None, description="The page number of the snippet (if applicable)."),
        current_user: User = Depends(get_current_user_from_api_key),
) -> Dict[str, str]:
    """
    Retrieves a text snippet for a given citation from a temporary document.
    """
    logger.info(f"User {current_user.id} requesting citation snippet for temp_doc_id: {temp_doc_id}")

    temp_vector_store_path = _TEMP_VECTORSTORE_DIR / temp_doc_id
    if not temp_vector_store_path.exists():
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND,
                            detail="Citation source not found or has been purged.")

    try:
        # Load the specific FAISS index for this temporary document
        embeddings = get_embedding_model()
        vectorstore = await asyncio.to_thread(
            FAISS.load_local, str(temp_vector_store_path), embeddings, allow_dangerous_deserialization=True
        )

        # The most reliable way to get all documents is to access the docstore directly
        if not hasattr(vectorstore, 'docstore') or not hasattr(vectorstore.docstore, '_docs'):
            raise HTTPException(status_code=status.HTTP_501_NOT_IMPLEMENTED,
                                detail="Vector store does not support direct document access.")

        all_docs_in_temp_index: List[Document] = list(vectorstore.docstore._docs.values())

        # Find the specific chunk that matches the citation criteria
        for doc in all_docs_in_temp_index:
            # Check for a match on temp_doc_id, doc_name, and page number
            if (doc.metadata.get('temp_doc_id') == temp_doc_id and
                    doc.metadata.get('doc_name') == doc_name and
                    (page is None or doc.metadata.get('page_number') == page)):
                logger.info(f"Found matching chunk for snippet: {doc_name} (Page: {page})")
                return {"snippet": doc.page_content}

        # If no chunk was found after checking all documents
        logger.warning(f"No exact chunk found for {doc_name} (Page: {page}) in temp_doc_id: {temp_doc_id}")
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND,
                            detail="Snippet for the specified page not found in the document.")

    except Exception as e:
        logger.error(f"Error retrieving citation snippet for {temp_doc_id}: {e}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Failed to retrieve snippet.")