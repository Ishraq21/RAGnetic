import asyncio
import logging
from typing import Optional, Dict, Any, List, Coroutine
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException, status, Query
from sqlalchemy.ext.asyncio import AsyncSession
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from pydantic import BaseModel, Field

from app.db import get_db
from app.core.security import get_http_api_key, get_current_user_from_api_key
# The new DAO function
from app.db.dao import get_document_chunk, get_document_chunks
from app.schemas.security import User
from app.core.config import get_path_settings
from app.core.embed_config import get_embedding_model

logger = logging.getLogger("ragnetic")

router = APIRouter(prefix="/api/v1/chat", tags=["Chat"])

# Define paths for temporary storage
_APP_PATHS = get_path_settings()
_TEMP_CHAT_UPLOADS_DIR = _APP_PATHS["TEMP_CLONES_DIR"] / "chat_uploads"
_TEMP_VECTORSTORE_DIR = _APP_PATHS["VECTORSTORE_DIR"] / "temp_chat_data"


class CitationSnippet(BaseModel):
    id: int = Field(..., description="The ID of the document chunk.")
    snippet: str = Field(..., description="The content snippet of the chunk.")
    document_name: str = Field(..., description="The name of the source document.")
    page_number: Optional[int] = Field(None, description="The page number in the source document.")


@router.get("/citation-snippet",  response_model=CitationSnippet)
async def get_citation_snippet(
        chunk_id: int = Query(..., description="The ID of the document chunk."),
        current_user: User = Depends(get_current_user_from_api_key),
        db: AsyncSession = Depends(get_db)
) -> CitationSnippet:
    """
    Retrieves a single text snippet for a given citation from the database.
    """
    logger.info(f"User {current_user.id} requesting citation snippet for chunk_id: {chunk_id}")

    chunk = await get_document_chunk(db, chunk_id)

    if not chunk:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND,
                            detail="Citation source not found.")

    return CitationSnippet(
        id=chunk["id"],
        snippet=chunk["content"],
        document_name=chunk["document_name"],
        page_number=chunk.get("page_number")
    )


@router.get("/citation-snippets", response_model=List[CitationSnippet])
async def get_citation_snippets(
        chunk_ids: List[int] = Query(..., description="The IDs of the document chunks."),
        current_user: User = Depends(get_current_user_from_api_key),
        db: AsyncSession = Depends(get_db)
) -> List[CitationSnippet]:
    """
    Retrieves text snippets and metadata for a list of citation chunk IDs from the database.
    """
    logger.info(f"User {current_user.id} requesting citation snippets for chunk_ids: {chunk_ids}")

    # Use the new DAO function to fetch multiple chunks efficiently
    chunks = await get_document_chunks(db, chunk_ids)

    if not chunks:
        # It's better to return an empty list than a 404 if no chunks match
        return []

    # Map the database results to the Pydantic model for the response
    response_list = [
        CitationSnippet(
            id=c['id'],
            content=c['content'],
            document_name=c['document_name'],
            page_number=c.get('page_number')
        )
        for c in chunks
    ]

    return response_list