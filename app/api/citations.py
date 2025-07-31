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
from app.db.dao import get_document_chunk
from app.schemas.security import User
from app.core.config import get_path_settings
from app.core.embed_config import get_embedding_model

logger = logging.getLogger("ragnetic")

router = APIRouter(prefix="/api/v1/chat", tags=["Chat"])

# Define paths for temporary storage
_APP_PATHS = get_path_settings()
_TEMP_CHAT_UPLOADS_DIR = _APP_PATHS["TEMP_CLONES_DIR"] / "chat_uploads"
_TEMP_VECTORSTORE_DIR = _APP_PATHS["VECTORSTORE_DIR"] / "temp_chat_data"



@router.get("/citation-snippet")
async def get_citation_snippet(
        chunk_id: int = Query(..., description="The ID of the document chunk."),
        current_user: User = Depends(get_current_user_from_api_key),
        db: AsyncSession = Depends(get_db)
) -> Dict[str, str]:
    """
    Retrieves a text snippet for a given citation from the database.
    """
    logger.info(f"User {current_user.id} requesting citation snippet for chunk_id: {chunk_id}")

    chunk = await get_document_chunk(db, chunk_id)

    if not chunk:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND,
                            detail="Citation source not found.")

    return {"snippet": chunk['content']}