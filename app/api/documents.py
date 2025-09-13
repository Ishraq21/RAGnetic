# app/api/documents.py

import logging
import os
from typing import Dict, Any

from fastapi import APIRouter, UploadFile, File, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.security import get_current_user_from_api_key
from app.db import get_db
from app.services.temporary_document_service import TemporaryDocumentService
from app.agents.config_manager import get_agent_configs

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/documents", tags=["Documents API"])

def _get_default_agent_config():
    """Get a default agent config for document processing"""
    try:
        # Try to get a default agent or create a minimal config
        agents = get_agent_configs()
        if agents:
            return agents[0]
        else:
            # Create minimal config if no agents exist
            class DefaultConfig:
                embedding_model = os.getenv("RAGNETIC_EMBEDDING_MODEL", "text-embedding-3-small")
            return DefaultConfig()
    except Exception as e:
        logger.warning(f"Could not get default agent config: {e}")
        # Fallback minimal config
        class DefaultConfig:
            embedding_model = os.getenv("RAGNETIC_EMBEDDING_MODEL", "text-embedding-3-small")
        return DefaultConfig()


@router.post("/upload", summary="Upload a temporary document for lambda processing")
async def upload_document(
    file: UploadFile = File(...),
    thread_id: str = "lambda-upload",
    current_user = Depends(get_current_user_from_api_key),
    db: AsyncSession = Depends(get_db)
) -> Dict[str, Any]:
    """
    Upload a document for temporary processing with the lambda tool.
    
    Returns:
    - temp_doc_id: ID to use in lambda tool inputs
    - filename: Original filename
    - file_size: Size in bytes
    """
    try:
        # Get default agent config for document processing
        agent_config = _get_default_agent_config()
        
        # Initialize temporary document service
        doc_service = TemporaryDocumentService(agent_config)
        
        # Process and store the document
        result = await doc_service.process_and_store_temp_document(
            file=file,
            user_id=current_user.id,
            thread_id=thread_id,
            db=db
        )
        
        logger.info(f"Document uploaded successfully: {result.file_name} -> {result.temp_doc_id}")
        
        return {
            "temp_doc_id": result.temp_doc_id,
            "filename": result.file_name,
            "file_size": result.file_size,
            "message": "Document uploaded successfully",
            "usage_note": "Use 'temp_doc_id' in lambda tool inputs array"
        }
        
    except ValueError as e:
        logger.error(f"Document upload validation error: {e}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Document upload failed: {str(e)}"
        )
    except Exception as e:
        logger.error(f"Document upload error: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error during document upload"
        )


@router.get("/temp/{temp_doc_id}", summary="Get temporary document metadata")
async def get_temp_document(
    temp_doc_id: str,
    current_user = Depends(get_current_user_from_api_key),
    db: AsyncSession = Depends(get_db)
) -> Dict[str, Any]:
    """
    Get metadata for a temporary document by its temp_doc_id.
    """
    try:
        from app.db.dao import get_temp_document_by_id
        
        doc = await get_temp_document_by_id(db, temp_doc_id, current_user.id)
        if not doc:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Temporary document not found: {temp_doc_id}"
            )
        
        return {
            "temp_doc_id": doc["temp_doc_id"],
            "filename": doc["original_name"],
            "file_size": doc["file_size"],
            "created_at": doc["created_at"].isoformat() if doc.get("created_at") else None,
            "user_id": doc["user_id"],
            "thread_id": doc["thread_id"]
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error retrieving temp document {temp_doc_id}: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error retrieving document"
        )