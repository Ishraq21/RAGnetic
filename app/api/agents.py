# app/api/agents.py
import os
import shutil
import yaml
import json
import asyncio
from datetime import datetime
from typing import List, Optional, Dict, Any

from fastapi import APIRouter, Depends, HTTPException, status, Body, BackgroundTasks, Query, File, UploadFile
from fastapi.responses import Response

from app.core.config import get_path_settings, get_api_key
from app.core.security import get_http_api_key, PermissionChecker, \
    get_current_user_from_api_key  # Import PermissionChecker and get_current_user_from_api_key
from app.core.rate_limit import rate_limiter as rate_limit_dep
from app.schemas.agent import AgentConfig, AgentInspectionResponse, ValidationReport, DocumentMetadata, ConnectionCheck, \
    VectorStoreConfig, AgentDeploymentStatus
from app.agents.config_manager import get_agent_configs, load_agent_config, save_agent_config
from app.pipelines.embed import embed_agent_data
from app.core.embed_config import get_embedding_model
from app.schemas.agent import DataSource
from app.schemas.security import User
from app.db import AsyncSessionLocal, get_db # Keep AsyncSessionLocal for now, but not directly used in background task
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, update, and_
from app.services.agent_manager import agent_manager
from app.services.agent_sync_service import agent_sync_service

import logging

# IMPORTS for inspect_agent dynamic vector store loading
from langchain_community.vectorstores import FAISS, Chroma
from langchain_qdrant import Qdrant
from langchain_pinecone import Pinecone as PineconeLangChain
from langchain_mongodb import MongoDBAtlasVectorSearch
from pinecone import Pinecone as PineconeClient
from pathlib import Path


# IMPORTS for connection checks
from sqlalchemy import create_engine, text
import requests
from urllib.parse import urlparse

logger = logging.getLogger("ragnetic")

# --- Path Settings ---
_APP_PATHS = get_path_settings()
_AGENTS_DIR = _APP_PATHS["AGENTS_DIR"]
_VECTORSTORE_DIR = _APP_PATHS["VECTORSTORE_DIR"]
_DATA_DIR = _APP_PATHS["DATA_DIR"]

# DB models for syncing deployments
from app.db.models import deployments_table, api_keys_table, agents_table





# --- API v1 Router for Agents ---
router = APIRouter(prefix="/api/v1/agents", tags=["Agents API"])




@router.get("", response_model=List[AgentConfig])
async def get_all_agents(
        current_user: User = Depends(PermissionChecker(["read:agents"]))

):
    """
    Retrieves the full configuration for all available agents.
    Requires: 'agent:read' permission.
    """
    try:
        agent_configs = get_agent_configs()
        return agent_configs
    except Exception as e:
        logger.error(f"API: Failed to get agent configs: {e}", exc_info=True)
        raise HTTPException(status_code=500,
                            detail="An unexpected error occurred while fetching all agent configurations. Please check server logs.")


# Alias to avoid potential client conflicts with parameterized routes
@router.get("/status-all", response_model=List[AgentDeploymentStatus])
async def get_all_agents_status_alias(
    search: Optional[str] = Query(None, description="Search agents by name, description, or tags"),
    status: Optional[str] = Query(None, description="Filter by agent status"),
    model: Optional[str] = Query(None, description="Filter by model name"),
    deployment_type: Optional[str] = Query(None, description="Filter by deployment type"),
    current_user: User = Depends(PermissionChecker(["read:agents"])),
    db: AsyncSession = Depends(get_db)
):
    return await get_all_agents_status(
        search=search,
        status=status,
        model=model,
        deployment_type=deployment_type,
        current_user=current_user,
        db=db
    )


@router.get("/{agent_name}", response_model=AgentConfig)
async def get_agent_by_name(
        agent_name: str,
        # Anyone can read agents, so basic authentication (API key exists) is enough
        current_user: User = Depends(PermissionChecker(["read:agents"])),
        db: AsyncSession = Depends(get_db)
):
    """
    Retrieves the full configuration for a single agent.
    Requires: 'agent:read' permission.
    """
    try:
        agent_config = load_agent_config(agent_name)
        
        # Get database timestamps
        agent_result = await db.execute(
            select(agents_table.c.created_at, agents_table.c.updated_at)
            .where(agents_table.c.name == agent_name)
        )
        agent_row = agent_result.fetchone()
        
        if agent_row:
            # Add database timestamps to the config
            agent_config.created_at = agent_row.created_at
            agent_config.updated_at = agent_row.updated_at
        
        return agent_config
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail=f"Agent '{agent_name}' not found.")
    except Exception as e:
        logger.error(f"API: Failed to get agent config for '{agent_name}': {e}", exc_info=True)
        raise HTTPException(status_code=500,
                            detail=f"An unexpected server error occurred while loading configuration for agent '{agent_name}'. Please check server logs.")


@router.post("", status_code=status.HTTP_201_CREATED, dependencies=[Depends(rate_limit_dep("agent_create", 5, 60))])
async def create_new_agent(
        config: AgentConfig = Body(...),
        bg: BackgroundTasks = BackgroundTasks(),
        current_user: User = Depends(PermissionChecker(["agent:create"])),
        db: AsyncSession = Depends(get_db)
):
    """
    Creates a new agent from a configuration payload and starts the data embedding process.
    Requires: 'agent:create' permission.
    """
    try:
        load_agent_config(config.name)
        raise HTTPException(
            status_code=409,
            detail=f"Agent '{config.name}' already exists. Use PUT to update."
        )
    except FileNotFoundError:
        pass

    try:

        save_agent_config(config)

        # Sync agent to database
        sync_result = await agent_sync_service.sync_agent_to_db(db, config, current_user.id)
        logger.info(f"Synced agent {config.name} to database: {sync_result}")

        # Pass the already acquired db session to the background task directly
        bg.add_task(embed_agent_data, config=config, db=db)


        response = {
            "status": "Agent config saved; embedding started.", 
            "agent": config.name,
        }
        

        logger.info(f"User '{current_user.username}' created agent '{config.name}'")
        return response
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"API: Error creating agent '{config.name}': {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"An unexpected server error occurred: {str(e)}")


@router.put("/{agent_name}", dependencies=[Depends(rate_limit_dep("agent_update", 10, 60))])
async def update_agent_by_name(
        agent_name: str,
        config: AgentConfig = Body(...),
        bg: BackgroundTasks = BackgroundTasks(),
        current_user: User = Depends(PermissionChecker(["agent:update"])),
        db: AsyncSession = Depends(get_db)
):
    """
    Updates an existing agent's configuration and triggers a re-embedding of its data.
    Requires: 'agent:update' permission.
    """
    if agent_name != config.name:
        raise HTTPException(status_code=400, detail="Agent name in path does not match agent name in body.")
    try:
        load_agent_config(agent_name)


        save_agent_config(config)

        # Sync agent to database
        sync_result = await agent_sync_service.sync_agent_to_db(db, config, current_user.id)
        logger.info(f"Synced agent {config.name} to database: {sync_result}")

        # Pass the already acquired db session to the background task directly
        bg.add_task(embed_agent_data, config=config, db=db)


        response = {
            "status": "Agent config updated; re-embedding started.", 
            "agent": config.name,
        }
        

        logger.info(f"User '{current_user.username}' updated agent '{agent_name}'")
        return response
    except HTTPException:
        raise
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail=f"Agent '{agent_name}' not found.")
    except Exception as e:
        logger.error(f"API: Error updating agent '{agent_name}': {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"An unexpected server error occurred: {str(e)}")

@router.delete("/{agent_name}", status_code=status.HTTP_204_NO_CONTENT, dependencies=[Depends(rate_limit_dep("agent_delete", 3, 60))])
async def delete_agent_by_name(
        agent_name: str,
        # Only users with 'agent:delete' permission can delete agents
        current_user: User = Depends(PermissionChecker(["agent:delete"])),
        db: AsyncSession = Depends(get_db)
):
    """
    Deletes an agent, its configuration, and all associated data.
    Requires: 'agent:delete' permission.
    """
    try:
        load_agent_config(agent_name)  # Ensure the agent exists

        # Delete from database first
        await agent_sync_service.delete_agent_from_db(db, agent_name)

        config_path = _AGENTS_DIR / f"{agent_name}.yaml"
        if os.path.exists(config_path):
            os.remove(config_path)

        vectorstore_path = _VECTORSTORE_DIR / f"{agent_name}"
        if os.path.exists(vectorstore_path):
            shutil.rmtree(vectorstore_path)

        logger.info(f"User '{current_user.username}' deleted agent '{agent_name}'.")
        return
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail=f"Agent '{agent_name}' not found.")
    except Exception as e:
        logger.error(f"API: Error deleting agent '{agent_name}': {e}", exc_info=True)
        raise HTTPException(status_code=500,
                            detail=f"An unexpected server error occurred during agent deletion for '{agent_name}'. Please check server logs.")


@router.get("/{agent_name}/inspection", response_model=AgentInspectionResponse,
            summary="Inspect agent config and data health")
async def inspect_agent_api(
        agent_name: str,
        show_documents_metadata: bool = Query(False, description="Display detailed metadata for ingested documents."),
        check_connections: bool = Query(False, description="Verify connectivity for each configured external source."),
        num_docs: int = Query(5, ge=1, description="Number of sample documents to retrieve."),
        # Users need 'agent:read' permission to inspect agents
        current_user: User = Depends(PermissionChecker(["agent:read"]))
) -> AgentInspectionResponse:
    """
    Inspects an agent's configuration, deployment status, and optionally its data sources.
    Requires: 'agent:read' permission.
    """
    # --- load config ---
    try:
        cfg = load_agent_config(agent_name)
    except FileNotFoundError:
        raise HTTPException(404, f"Agent '{agent_name}' not found.")
    except Exception as e:
        logger.error(f"API: Failed to load agent config for '{agent_name}': {e}", exc_info=True)
        raise HTTPException(500, "Error loading agent configuration.")

    # --- base status flags ---
    vs_dir = _VECTORSTORE_DIR / agent_name
    offline_marker = vs_dir / ".offline"
    is_deployed = vs_dir.exists()
    # Detect basic ingesting/online states. Ingesting detection can be improved later.
    is_ingesting = False
    # An agent is online only if it's deployed and not marked offline
    is_online = bool(is_deployed and not offline_marker.exists())

    validation_errors = []
    document_metadata = None
    vector_store_status = ConnectionCheck(status="SKIPPED", message="Not requested")
    sources_status: Dict[str, ConnectionCheck] = {}

    # --- vector store connectivity ---
    if show_documents_metadata:
        vector_store_status.status = "PENDING"
        try:
            vectorstore_path = _VECTORSTORE_DIR / agent_name
            if not os.path.exists(vectorstore_path):
                vector_store_status = ConnectionCheck(status="FAIL",
                                                      message="Vector store not found. Agent not deployed.")
                validation_errors.append("Vector store not found. Agent not deployed.")
            else:
                embeddings = get_embedding_model(cfg.embedding_model)
                cls_map = {'faiss': FAISS, 'chroma': Chroma, 'qdrant': Qdrant, 'pinecone': PineconeLangChain,
                           'mongodb_atlas': MongoDBAtlasVectorSearch}
                db_type = cfg.vector_store.type
                db = None

                if db_type in ('pinecone', 'mongodb_atlas', 'qdrant'):
                    key_map = {'pinecone': 'pinecone', 'mongodb_atlas': 'mongodb', 'qdrant': 'qdrant'}
                    key = get_api_key(key_map[db_type])
                    if not key:
                        vector_store_status = ConnectionCheck(status="FAIL", message=f"Missing API key for {db_type}.")
                        validation_errors.append(f"Missing API key for {db_type}.")
                    else:
                        if db_type == 'pinecone':
                            PineconeClient(api_key=key)
                            idx = cfg.vector_store.pinecone_index_name
                            db = await asyncio.to_thread(PineconeLangChain.from_existing_index, index_name=idx,
                                                         embedding=embeddings)
                        elif db_type == 'qdrant':
                            vs_cfg = cfg.vector_store
                            db = await asyncio.to_thread(Qdrant, client=None, collection_name=agent_name,
                                                         embeddings=embeddings,
                                                         host=vs_cfg.qdrant_host, port=vs_cfg.qdrant_port,
                                                         prefer_grpc=True)
                        else:  # mongodb_atlas
                            vs = cfg.vector_store
                            db = await asyncio.to_thread(MongoDBAtlasVectorSearch.from_connection_string,
                                                         get_api_key("mongodb"), vs.mongodb_db_name,
                                                         vs.mongodb_collection_name,
                                                         embeddings, vs.mongodb_index_name)
                    if db:
                        vector_store_status.status = "PASS"
                elif db_type == 'faiss':
                    db = await asyncio.to_thread(FAISS.load_local, str(vectorstore_path), embeddings,
                                                 allow_dangerous_deserialization=True)
                    if db:
                        vector_store_status.status = "PASS"
                elif db_type == 'chroma':
                    db = await asyncio.to_thread(Chroma, persist_directory=str(vectorstore_path),
                                                 embedding_function=embeddings)
                    if db:
                        vector_store_status.status = "PASS"
                else:
                    vector_store_status = ConnectionCheck(status="FAIL",
                                                          message=f"Unsupported vector store type '{db_type}'.")
                    validation_errors.append(f"Unsupported vector store type '{db_type}'.")

            if vector_store_status.status == "PASS":
                docs = await asyncio.to_thread(db.similarity_search_with_score, "document", k=num_docs)
                document_metadata = [
                    DocumentMetadata(page_content=d.page_content, metadata=d.metadata, score=s)
                    for d, s in docs
                ]
        except Exception as e:
            if vector_store_status.status == "PENDING":
                vector_store_status = ConnectionCheck(status="FAIL",
                                                      message="Failed to connect or retrieve documents from vector store.",
                                                      detail="Check server logs for details.")
                validation_errors.append("Document inspection failed. Check server logs for details.")
            logger.error(f"API: An unexpected error occurred during document inspection for '{agent_name}': {e}",
                         exc_info=True)

    # --- source connectivity ---
    if check_connections:
        for idx, source in enumerate(cfg.sources or [], 1):
            info_key = f"Source {idx} ({source.type})"
            status_message = "UNKNOWN"
            status_detail = None
            try:
                if source.type == "db" and source.db_connection:
                    eng = create_engine(source.db_connection, connect_args={"connect_timeout": 5})
                    with eng.connect() as conn:
                        conn.execute(text("SELECT 1"))
                    status_message = "PASS"
                    eng.dispose()
                elif source.type in ("url", "api") and source.url:
                    requests.head(source.url, timeout=5).raise_for_status()
                    status_message = "PASS"
                elif source.type == "gdoc":
                    if get_api_key("google"):
                        status_message = "PASS"
                    else:
                        status_message = "FAIL"
                        status_detail = "Missing Google credentials."
                        validation_errors.append(status_detail)
                else:
                    status_message = "SKIP"
            except Exception as e:
                status_message = "FAIL"
                status_detail = "An unexpected error occurred during connection check."
                validation_errors.append(f"Connection check for {source.type} failed. Check server logs for details.")

            sources_status[info_key] = ConnectionCheck(status=status_message, message=status_detail)

    # --- final response build ---
    validation_report = ValidationReport(
        is_valid=not validation_errors,
        errors=validation_errors
    )

    return AgentInspectionResponse(
        name=agent_name,
        is_deployed=is_deployed,
        is_ingesting=is_ingesting,
        is_online=is_online,
        vector_store_status=vector_store_status,
        sources_status=sources_status,
        document_metadata=document_metadata,
        validation_report=validation_report
    )


@router.post("/{agent_name}/deploy", response_model=dict, dependencies=[Depends(rate_limit_dep("agent_deploy", 5, 60))])
async def deploy_agent(
        agent_name: str,
        bg: BackgroundTasks = BackgroundTasks(),
        current_user: User = Depends(PermissionChecker(["agent:update"])),
        db: AsyncSession = Depends(get_db)
):
    """
    Deploy an agent by (re)running the embedding pipeline and marking it online.
    This ensures the vectorstore directory exists and clears any offline marker.
    """
    try:
        # Ensure config exists
        cfg = load_agent_config(agent_name)

        # Ensure vectorstore directory exists
        vs_dir = _VECTORSTORE_DIR / agent_name
        os.makedirs(vs_dir, exist_ok=True)

        # Remove offline marker if present
        offline_marker = vs_dir / ".offline"
        if offline_marker.exists():
            try:
                os.remove(offline_marker)
            except Exception:
                pass

        # Trigger (re)embedding in background so status becomes deployed when finished
        bg.add_task(embed_agent_data, config=cfg, db=db)

        # Activate any existing deployments for this agent and ensure API keys active
        try:
            result = await db.execute(select(agents_table.c.id).where(agents_table.c.name == agent_name))
            agent_row = result.fetchone()
            if agent_row:
                agent_id = agent_row.id
                await db.execute(
                    update(deployments_table)
                    .where(deployments_table.c.agent_id == agent_id)
                    .values(status="active")
                )
                # Activate associated API keys
                # We need to look up deployments to get api_key_id
                dep_rows = (await db.execute(select(deployments_table.c.api_key_id).where(deployments_table.c.agent_id == agent_id))).fetchall()
                for dep in dep_rows:
                    if dep.api_key_id:
                        await db.execute(
                            update(api_keys_table).where(api_keys_table.c.id == dep.api_key_id).values(is_active=True)
                        )
                await db.commit()
        except Exception:
            # Non-fatal; log and continue
            logger.warning("Failed to activate deployments during deploy sync.", exc_info=True)

        return {"status": "ok", "message": f"Deployment started for '{agent_name}'."}
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail=f"Agent '{agent_name}' not found.")
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"API: Error deploying agent '{agent_name}': {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to deploy agent '{agent_name}'.")


@router.post("/{agent_name}/shutdown", response_model=dict, dependencies=[Depends(rate_limit_dep("agent_shutdown", 5, 60))])
async def shutdown_agent(
        agent_name: str,
        current_user: User = Depends(PermissionChecker(["agent:update"])),
        db: AsyncSession = Depends(get_db)
):
    """
    Gracefully take an agent offline without deleting its data by placing an '.offline' marker
    in its vectorstore directory. This keeps it deployed but marked offline.
    """
    try:
        # Ensure config exists
        _ = load_agent_config(agent_name)

        vs_dir = _VECTORSTORE_DIR / agent_name
        if not vs_dir.exists():
            # Not deployed yet; nothing to shutdown
            raise HTTPException(status_code=409, detail=f"Agent '{agent_name}' is not deployed.")

        offline_marker = vs_dir / ".offline"
        # Create or update marker
        with open(offline_marker, "w", encoding="utf-8") as f:
            f.write(f"offline set at {datetime.utcnow().isoformat()}Z")

        # Deactivate any existing deployments for this agent and disable API keys
        try:
            result = await db.execute(select(agents_table.c.id).where(agents_table.c.name == agent_name))
            agent_row = result.fetchone()
            if agent_row:
                agent_id = agent_row.id
                
                # Update agent status to stopped
                await db.execute(
                    update(agents_table)
                    .where(agents_table.c.name == agent_name)
                    .values(status="stopped", last_updated=datetime.utcnow())
                )
                
                # Update deployments table
                await db.execute(
                    update(deployments_table)
                    .where(deployments_table.c.agent_id == agent_id)
                    .values(status="inactive")
                )
                dep_rows = (await db.execute(select(deployments_table.c.api_key_id).where(deployments_table.c.agent_id == agent_id))).fetchall()
                for dep in dep_rows:
                    if dep.api_key_id:
                        await db.execute(
                            update(api_keys_table).where(api_keys_table.c.id == dep.api_key_id).values(is_active=False)
                        )
                await db.commit()
                logger.info(f"Agent {agent_name} status updated to stopped in database")
        except Exception:
            logger.warning("Failed to deactivate deployments during shutdown sync.", exc_info=True)

        return {"status": "ok", "message": f"Agent '{agent_name}' is now offline."}
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail=f"Agent '{agent_name}' not found.")
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"API: Error shutting down agent '{agent_name}': {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to shutdown agent '{agent_name}'.")




@router.post("/upload-file", summary="Upload a local file for ingestion", response_model=Dict[str, str], dependencies=[Depends(rate_limit_dep("agent_upload", 20, 60))])
async def upload_file_for_ingestion(
    file: UploadFile = File(..., description="The file to upload."),
    current_user: User = Depends(PermissionChecker(["agent:create", "agent:update"]))
) -> Dict[str, str]:
    """
    Receives a file upload, saves it to a user-specific location, and returns its server path.
    This path can then be used in the agent's data source configuration for 'local' type.
    Requires 'agent:create' or 'agent:update' permission.
    Files are stored per-user for isolation and security.
    """
    # Create user-specific upload directory
    user_upload_dir = _DATA_DIR / "uploads" / f"user_{current_user.id}"
    user_upload_dir.mkdir(parents=True, exist_ok=True)

    # Sanitize filename to prevent directory traversal
    filename = Path(file.filename).name
    # Add timestamp to prevent filename conflicts
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_filename = f"{timestamp}_{filename}"
    file_location = user_upload_dir / safe_filename
    
    logger.info(f"Uploading {filename} to {file_location} for user {current_user.id}")

    try:
        with open(file_location, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        logger.info(f"Successfully uploaded file: {file_location}")
        # Return the absolute path as expected by the frontend
        return {"file_path": str(file_location.resolve())}
    except Exception as e:
        logger.error(f"Error uploading file {filename}: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Could not upload file: {e}"
        )
    finally:
        await file.close()


# --- Agent State Machine Endpoints ---

@router.get("/status", response_model=List[AgentDeploymentStatus])
async def get_all_agents_status(
    search: Optional[str] = Query(None, description="Search agents by name, description, or tags"),
    status: Optional[str] = Query(None, description="Filter by agent status"),
    model: Optional[str] = Query(None, description="Filter by model name"),
    deployment_type: Optional[str] = Query(None, description="Filter by deployment type"),
    current_user: User = Depends(PermissionChecker(["read:agents"])),
    db: AsyncSession = Depends(get_db)
):
    """Get deployment status for all agents with optional filtering."""
    try:
        agents_status = await agent_manager.get_all_agents_status(db, current_user)
        
        # Apply filters
        if search:
            search_lower = search.lower()
            agents_status = [
                agent for agent in agents_status
                if (search_lower in agent.get("name", "").lower() or
                    search_lower in agent.get("description", "").lower() or
                    any(search_lower in tag.lower() for tag in (agent.get("tags") or [])))
            ]
        
        if status:
            agents_status = [agent for agent in agents_status if agent.get("status") == status]
        
        if model:
            agents_status = [agent for agent in agents_status if agent.get("model_name") == model]
        
        if deployment_type:
            agents_status = [agent for agent in agents_status if agent.get("deployment_type") == deployment_type]
        
        
        return agents_status
    except Exception as e:
        logger.error(f"Failed to get agents status: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get agents status: {e}"
        )


@router.get("/{agent_name}/status", response_model=dict)
async def get_agent_status(
    agent_name: str,
    current_user: User = Depends(PermissionChecker(["read:agents"])),
    db: AsyncSession = Depends(get_db)
):
    """Get deployment status for a specific agent with actual deployment status."""
    try:
        # Get database status
        result = await db.execute(
            select(agents_table.c.status, agents_table.c.last_updated)
            .where(agents_table.c.name == agent_name)
        )
        agent_row = result.fetchone()
        
        if not agent_row:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Agent {agent_name} not found"
            )
        
        db_status = agent_row.status
        last_updated = agent_row.last_updated
        
        # Check actual deployment status
        vs_dir = _VECTORSTORE_DIR / agent_name
        offline_marker = vs_dir / ".offline"
        
        actual_status = db_status
        if vs_dir.exists():
            if offline_marker.exists():
                actual_status = "stopped"
            else:
                actual_status = "deployed"
        else:
            actual_status = "stopped"
        
        return {
            "agent_name": agent_name,
            "database_status": db_status,
            "actual_status": actual_status,
            "last_updated": last_updated.isoformat() if last_updated else None,
            "vectorstore_exists": vs_dir.exists(),
            "offline_marker_exists": offline_marker.exists() if vs_dir.exists() else False
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get agent status for {agent_name}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get agent status: {e}"
        )

@router.post("/{agent_name}/deploy-state", response_model=dict)
async def deploy_agent_state(
    agent_name: str,
    current_user: User = Depends(PermissionChecker(["write:agents"])),
    db: AsyncSession = Depends(get_db)
):
    """Deploy an agent using the state machine."""
    try:
        success = await agent_manager.deploy_agent(db, agent_name)
        if not success:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Failed to deploy agent {agent_name}"
            )
        return {"message": f"Agent {agent_name} deployment initiated", "status": "success"}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to deploy agent {agent_name}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to deploy agent: {e}"
        )

@router.post("/{agent_name}/stop-state", response_model=dict)
async def stop_agent_state(
    agent_name: str,
    current_user: User = Depends(PermissionChecker(["write:agents"])),
    db: AsyncSession = Depends(get_db)
):
    """Stop an agent using the state machine."""
    try:
        success = await agent_manager.stop_agent(db, agent_name)
        if not success:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Failed to stop agent {agent_name}"
            )
        return {"message": f"Agent {agent_name} stopped", "status": "success"}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to stop agent {agent_name}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to stop agent: {e}"
        )

@router.post("/{agent_name}/resume", response_model=dict)
async def resume_agent_state(
    agent_name: str,
    current_user: User = Depends(PermissionChecker(["write:agents"])),
    db: AsyncSession = Depends(get_db)
):
    """Resume an idle agent."""
    try:
        success = await agent_manager.resume_agent(db, agent_name)
        if not success:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Failed to resume agent {agent_name}"
            )
        return {"message": f"Agent {agent_name} resumed", "status": "success"}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to resume agent {agent_name}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to resume agent: {e}"
        )

@router.post("/{agent_name}/retry", response_model=dict)
async def retry_agent_deployment(
    agent_name: str,
    current_user: User = Depends(PermissionChecker(["write:agents"])),
    db: AsyncSession = Depends(get_db)
):
    """Retry deployment for an agent in error state."""
    try:
        success = await agent_manager.retry_deployment(db, agent_name)
        if not success:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Failed to retry deployment for agent {agent_name}"
            )
        return {"message": f"Agent {agent_name} deployment retry initiated", "status": "success"}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to retry deployment for agent {agent_name}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retry deployment: {e}"
        )

@router.get("/{agent_name}/badge-info", response_model=dict)
async def get_agent_badge_info(
    agent_name: str,
    current_user: User = Depends(PermissionChecker(["read:agents"])),
    db: AsyncSession = Depends(get_db)
):
    """Get badge styling information for an agent."""
    try:
        agent_status = await agent_manager.get_agent_status(db, agent_name)
        if not agent_status:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Agent {agent_name} not found"
            )
        
        badge_info = agent_manager.get_status_badge_info(agent_status["status"])
        actions = agent_manager.get_available_actions(agent_status["status"])
        
        return {
            "badge_info": badge_info,
            "actions": actions,
            "status": agent_status["status"]
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get badge info for agent {agent_name}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get badge info: {e}"
        )

@router.post("/sync", response_model=dict)
async def sync_all_agents(
    current_user: User = Depends(PermissionChecker(["write:agents"])),
    db: AsyncSession = Depends(get_db)
):
    """Sync all YAML agents to database."""
    try:
        sync_result = await agent_sync_service.sync_all_agents_to_db(db, current_user.id)
        return {
            "message": "Agent sync completed",
            "results": sync_result
        }
    except Exception as e:
        logger.error(f"Failed to sync agents: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to sync agents: {e}"
        )

@router.put("/{agent_name}/update-fields", response_model=dict)
async def update_agent_fields(
    agent_name: str,
    fields: Dict[str, Any] = Body(...),
    current_user: User = Depends(PermissionChecker(["write:agents"])),
    db: AsyncSession = Depends(get_db)
):
    """Update specific fields of an agent."""
    try:
        from sqlalchemy import update
        
        # Allowed fields that can be updated
        allowed_fields = {
            'display_name', 'description', 'embedding_model', 
            'deployment_type', 'tags'
        }
        
        # Filter to only allowed fields
        update_data = {k: v for k, v in fields.items() if k in allowed_fields}
        
        if not update_data:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="No valid fields provided for update"
            )
        
        # Add updated_at timestamp
        update_data['updated_at'] = datetime.utcnow()
        
        stmt = update(agents_table).where(agents_table.c.name == agent_name).values(**update_data)
        result = await db.execute(stmt)
        await db.commit()
        
        if result.rowcount == 0:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Agent {agent_name} not found"
            )
        
        return {"message": f"Agent {agent_name} updated successfully"}
    except HTTPException:
        raise
    except Exception as e:
        await db.rollback()
        logger.error(f"Failed to update agent {agent_name}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to update agent: {e}"
        )

@router.get("/filters/options", response_model=dict)
async def get_filter_options(
    current_user: User = Depends(PermissionChecker(["read:agents"])),
    db: AsyncSession = Depends(get_db)
):
    """Get available filter options for agents."""
    try:
        from sqlalchemy import select, distinct
        
        # Get unique values for each filter
        statuses = await db.execute(select(distinct(agents_table.c.status)))
        models = await db.execute(select(distinct(agents_table.c.model_name)))
        deployment_types = await db.execute(select(distinct(agents_table.c.deployment_type)))
        
        return {
            "statuses": [row[0] for row in statuses.fetchall() if row[0]],
            "models": [row[0] for row in models.fetchall() if row[0]],
            "deployment_types": [row[0] for row in deployment_types.fetchall() if row[0]]
        }
    except Exception as e:
        logger.error(f"Failed to get filter options: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get filter options: {e}"
        )


# Cost data is now provided by the analytics endpoint: /api/v1/analytics/usage-summary


# Usage data is now provided by the analytics endpoint: /api/v1/analytics/usage-summary


@router.get("/{agent_name}/yaml")
async def get_agent_yaml(
    agent_name: str,
    current_user: User = Depends(PermissionChecker(["read:agents"])),
    db: AsyncSession = Depends(get_db)
):
    """
    Get YAML configuration for a specific agent.
    Requires: 'read:agents' permission.
    """
    try:
        # Check if agent exists and load config
        try:
            agent_config = load_agent_config(agent_name)
            if not agent_config:
                raise HTTPException(status_code=404, detail=f"Agent '{agent_name}' not found.")
        except FileNotFoundError:
            raise HTTPException(status_code=404, detail=f"Agent '{agent_name}' not found.")
        
        # Convert Pydantic model to clean dictionary, then to YAML
        config_dict = agent_config.model_dump(exclude_unset=True)
        yaml_content = yaml.dump(config_dict, default_flow_style=False, sort_keys=False)
        
        return Response(
            content=yaml_content,
            media_type="text/yaml",
            status_code=200
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get YAML for agent '{agent_name}': {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to retrieve YAML for agent '{agent_name}': {str(e)}"
        )


@router.get("/{agent_name}/logs")
async def get_agent_logs(
    agent_name: str,
    lines: int = Query(100, ge=1, le=1000, description="Number of log lines to retrieve"),
    current_user: User = Depends(PermissionChecker(["read:agents"])),
    db: AsyncSession = Depends(get_db)
):
    """
    Get logs for a specific agent.
    Requires: 'read:agents' permission.
    """
    try:
        # Check if agent exists
        try:
            agent_config = load_agent_config(agent_name)
            if not agent_config:
                raise HTTPException(status_code=404, detail=f"Agent '{agent_name}' not found.")
        except FileNotFoundError:
            raise HTTPException(status_code=404, detail=f"Agent '{agent_name}' not found.")
        
        # Try to get logs from various sources
        logs_content = ""
        
        # 1. Try to get logs from agent's log file if it exists
        log_file_path = f"logs/{agent_name}.log"
        if os.path.exists(log_file_path):
            try:
                with open(log_file_path, 'r', encoding='utf-8') as f:
                    all_lines = f.readlines()
                    recent_lines = all_lines[-lines:] if len(all_lines) > lines else all_lines
                    logs_content = ''.join(recent_lines)
            except Exception as e:
                logger.warning(f"Failed to read agent log file: {e}")
        
        # 2. If no agent-specific logs, try to get from main application logs
        if not logs_content:
            main_log_path = "logs/ragnetic_app.log"
            if os.path.exists(main_log_path):
                try:
                    with open(main_log_path, 'r', encoding='utf-8') as f:
                        all_lines = f.readlines()
                        # Filter for agent-related logs
                        agent_lines = [line for line in all_lines if agent_name.lower() in line.lower()]
                        recent_lines = agent_lines[-lines:] if len(agent_lines) > lines else agent_lines
                        logs_content = ''.join(recent_lines)
                except Exception as e:
                    logger.warning(f"Failed to read main log file: {e}")
        
        # 3. If still no logs, return empty state
        if not logs_content.strip():
            return Response(
                content="No logs available for this agent yet.",
                media_type="text/plain",
                status_code=200
            )
        
        return Response(
            content=logs_content,
            media_type="text/plain",
            status_code=200
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get logs for agent '{agent_name}': {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to retrieve logs for agent '{agent_name}': {str(e)}"
        )