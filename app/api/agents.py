# app/api/agents.py
import os
import shutil
import yaml
import json
import asyncio
from typing import List, Optional, Dict, Any

from fastapi import APIRouter, Depends, HTTPException, status, Body, BackgroundTasks, Query

from app.core.config import get_path_settings, get_api_key
from app.core.security import get_http_api_key, PermissionChecker, \
    get_current_user_from_api_key  # Import PermissionChecker and get_current_user_from_api_key
from app.schemas.agent import AgentConfig, AgentInspectionResponse, ValidationReport, DocumentMetadata, ConnectionCheck, \
    VectorStoreConfig
from app.agents.config_manager import get_agent_configs, load_agent_config, save_agent_config
from app.pipelines.embed import embed_agent_data
from app.core.embed_config import get_embedding_model
from app.schemas.agent import DataSource
from app.schemas.security import User  # Import User schema
import logging

# IMPORTS for inspect_agent dynamic vector store loading
from langchain_community.vectorstores import FAISS, Chroma
from langchain_qdrant import Qdrant
from langchain_pinecone import Pinecone as PineconeLangChain
from langchain_mongodb import MongoDBAtlasVectorSearch
from pinecone import Pinecone as PineconeClient

# IMPORTS for connection checks
from sqlalchemy import create_engine, text
import requests
from urllib.parse import urlparse

logger = logging.getLogger("ragnetic")

# --- Path Settings ---
_APP_PATHS = get_path_settings()
_AGENTS_DIR = _APP_PATHS["AGENTS_DIR"]
_VECTORSTORE_DIR = _APP_PATHS["VECTORSTORE_DIR"]

# --- API v1 Router for Agents ---
router = APIRouter(prefix="/api/v1/agents", tags=["Agents API"])


@router.get("", response_model=List[AgentConfig])
async def get_all_agents(
        # Anyone can read agents, so basic authentication (API key exists) is enough
        api_key: str = Depends(get_http_api_key)
):
    """
    Retrieves the full configuration for all available agents.
    Requires: Valid API Key.
    """
    try:
        agent_configs = get_agent_configs()
        return agent_configs
    except Exception as e:
        logger.error(f"API: Failed to get agent configs: {e}", exc_info=True)
        raise HTTPException(status_code=500,
                            detail="An unexpected error occurred while fetching all agent configurations. Please check server logs.")


@router.get("/{agent_name}", response_model=AgentConfig)
async def get_agent_by_name(
        agent_name: str,
        # Anyone can read agents, so basic authentication (API key exists) is enough
        api_key: str = Depends(get_http_api_key)
):
    """
    Retrieves the full configuration for a single agent.
    Requires: Valid API Key.
    """
    try:
        agent_config = load_agent_config(agent_name)
        return agent_config
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail=f"Agent '{agent_name}' not found.")
    except Exception as e:
        logger.error(f"API: Failed to get agent config for '{agent_name}': {e}", exc_info=True)
        raise HTTPException(status_code=500,
                            detail=f"An unexpected server error occurred while loading configuration for agent '{agent_name}'. Please check server logs.")


@router.post("", status_code=status.HTTP_201_CREATED)
async def create_new_agent(
        config: AgentConfig = Body(...),
        bg: BackgroundTasks = BackgroundTasks(),
        # Only users with 'agent:create' permission can create agents
        current_user: User = Depends(PermissionChecker(["agent:create"]))
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
        pass  # This is the expected case for a new agent

    try:
        save_agent_config(config)
        bg.add_task(embed_agent_data, config)
        logger.info(f"User '{current_user.username}' created agent '{config.name}'.")
        return {"status": "Agent config saved; embedding started.", "agent": config.name}
    except Exception as e:
        logger.error(f"API: Error creating agent '{config.name}': {e}", exc_info=True)
        raise HTTPException(status_code=500,
                            detail=f"An unexpected server error occurred during agent creation for '{config.name}'. Please check server logs.")


@router.put("/{agent_name}")
async def update_agent_by_name(
        agent_name: str,
        config: AgentConfig = Body(...),
        bg: BackgroundTasks = BackgroundTasks(),
        # Only users with 'agent:update' permission can update agents
        current_user: User = Depends(PermissionChecker(["agent:update"]))
):
    """
    Updates an existing agent's configuration and triggers a re-embedding of its data.
    Requires: 'agent:update' permission.
    """
    if agent_name != config.name:
        raise HTTPException(status_code=400, detail="Agent name in path does not match agent name in body.")
    try:
        load_agent_config(agent_name)  # Ensure the agent exists

        save_agent_config(config)
        bg.add_task(embed_agent_data, config)
        logger.info(f"User '{current_user.username}' updated agent '{agent_name}'.")
        return {"status": "Agent config updated; re-embedding started.", "agent": config.name}
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail=f"Agent '{agent_name}' not found.")
    except Exception as e:
        logger.error(f"API: Error updating agent '{agent_name}': {e}", exc_info=True)
        raise HTTPException(status_code=500,
                            detail=f"An unexpected server error occurred during agent update for '{agent_name}'. Please check server logs.")


@router.delete("/{agent_name}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_agent_by_name(
        agent_name: str,
        # Only users with 'agent:delete' permission can delete agents
        current_user: User = Depends(PermissionChecker(["agent:delete"]))
):
    """
    Deletes an agent, its configuration, and all associated data.
    Requires: 'agent:delete' permission.
    """
    try:
        load_agent_config(agent_name)  # Ensure the agent exists

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
    is_deployed = vs_dir.exists()
    # you’ll need your own logic to detect ingesting / online:
    is_ingesting = False
    is_online = True

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

