import os
import logging
import json
from typing import List

from langchain.retrievers import EnsembleRetriever, ContextualCompressionRetriever
from langchain.retrievers.document_compressors import CrossEncoderReranker
from langchain_community.retrievers import BM25Retriever
# Ensure all these specific imports are here:
from langchain_community.vectorstores import FAISS
from langchain_chroma import Chroma
from langchain_qdrant import Qdrant
from langchain_pinecone import Pinecone as PineconeLangChain
from langchain_mongodb import MongoDBAtlasVectorSearch
from langchain_core.documents import Document
from langchain_core.tools import Tool
from langchain_community.cross_encoders import HuggingFaceCrossEncoder

from pinecone import Pinecone as PineconeClient # Import the client directly

from app.core.embed_config import get_embedding_model
from app.core.config import get_path_settings, get_api_key
from app.schemas.agent import AgentConfig

logger = logging.getLogger(__name__)

_APP_PATHS = get_path_settings()
_VECTORSTORE_DIR = _APP_PATHS["VECTORSTORE_DIR"]


def get_retriever_tool(agent_config: AgentConfig) -> Tool:
    agent_name = agent_config.name
    vs_config = agent_config.vector_store
    strategy = vs_config.retrieval_strategy

    try:
        embeddings = get_embedding_model(agent_config.embedding_model)
        vectorstore = None

        # --- Dynamic Vector Store Loading ---
        if vs_config.type == 'faiss':
            vectorstore_path = os.path.join(_VECTORSTORE_DIR, agent_name)
            if not os.path.exists(vectorstore_path):
                raise FileNotFoundError(f"FAISS store not found for '{agent_name}'. Please deploy the agent.")
            vectorstore = FAISS.load_local(vectorstore_path, embeddings, allow_dangerous_deserialization=True)
            logger.info(f"Loaded FAISS vector store for agent '{agent_name}'.")

        elif vs_config.type == 'chroma':
            # Chroma persists to a directory
            vectorstore_path = os.path.join(_VECTORSTORE_DIR, agent_name, "chroma_db") # Use a specific sub-dir for Chroma
            if not os.path.exists(vectorstore_path):
                raise FileNotFoundError(f"Chroma store not found for '{agent_name}'. Please deploy the agent.")
            # Chroma client auto-loads from persist_directory if it exists
            vectorstore = Chroma(persist_directory=vectorstore_path, embedding_function=embeddings)
            logger.info(f"Loaded Chroma vector store for agent '{agent_name}'.")

        elif vs_config.type == 'qdrant':
            if not vs_config.qdrant_host:
                raise ValueError("Qdrant host not specified in agent config.")
            # Qdrant client connects to existing collection
            vectorstore = Qdrant(
                client=None, # Client can be None, Qdrant will create one
                collection_name=agent_name, # Use agent_name as collection name
                embeddings=embeddings,
                host=vs_config.qdrant_host,
                port=vs_config.qdrant_port,
                prefer_grpc=True # Often faster
            )
            logger.info(f"Loaded Qdrant vector store for agent '{agent_name}' on {vs_config.qdrant_host}.")

        elif vs_config.type == 'pinecone':
            if not vs_config.pinecone_index_name:
                raise ValueError("Pinecone index name not specified in agent config.")
            pinecone_api_key = get_api_key("pinecone") # Retrieve API key
            if not pinecone_api_key:
                raise ValueError("Pinecone API key not found. Please configure it.")

            # Initialize Pinecone client
            PineconeClient(api_key=pinecone_api_key)

            # Connect to existing Pinecone index
            vectorstore = PineconeLangChain.from_existing_index(
                index_name=vs_config.pinecone_index_name,
                embedding=embeddings
            )
            logger.info(f"Loaded Pinecone vector store for agent '{agent_name}' index '{vs_config.pinecone_index_name}'.")

        elif vs_config.type == 'mongodb_atlas':
            if not vs_config.mongodb_db_name or not vs_config.mongodb_collection_name:
                raise ValueError("MongoDB Atlas database or collection name not specified.")
            mongodb_conn_string = get_api_key("mongodb") # Retrieve connection string
            if not mongodb_conn_string:
                raise ValueError("MongoDB connection string not found. Please configure it.")

            vectorstore = MongoDBAtlasVectorSearch.from_connection_string(
                mongodb_conn_string,
                vs_config.mongodb_db_name,
                vs_config.mongodb_collection_name,
                embeddings,
                index_name=vs_config.mongodb_index_name
            )
            logger.info(f"Loaded MongoDB Atlas vector store for agent '{agent_name}' collection '{vs_config.mongodb_collection_name}'.")

        else:
            raise ValueError(f"Unsupported vector store type: {vs_config.type}")

        if vectorstore is None:
            raise ValueError(f"Failed to initialize vector store of type {vs_config.type}.")

        # Load BM25 documents (still necessary for hybrid)
        docs_path = os.path.join(_VECTORSTORE_DIR, agent_name, "bm25_documents.jsonl")
        if not os.path.exists(docs_path):
            # For non-FAISS cases, bm25_documents.jsonl might not be auto-generated or needed if only semantic search is desired.
            # However, for EnsembleRetriever, BM25 docs are required.
            logger.warning(f"BM25 source file not found at {docs_path}. Hybrid retrieval might be impacted.")
            bm25 = None # Set to None if file not found to handle gracefully
        else:
            bm25_docs = [Document(page_content=json.loads(line)["page_content"]) for line in
                         open(docs_path, 'r', encoding='utf-8')]
            if not bm25_docs:
                logger.warning("No documents found for BM25 retriever in bm25_documents.jsonl.")
                bm25 = None
            else:
                bm25 = BM25Retriever.from_documents(bm25_docs, k=vs_config.bm25_k)

        # Ensure ensemble retriever only includes BM25 if it was successfully initialized
        retrievers = [semantic for semantic in [vectorstore.as_retriever(search_kwargs={"k": vs_config.semantic_k})] if semantic]
        if bm25: # Only add BM25 if it's not None
            retrievers.insert(0, bm25) # Insert at beginning to match original ensemble order (BM25, Semantic)

        # Adjust weights if only one retriever is present
        if len(retrievers) == 1:
            base_retriever = retrievers[0]
            logger.info(f"Using single retriever ('{retrievers[0].__class__.__name__}') as BM25 docs not found or not used.")
        elif len(retrievers) == 2:
            ensemble = EnsembleRetriever(retrievers=retrievers, weights=[0.5, 0.5])
            base_retriever = ensemble
            logger.info("Using EnsembleRetriever (BM25 + Semantic).")
        else:
            raise ValueError("No valid retrievers could be initialized.")


        if strategy == 'enhanced':
            cross_encoder = HuggingFaceCrossEncoder(model_name="cross-encoder/ms-marco-MiniLM-L-6-v2")
            compressor = ContextualCompressionRetriever(base_compressor=CrossEncoderReranker(model=cross_encoder, top_n=vs_config.rerank_top_n), base_retriever=base_retriever) # Pass base_retriever to compressor
            base_retriever = compressor
            logger.info("Enhanced retrieval strategy enabled with Cross-Encoder Reranker.")

        def _run(query: str) -> List[Document]:
            """The actual function that executes when the tool is called."""
            logger.info(f"Retriever tool running with query: '{query}'")
            return base_retriever.invoke(query)

        return Tool.from_function(
            name="retriever",
            func=_run,
            description=f"Searches the knowledge base of '{agent_name}' for information. Input must be a single question or topic."
        )

    except Exception as err:
        logger.error(f"Failed to init retriever for '{agent_name}': {err}", exc_info=True)

        def error_func(query: str) -> str:
            return f"Error: Could not load retriever for '{agent_name}'. Details: {err}"

        return Tool.from_function(name="retriever_error", func=error_func,
                                  description="Reports an error during retriever initialization.")