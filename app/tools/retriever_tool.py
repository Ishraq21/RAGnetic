import os
import logging
import json
from typing import List, Dict, Union

from langchain.retrievers import EnsembleRetriever, ContextualCompressionRetriever
from langchain.retrievers.document_compressors import CrossEncoderReranker
from langchain_community.retrievers import BM25Retriever
from langchain_community.vectorstores import FAISS, Chroma, Qdrant, Pinecone as PineconeVectorStore, \
    MongoDBAtlasVectorSearch
from langchain_core.documents import Document
from langchain_core.tools import Tool

from pinecone import Pinecone as PineconeClient
from langchain_community.cross_encoders import HuggingFaceCrossEncoder

from app.core.embed_config import get_embedding_model
from app.core.config import get_api_key, get_path_settings
from app.schemas.agent import AgentConfig

logger = logging.getLogger(__name__)

_APP_PATHS        = get_path_settings()
_VECTORSTORE_DIR  = _APP_PATHS["VECTORSTORE_DIR"]


def _load_bm25_documents(agent_name: str) -> List[Document]:
    docs_path = os.path.join(_VECTORSTORE_DIR, agent_name, "bm25_documents.jsonl")
    if not os.path.exists(docs_path):
        raise FileNotFoundError(f"BM25 source file not found at {docs_path}. Please deploy the agent again.")
    docs = []
    with open(docs_path, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line)
            docs.append(Document(id=data["id"], page_content=data["page_content"]))
    return docs


def tool_fn(retriever, query: Union[str, Dict[str, str]]) -> List[Document]:
    if isinstance(query, dict):
        query = query.get("input", "")
    return retriever.invoke(query)


def get_retriever_tool(agent_config: AgentConfig) -> Tool:
    agent_name = agent_config.name
    vs_config  = agent_config.vector_store
    strategy   = vs_config.retrieval_strategy

    try:
        vectorstore_path = os.path.join(_VECTORSTORE_DIR, agent_name)

        embeddings = get_embedding_model(agent_config.embedding_model)

        if vs_config.type == 'faiss':
            if not os.path.exists(vectorstore_path):
                raise FileNotFoundError(f"FAISS store not found for '{agent_name}' at {vectorstore_path}")
            vectorstore = FAISS.load_local(
                vectorstore_path, embeddings,
                allow_dangerous_deserialization=True
            )
        else:
            raise ValueError(f"Unsupported vector store type: {vs_config.type}")

        bm25_docs = _load_bm25_documents(agent_name)
        if not bm25_docs:
            raise ValueError("No documents found for BM25 retriever; source file is empty.")

        bm25 = BM25Retriever.from_documents(bm25_docs)
        bm25.k = vs_config.bm25_k

        semantic = vectorstore.as_retriever(search_kwargs={"k": vs_config.semantic_k})
        ensemble = EnsembleRetriever(retrievers=[bm25, semantic], weights=[0.5, 0.5])

        if strategy == 'enhanced':
            xformer = HuggingFaceCrossEncoder(model_name="cross-encoder/ms-marco-MiniLM-L-6-v2")
            reranker = CrossEncoderReranker(model=xformer, top_n=vs_config.rerank_top_n)
            retriever = ContextualCompressionRetriever(
                base_compressor=reranker,
                base_retriever=ensemble
            )
        else:
            retriever = ensemble

        return Tool(
            name="retriever",
            func=lambda q: tool_fn(retriever, q),
            description=f"Searches the knowledge base of '{agent_name}'."
        )

    except Exception as err:
        logger.error(f"Failed to init retriever for '{agent_name}': {err}", exc_info=True)
        error_msg = str(err)

        def error_func(_: str, msg=error_msg) -> str:
            return f"Error: Could not load retriever for '{agent_name}'. Details: {msg}"

        return Tool(
            name="retriever_error",
            func=error_func,
            description="Reports an error during retriever initialization."
        )