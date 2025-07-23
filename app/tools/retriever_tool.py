import os
import logging
import json
from typing import List

from langchain.retrievers import EnsembleRetriever, ContextualCompressionRetriever
from langchain.retrievers.document_compressors import CrossEncoderReranker
from langchain_community.retrievers import BM25Retriever
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.tools import Tool
from langchain_community.cross_encoders import HuggingFaceCrossEncoder

from app.core.embed_config import get_embedding_model
from app.core.config import get_path_settings
from app.schemas.agent import AgentConfig

logger = logging.getLogger(__name__)

_APP_PATHS = get_path_settings()
_VECTORSTORE_DIR = _APP_PATHS["VECTORSTORE_DIR"]


def get_retriever_tool(agent_config: AgentConfig) -> Tool:
    agent_name = agent_config.name
    vs_config = agent_config.vector_store
    strategy = vs_config.retrieval_strategy

    try:
        vectorstore_path = os.path.join(_VECTORSTORE_DIR, agent_name)
        embeddings = get_embedding_model(agent_config.embedding_model)

        if vs_config.type != 'faiss':
            raise ValueError(f"Unsupported vector store type: {vs_config.type}")
        if not os.path.exists(vectorstore_path):
            raise FileNotFoundError(f"FAISS store not found for '{agent_name}'. Please deploy the agent.")

        vectorstore = FAISS.load_local(vectorstore_path, embeddings, allow_dangerous_deserialization=True)

        docs_path = os.path.join(_VECTORSTORE_DIR, agent_name, "bm25_documents.jsonl")
        if not os.path.exists(docs_path):
            raise FileNotFoundError(f"BM25 source file not found at {docs_path}. Please deploy the agent again.")

        bm25_docs = [Document(page_content=json.loads(line)["page_content"]) for line in
                     open(docs_path, 'r', encoding='utf-8')]

        if not bm25_docs:
            raise ValueError("No documents found for BM25 retriever.")

        bm25 = BM25Retriever.from_documents(bm25_docs, k=vs_config.bm25_k)
        semantic = vectorstore.as_retriever(search_kwargs={"k": vs_config.semantic_k})
        ensemble = EnsembleRetriever(retrievers=[bm25, semantic], weights=[0.5, 0.5])

        base_retriever = ensemble
        if strategy == 'enhanced':
            cross_encoder = HuggingFaceCrossEncoder(model_name="cross-encoder/ms-marco-MiniLM-L-6-v2")
            compressor = CrossEncoderReranker(model=cross_encoder, top_n=vs_config.rerank_top_n)
            base_retriever = ContextualCompressionRetriever(base_compressor=compressor, base_retriever=ensemble)

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