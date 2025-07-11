import os
import logging
from typing import List

from langchain_core.documents import Document as LangChainDocument
from llama_index.core.schema import Document as LlamaDocument
from llama_index.core.node_parser import SemanticSplitterNodeParser
from llama_index.embeddings.langchain import LangchainEmbedding

from app.core.config import get_api_key
from langchain_community.vectorstores import FAISS, Chroma
from langchain_qdrant import Qdrant
from langchain_pinecone import Pinecone as PineconeLangChain
from langchain_mongodb import MongoDBAtlasVectorSearch
from pinecone import Pinecone as PineconeClient

from app.schemas.agent import AgentConfig, DataSource
from app.core.embed_config import get_embedding_model
from app.pipelines.loaders import (
    directory_loader,
    url_loader,
    db_loader,
    api_loader,
    code_repository_loader,
    gdoc_loader,
    web_crawler_loader,
    notebook_loader,
    pdf_loader,
    docx_loader,
    csv_loader,
    text_loader,
    iac_loader
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_documents_from_source(source: DataSource) -> list[LangChainDocument]:
    source_type = source.type
    logger.info(f"Dispatching to loader for source type: '{source_type}'")

    if source_type == "local":
        path = source.path
        if not path or not os.path.exists(path):
            logger.warning(f"Local path not found: {path}. Skipping.")
            return []

        if os.path.isfile(path):
            if path.lower().endswith('.txt'):
                return text_loader.load(path)
            elif path.lower().endswith('.pdf'):
                return pdf_loader.load(path)
            elif path.lower().endswith('.docx'):
                return docx_loader.load(path)
            elif path.lower().endswith('.csv'):
                return csv_loader.load(path)
            elif path.lower().endswith('.ipynb'):
                return notebook_loader.load_notebook(path)
            elif path.lower().endswith(('.tf', '.tfvars')):
                return iac_loader.load(path)
            elif path.lower().endswith(('.yaml', '.yml')):
                return iac_loader.load(path)
            else:
                logger.warning(f"Unsupported local file type: {path}. Skipping.")
                return []
        elif os.path.isdir(path):
            return directory_loader.load(path)
        else:
            return []

    elif source_type == "code_repository":
        return code_repository_loader.load(source.path)
    elif source_type == "url":
        return url_loader.load(source.url)
    elif source_type == "db":
        return db_loader.load(source.db_connection)
    elif source_type == "gdoc":
        return gdoc_loader.load(folder_id=source.folder_id, document_ids=source.document_ids, file_types=source.file_types)
    elif source_type == "web_crawler":
        return web_crawler_loader.load(url=source.url, max_depth=source.max_depth)
    elif source_type == "api":
        return api_loader.load(url=source.url, method=source.method, headers=source.headers, params=source.params, payload=source.payload, json_pointer=source.json_pointer)
    else:
        logger.warning(f"Unknown or unsupported source type: '{source_type}'. Skipping.")
        return []


def embed_agent_data(config: AgentConfig):
    all_docs = []
    logger.info(f"Starting data embedding process for agent: '{config.name}'")
    for source in config.sources:
        loaded_docs = load_documents_from_source(source)
        if loaded_docs:
            all_docs.extend(loaded_docs)

    if not all_docs:
        raise ValueError("No valid documents found to embed from any source.")

    logger.info(f"Loaded a total of {len(all_docs)} documents from all sources.")

    try:
        langchain_embeddings = get_embedding_model(config.embedding_model)
        llama_embeddings = LangchainEmbedding(langchain_embeddings)

        semantic_splitter = SemanticSplitterNodeParser.from_defaults(
            embed_model=llama_embeddings,
            breakpoint_percentile_threshold=95,
        )

        llama_docs = [LlamaDocument.from_langchain_format(doc) for doc in all_docs]
        nodes = semantic_splitter.get_nodes_from_documents(llama_docs)
        logger.info(f"Semantically split documents into {len(nodes)} nodes.")

        # Manually create LangChain Documents from the LlamaIndex nodes.
        # Use node.get_content() to access the text of the node.
        chunks = [LangChainDocument(page_content=node.get_content(), metadata=node.metadata) for node in nodes]

        vs_config = config.vector_store
        db_type = vs_config.type
        vectorstore_path = f"vectorstore/{config.name}"

        logger.info(f"Creating vector store of type '{db_type}' for agent '{config.name}'.")

        if db_type == 'faiss':
            db = FAISS.from_documents(chunks, langchain_embeddings)
            db.save_local(vectorstore_path)
        elif db_type == 'chroma':
            Chroma.from_documents(chunks, langchain_embeddings, persist_directory=vectorstore_path)
        elif db_type == 'qdrant':
            Qdrant.from_documents(chunks, langchain_embeddings, host=vs_config.qdrant_host, port=vs_config.qdrant_port, path=None if vs_config.qdrant_host else vectorstore_path, collection_name=config.name)
        elif db_type == 'pinecone':
            pc_api_key = get_api_key("pinecone")
            PineconeClient(api_key=pc_api_key)
            PineconeLangChain.from_documents(chunks, langchain_embeddings, index_name=vs_config.pinecone_index_name)
        elif db_type == 'mongodb_atlas':
            conn_string = get_api_key("mongodb")
            MongoDBAtlasVectorSearch.from_documents(documents=chunks, embedding=langchain_embeddings, connection_string=conn_string, namespace=f"{vs_config.mongodb_db_name}.{vs_config.mongodb_collection_name}", index_name=vs_config.mongodb_index_name)
        else:
            raise ValueError(f"Unsupported vector store type: {db_type}")

        logger.info(f"Successfully created and populated vector store for agent '{config.name}'.")

    except Exception as e:
        logger.error(f"Failed to create vector store. Error: {e}", exc_info=True)
        raise