from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any, Literal

class ChunkingConfig(BaseModel):
    """Configuration for the document chunking strategy."""
    mode: Literal['default', 'semantic'] = Field(
        'default',
        description="The chunking mode. 'default' for fast, recursive splitting; 'semantic' for higher quality, context-aware splitting."
    )
    chunk_size: int = Field(
        1000,
        description="The target size for each text chunk (in characters or tokens)."
    )
    chunk_overlap: int = Field(
        100,
        description="The number of characters or tokens to overlap between chunks."
    )

    breakpoint_percentile_threshold: int = Field(
        95,
        description="Percentile threshold for LlamaIndex semantic split."
    )

class DataSource(BaseModel):
    type: Literal['local', 'url', 'code_repository','db','gdoc','web_crawler','api','notebook']
    path: Optional[str] = None
    url: Optional[str] = None
    db_connection: Optional[str] = None
    folder_id: Optional[str] = None
    document_ids: Optional[List[str]] = None

    file_types: Optional[List[str]] = None

    max_depth: Optional[int] = 2

    headers: Optional[Dict[str, str]] = None
    params: Optional[Dict[str, Any]] = None
    method: Optional[Literal['GET', 'POST']] = 'GET'
    payload: Optional[Dict[str, Any]] = None
    json_pointer: Optional[str] = None

class ModelParams(BaseModel):

    temperature: Optional[float] = Field(None, description="Controls randomness. Lower is more deterministic.")
    max_tokens: Optional[int] = Field(None, description="The maximum number of tokens to generate.")
    top_p: Optional[float] = Field(None, description="Nucleus sampling probability.")


class VectorStoreConfig(BaseModel):
    """Configuration for the vector database."""
    type: Literal['faiss', 'chroma', 'qdrant', 'pinecone', 'mongodb_atlas'] = 'faiss'

    bm25_k: int = Field(5, description="k for BM25 retriever")
    semantic_k: int = Field(5, description="k for semantic retriever")
    rerank_top_n: int = Field(5, description="how many to keep after cross-encoder rerank")

    hit_rate_k_value: int = Field(
        5,
        description="K for Hit Rate@K (how many top docs to consider in the metric)"
    )

    retrieval_strategy: Literal['hybrid', 'enhanced'] = Field(
        'hybrid',
        description="The retrieval strategy to use. 'hybrid' is fast, 'enhanced' is more accurate but uses more compute."
    )

    # Qdrant specific
    qdrant_host: Optional[str] = Field(None, description="Hostname for a remote Qdrant server.")
    qdrant_port: Optional[int] = Field(6333, description="Port for a remote Qdrant server.")

    # Pinecone specific
    pinecone_index_name: Optional[str] = Field(None, description="The name of the Pinecone index.")

    # MongoDB Atlas specific
    mongodb_db_name: Optional[str] = Field(None, description="The name of the MongoDB database.")
    mongodb_collection_name: Optional[str] = Field(None, description="The name of the MongoDB collection.")
    mongodb_index_name: Optional[str] = Field("vector_search_index",
                                              description="The name of the Atlas Vector Search index.")

    # id_key field has been removed as per simplification. Document.id will be used.


class AgentConfig(BaseModel):
    name: str
    display_name: Optional[str] = None
    description: Optional[str] = None
    persona_prompt: str = "You are a helpful assistant."

    execution_prompt: Optional[str] = Field(
        None,
        description="A custom prompt template string. Use {user_query} and {retrieved_context} as placeholders."
    )

    chunking: ChunkingConfig = Field(
        default_factory=ChunkingConfig,
        description="The configuration for how documents are chunked before embedding."
    )

    sources: List[DataSource]
    tools: Optional[List[Literal['retriever', 'sql_toolkit', 'arxiv', 'extractor']]] = ['retriever']

    extraction_schema: Optional[Dict[str, str]] = Field(
        None,
        description="A dictionary defining the schema for the extractor tool. Key is the attribute name, value is the description."
    )


    # Users can now specify models in their agent.yaml file.
    # Default models if they are omitted.
    embedding_model: str = "text-embedding-3-small"
    llm_model: str = "gpt-4o-mini"
    model_params: Optional[ModelParams] = Field(None, description="Advanced configuration parameters for the LLM.")
    vector_store: VectorStoreConfig = Field(default_factory=VectorStoreConfig, description="The vector database to use for the agent.")

    # For PyTest
    extraction_examples: Optional[List[tuple[str, Any]]] = Field(
        None,
        description="A list of few-shot examples (text, expected_output) for the extractor tool."
    )
    evaluation_llm_model: Optional[str] = Field(
        None,
        description="Optional LLM to use specifically for evaluation tasks (e.g., test set generation, LLM-as-a-judge)."
    )
    reproducible_ids: Optional[bool] = Field(
        False,
        description="If true, generates IDs based on content hash for reproducibility; otherwise, uses random UUIDs."
    )

    llm_retries: Optional[int] = Field(
        0, # Default to no retries unless specified
        description="Number of times to retry LLM calls on transient errors. Defaults to 0 (no retries)."
    )
    llm_timeout: Optional[int] = Field(
        60, # Default to 60 seconds
        description="Timeout in seconds for LLM calls. Defaults to 60 seconds."
    )