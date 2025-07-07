from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any, Literal

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


class AgentConfig(BaseModel):
    name: str
    display_name: Optional[str] = None
    description: Optional[str] = None
    persona_prompt: str
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


