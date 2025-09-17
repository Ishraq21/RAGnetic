import re
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Union
from pydantic import BaseModel, Field, field_validator, model_validator
from urllib.parse import urlparse

class ChunkingConfig(BaseModel):
    """Configuration for the document chunking strategy."""
    mode: Literal['default', 'semantic', 'none'] = Field(
        'default',
        description="The chunking mode. 'default' for recursive splitting, 'semantic' for context-aware splitting, or 'none' if the loader handles chunking."
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
    type: Literal[
        'local', 'url', 'code_repository', 'db', 'gdoc', 'web_crawler', 'api', 'notebook', 'parquet', 'csv', 'pdf', 'txt', 'docx']
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

    @field_validator('url')
    @classmethod
    def validate_url_scheme_and_form(cls, v: Optional[str]) -> Optional[str]:
        if v:
            try:
                parsed_url = urlparse(v)
                # Only allow http and https schemes for web-based sources
                if parsed_url.scheme not in ['http', 'https']:
                    raise ValueError(f"URL must use 'http' or 'https' scheme, got '{parsed_url.scheme}'.")
                # Basic check for a network location (domain)
                if not parsed_url.netloc:
                    raise ValueError('Invalid URL format: missing domain/host.')
            except ValueError as e: # Catch ValueError from urlparse or our custom checks
                raise ValueError(f"Invalid URL: {v} - {e}")
            except Exception as e: # Catch any other unexpected errors during parsing/validation
                raise ValueError(f"An unexpected error occurred during URL validation for {v}: {e}")
        return v

    @field_validator('path')
    @classmethod
    def validate_local_path_for_traversal(cls, v: Optional[str]) -> Optional[str]:
        if v:
            # Basic check to prevent explicit '..' in the path string at schema level.
            # This doesn't replace the robust check in the loader, but adds a layer.
            # Allow '~' for home directory expansion, if you intend to support it.
            if ".." in v and not v.startswith("~"):
                raise ValueError(f"Relative path traversal ('..') is not allowed in path: {v}. "
                                 "Paths must be within designated data directories.")
            # Normalization might be done here but security enforcement happens in loader.
        return v

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


class PIIConfig(BaseModel):
    types: List[Literal['email', 'phone', 'ssn', 'credit_card', 'name']] = Field(
        default_factory=lambda: ['email', 'phone'],
        description="List of PII types to redact. Supported: 'email', 'phone', 'ssn', 'credit_card', 'name'."
    )
    redaction_char: str = Field(
        "*",
        description="Character to replace redacted PII with."
    )
    redaction_placeholder: Optional[str] = Field(
        None,
        description="Optional string to replace redacted PII (e.g., '[REDACTED_EMAIL]'). If None, uses redaction_char."
    )

class KeywordFilterConfig(BaseModel):
    keywords: List[str] = Field(
        ...,
        description="List of keywords or phrases to filter/redact."
    )
    action: Literal['redact', 'block_chunk', 'block_document'] = Field(
        'redact',
        description="Action to take: 'redact' keyword, 'block_chunk' containing keyword, or 'block_document' containing keyword."
    )
    redaction_char: str = Field(
        "*",
        description="Character to replace redacted keywords with (if action is 'redact')."
    )
    redaction_placeholder: Optional[str] = Field(
        None,
        description="Optional string to replace redacted keywords (e.g., '[REDACTED_TERM]'). If None, uses redaction_char."
    )

class DataPolicy(BaseModel):
    type: Literal['pii_redaction', 'keyword_filter']
    pii_config: Optional[PIIConfig] = Field(
        None,
        description="Configuration for PII redaction policy."
    )
    keyword_filter_config: Optional[KeywordFilterConfig] = Field(
        None,
        description="Configuration for keyword filtering policy."
    )

    @field_validator('pii_config', 'keyword_filter_config')
    @classmethod
    def check_policy_config_is_present(cls, v, info):
        if info.data['type'] == 'pii_redaction' and v is None:
            raise ValueError('pii_config must be provided for pii_redaction policy type.')
        if info.data['type'] == 'keyword_filter' and v is None:
            raise ValueError('keyword_filter_config must be provided for keyword_filter policy type.')
        return v


class ScalingConfig(BaseModel):
    """Configuration for scaling data ingestion and processing."""
    parallel_ingestion: bool = Field(
        False,
        description="If true, enables parallel processing of multiple items within a single source (e.g., files in a directory, API records)."
    )
    num_ingestion_workers: Optional[int] = Field(
        None,
        description="Number of parallel workers/processes to use for ingestion. If None, defaults to CPU count."
    )

class SearchEngineToolInput(BaseModel):
    """Input schema for the AI Search Engine Tool."""
    query: str = Field(
        ...,
        description="The search query to send to the web search engine."
    )
    num_results: int = Field(
        5, # Default to 5 results
        description="The maximum number of search results to retrieve."
    )
    # Optional fields for more advanced filtering (can be expanded later)
    time_period: Optional[Literal['past_day', 'past_week', 'past_month', 'past_year']] = Field(
        None,
        description="Filter search results by a specific time period."
    )
    region: Optional[str] = Field(
        None,
        description="Filter search results by a geographical region (e.g., 'US', 'GB', 'DE')."
    )
class BenchmarkConfig(BaseModel):
    """Settings that affect only benchmarking/evaluation."""
    context_window_tokens: int = Field(8000, description="Total model context window used for budgeting.")
    context_budget_ratio: float = Field(0.70, ge=0.10, le=0.95, description="Fraction of the window allocated to retrieved context.")
    answer_reserve_tokens: int = Field(1024, description="Tokens reserved for the model's answer during benchmarking.")
    enable_doc_truncation: bool = Field(True, description="If true, truncate retrieved docs to fit within the budget.")
    max_context_docs: int = Field(20, description="Hard cap on how many retrieved docs are passed into the prompt.")
    request_delay_seconds: float = Field(2.0, ge=0.0, le=60.0, description="Delay between benchmark requests to avoid rate limiting (seconds).")


class GPUConfig(BaseModel):
    """Configuration for GPU requirements and provisioning following RunPod architecture."""
    
    # Platform Selection
    platform: Literal["native", "runpod"] = Field(
        "native",
        description="Platform to use: 'native' for local/cloud platform, 'runpod' for RunPod GPU instances."
    )
    
    # RunPod-specific configuration (only used when platform="runpod")
    enabled: bool = Field(
        False,
        description="Whether GPU acceleration is enabled for this agent (RunPod only)."
    )
    
    gpu_type: Optional[str] = Field(
        None,
        description="Required GPU type following RunPod naming (e.g., 'NVIDIA RTX 4090', 'NVIDIA A100 80GB PCIe')."
    )
    
    provider: Optional[str] = Field(
        None,
        description="RunPod provider tier: 'RunPod' (secure), 'RunPod Community' (community), 'RunPod Spot' (spot pricing)."
    )
    
    max_hours: Optional[float] = Field(
        None,
        ge=0.1,
        le=168.0,  # Max 1 week
        description="Maximum hours to run the GPU instance (0.1 to 168 hours)."
    )
    
    auto_provision: bool = Field(
        True,
        description="Whether to automatically provision GPU when agent is deployed."
    )
    
    # GPU Requirements
    min_memory_gb: Optional[int] = Field(
        None,
        ge=4,
        le=320,
        description="Minimum GPU memory requirement in GB (4-320 GB)."
    )
    
    min_cuda_cores: Optional[int] = Field(
        None,
        ge=1000,
        le=20000,
        description="Minimum CUDA cores requirement (1000-20000)."
    )
    
    # RunPod Instance Configuration
    container_disk_gb: Optional[int] = Field(
        50,
        ge=10,
        le=500,
        description="Container disk size in GB (10-500 GB)."
    )
    
    volume_gb: Optional[int] = Field(
        0,
        ge=0,
        le=1000,
        description="Additional volume size in GB (0-1000 GB)."
    )
    
    volume_mount_path: Optional[str] = Field(
        "/workspace",
        description="Volume mount path for persistent storage."
    )
    
    # Network and Ports
    ports: Optional[str] = Field(
        "8000/http",
        description="Ports to expose (e.g., '8000/http,8080/http')."
    )
    
    # Environment Variables
    environment_vars: Optional[Dict[str, str]] = Field(
        default_factory=dict,
        description="Environment variables for the GPU instance."
    )
    
    # Docker Configuration
    docker_args: Optional[str] = Field(
        "",
        description="Additional Docker arguments."
    )
    
    # RunPod Features
    start_jupyter: bool = Field(
        False,
        description="Whether to start Jupyter Lab on the instance."
    )
    
    start_ssh: bool = Field(
        True,
        description="Whether to enable SSH access to the instance."
    )
    
    # Purpose and Use Case
    purpose: Literal["training", "inference", "both", "development"] = Field(
        "inference",
        description="Primary purpose for GPU usage."
    )
    
    # Cost and Billing
    max_cost_per_hour: Optional[float] = Field(
        None,
        ge=0.1,
        le=10.0,
        description="Maximum cost per hour willing to pay (0.1-10.0 USD)."
    )
    
    # Scheduling
    schedule_start: Optional[str] = Field(
        None,
        description="Scheduled start time (ISO format) for the instance."
    )
    
    schedule_stop: Optional[str] = Field(
        None,
        description="Scheduled stop time (ISO format) for the instance."
    )
    
    @model_validator(mode='after')
    def validate_gpu_config(self):
        """Validate GPU configuration consistency."""
        if self.platform == "runpod" and self.enabled:
            if not self.gpu_type:
                raise ValueError("gpu_type is required when RunPod GPU is enabled")
            if not self.provider:
                raise ValueError("provider is required when RunPod GPU is enabled")
            if not self.max_hours:
                raise ValueError("max_hours is required when RunPod GPU is enabled")
            
            # Validate provider options
            valid_providers = ["RunPod", "RunPod Community", "RunPod Spot"]
            if self.provider not in valid_providers:
                raise ValueError(f"provider must be one of: {', '.join(valid_providers)}")
            
            # Validate cost limits
            if self.max_cost_per_hour and self.max_cost_per_hour < 0.1:
                raise ValueError("max_cost_per_hour must be at least 0.1 USD")
        
        return self


class AgentConfig(BaseModel):
    name: str
    display_name: Optional[str] = None
    description: Optional[str] = None
    persona_prompt: str = "You are a helpful assistant."
    
    # Database timestamps
    created_at: Optional[datetime] = Field(None, description="When the agent was created.")
    updated_at: Optional[datetime] = Field(None, description="When the agent was last updated.")

    execution_prompt: Optional[str] = Field(
        None,
        description="A custom prompt template string. Use {user_query} and {retrieved_context} as placeholders."
    )

    chunking: ChunkingConfig = Field(
        default_factory=ChunkingConfig,
        description="The configuration for how documents are chunked before embedding."
    )

    sources: Optional[List[DataSource]] = Field(
        default_factory=list,
        description="A list of data sources for the agent to retrieve information from. Optional for generic chatbots."
    )
    tools: Optional[List[str]] = Field(
        default_factory=list,
        description="A list of tools the agent can use. Defaults to an empty list."
    )

    # Users can now specify models in their agent.yaml file.
    # Default models if they are omitted.
    embedding_model: str = "text-embedding-3-small"
    llm_model: str = "gpt-4o-mini"
    model_params: Optional[ModelParams] = Field(None, description="Advanced configuration parameters for the LLM.")
    vector_store: VectorStoreConfig = Field(default_factory=VectorStoreConfig, description="The vector database to use for the agent.")

    # For PyTest
    evaluation_llm_model: Optional[str] = Field(
        None,
        description="Optional LLM to use specifically for evaluation tasks (e.g., test set generation, LLM-as-a-judge)."
    )
    reproducible_ids: Optional[bool] = Field(
        True,
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

    data_policies: Optional[List[DataPolicy]] = Field(
        default_factory=list,
        description="A list of data policies for redaction or filtering sensitive information during ingestion."
    )


    scaling: ScalingConfig = Field(
        default_factory=ScalingConfig,
        description="Configuration for parallel data ingestion and processing within a single source."
    )

    fine_tuned_model_id: Optional[str] = Field(
        None,
        description="The unique ID of a specific fine-tuned model or LoRA adapter to use with this agent's base LLM."
    )

    benchmark: BenchmarkConfig = Field(
        default_factory=BenchmarkConfig,
        description="Benchmark/eval-time overrides (context budget, truncation, etc.)."
    )

    gpu: GPUConfig = Field(
        default_factory=GPUConfig,
        description="GPU configuration for acceleration and provisioning."
    )

# --- Agent Inspection Response Models ---

class DocumentMetadata(BaseModel):
    page_content: str
    metadata: Dict[str, Any]
    score: Optional[float] = None


class ConnectionCheck(BaseModel):
    status: str
    message: Optional[str] = None
    detail: Optional[str] = None


class ValidationReport(BaseModel):
    is_valid: bool
    errors: List[str]
    warnings: Optional[List[str]] = Field(default_factory=list)


class AgentDeploymentStatus(BaseModel):
    """Agent deployment status information."""
    name: str
    status: str
    created_at: datetime
    total_cost: float
    gpu_instance_id: Optional[str] = None
    display_name: Optional[str] = None
    description: Optional[str] = None
    model_name: str
    embedding_model: Optional[str] = None
    last_run: Optional[datetime] = None
    deployment_type: Optional[str] = None
    project_id: Optional[int] = None
    tags: Optional[List[str]] = None

class AgentInspectionResponse(BaseModel):
    name: str
    is_deployed: bool
    is_ingesting: bool
    is_online: bool
    vector_store_status: ConnectionCheck
    sources_status: Dict[str, ConnectionCheck]
    document_metadata: Optional[List[DocumentMetadata]] = None
    validation_report: ValidationReport


