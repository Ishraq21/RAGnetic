from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any

class LambdaResourceSpec(BaseModel):
    cpu: str = "1"  # e.g., "1", "2.5"
    memory_gb: int = 4
    gpu_type: Optional[str] = None # e.g., "nvidia.com/gpu"
    gpu_count: Optional[int] = None
    disk_mb: int = 1000

class LambdaNetworkPolicy(BaseModel):
    allow_outbound: bool = False
    allowlist_domains: List[str] = Field(default_factory=list)

class LambdaInputFile(BaseModel):
    temp_doc_id: str
    file_name: str
    path_in_sandbox: Optional[str] = Field(
        None,
        description="The path where the file will be mounted in the sandbox."
    )

    # 🔹 Enriched metadata for robust staging
    original_name: Optional[str] = None
    user_id: Optional[int] = None
    thread_id: Optional[str] = None
    file_path: Optional[str] = Field(
        None,
        description="Absolute path on the host where the original file is stored (if still present)."
    )

class LambdaRequestPayload(BaseModel):
    mode: str = Field(..., description="Execution mode: 'code', 'function', or 'notebook'.")
    code: Optional[str] = None
    function_name: Optional[str] = None
    function_args: Optional[Dict[str, Any]] = Field(default_factory=dict)
    inputs: List[LambdaInputFile] = Field(default_factory=list)
    outputs: Optional[List[str]] = Field(
        default_factory=list,
        description="List of file paths in the sandbox to collect as outputs."
    )
    resource_spec: LambdaResourceSpec = Field(default_factory=LambdaResourceSpec)
    network_policy: LambdaNetworkPolicy = Field(default_factory=LambdaNetworkPolicy)
    ttl_seconds: int = Field(3600, description="Time-to-live for the run and artifacts.")
    user_id: Optional[int] = None
    thread_id: Optional[str] = None
    function_source: Optional[str] = None
