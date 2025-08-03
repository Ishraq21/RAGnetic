# app/schemas/fine_tuning.py
from pydantic import BaseModel, Field, model_validator
from typing import Optional, List, Dict, Any, Literal
from datetime import datetime
from enum import Enum

# Enum for fine-tuning job status, used in database and API responses
class FineTuningStatus(str, Enum):
    PENDING = "pending"    # Job submitted, awaiting execution
    RUNNING = "running"    # Training is actively in progress
    COMPLETED = "completed"  # Training finished successfully
    FAILED = "failed"      # Training encountered an error
    PAUSED = "paused"      # Training temporarily suspended (optional, for advanced control)

# Pydantic model for hyperparameters defined within the fine-tuning YAML configuration
class HyperparametersConfig(BaseModel):
    lora_rank: int = Field(8, description="LoRA rank (dimension) for PEFT fine-tuning. Higher rank offers more capacity.")
    learning_rate: float = Field(2e-4, description="Learning rate for the optimizer during training.")
    epochs: int = Field(3, description="Number of full passes over the training dataset.")
    batch_size: int = Field(4, description="Training batch size per device.")
    lora_alpha: Optional[int] = Field(None, description="LoRA alpha parameter (scaling factor for LoRA weights).")
    target_modules: Optional[List[str]] = Field(None, description="List of target modules (e.g., attention layers) for LoRA application.")
    lora_dropout: Optional[float] = Field(None, description="Dropout probability for LoRA layers to prevent overfitting.")
    gradient_accumulation_steps: Optional[int] = Field(1, description="Number of updates steps to accumulate before performing a backward/update pass.")
    logging_steps: Optional[int] = Field(10, description="How often to log training loss and metrics.")
    save_steps: Optional[int] = Field(500, description="How often to save a model checkpoint.")
    save_total_limit: Optional[int] = Field(1, description="Maximum number of checkpoints to keep.")
    cost_per_gpu_hour: Optional[float] = Field(0.5, description="Estimated cost in USD per GPU hour for metrics tracking.")


    mixed_precision_dtype: Optional[Literal['no', 'fp16', 'bf16']] = Field(
        'no', # Default to 'no' mixed precision for widest compatibility
        description="Type of mixed precision to use ('no', 'fp16', or 'bf16'). 'fp16' is for most NVIDIA GPUs, 'bf16' for newer NVIDIA and some Apple Silicon. 'no' for full float32."
    )


# Pydantic model representing the structure of a fine-tuning job YAML configuration file
class FineTuningJobConfig(BaseModel):
    job_name: str = Field(..., description="A user-defined, unique name for this specific fine-tuning job configuration.")
    base_model_name: str = Field(..., description="The name of the foundational open-source LLM to fine-tune (e.g., 'ollama/llama2', 'mistral', 'meta-llama/Llama-2-7b-hf').")
    dataset_path: str = Field(..., description="Absolute or relative path to the prepared training dataset file (e.g., 'data/prepared_qa.jsonl').")
    output_base_dir: str = Field("models/fine_tuned", description="Base directory where the fine-tuned model/adapter will be saved. A unique sub-directory will be created for each training run.")
    hyperparameters: HyperparametersConfig = Field(
        default_factory=HyperparametersConfig,  # Provides default values if not specified in YAML
        description="Detailed configuration for fine-tuning hyperparameters."
    )
    gpu_type_preference: Optional[str] = Field(None,
                                               description="Optional: Preferred GPU type for training (e.g., 'A100', 'V100'). Useful in multi-GPU environments.")
    notification_emails: Optional[List[str]] = Field(None,
                                                     description="Optional: List of email addresses to notify about job status changes (e.g., completion, failure).")

    device: Optional[Literal['cpu', 'cuda', 'mps']] = Field(
        None,
        description="Explicitly specify the device for training: 'cpu', 'cuda' (for NVIDIA GPUs), or 'mps' (for Apple Silicon GPUs). If None, the system will auto-detect."
    )
class FineTunedModel(BaseModel):
    id: int = Field(..., description="Primary key from the database for this fine-tuned model entry.")
    adapter_id: str = Field(..., description="Unique system-generated identifier (UUID) for this specific fine-tuned model or LoRA adapter instance.")
    job_name: str = Field(..., description="The user-defined name of the fine-tuning job that produced this model.")
    base_model_name: str = Field(..., description="The name of the original pre-trained LLM used.")
    adapter_path: str = Field(..., description="Absolute file path to the saved LoRA weights or full fine-tuned model on the filesystem.")
    training_dataset_id: Optional[str] = Field(None, description="Reference (e.g., path or ID) to the specific dataset used for this training run.")
    training_status: FineTuningStatus = Field(FineTuningStatus.PENDING, description="Current status of the fine-tuning job.")
    training_logs_path: Optional[str] = Field(None, description="Path to the training logs generated during this job.")
    hyperparameters: Optional[Dict[str, Any]] = Field(None, description="A dictionary of the actual hyperparameters used for this training run.")
    final_loss: Optional[float] = Field(None, description="The final training loss recorded at the end of the job.")
    validation_loss: Optional[float] = Field(None, description="The final validation loss recorded (if validation set was used).")
    gpu_hours_consumed: Optional[float] = Field(None, description="Estimated total GPU hours consumed by this training job.")
    estimated_training_cost_usd: Optional[float] = Field(None, description="Estimated monetary cost of this training job in USD.")
    created_by_user_id: int = Field(..., description="The ID of the user who initiated this fine-tuning job.")
    created_at: datetime = Field(..., description="Timestamp when this fine-tuning job record was created.")
    updated_at: datetime = Field(..., description="Timestamp when this fine-tuned model entry was last updated.")

    class Config:
        from_attributes = True