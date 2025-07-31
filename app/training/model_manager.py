import logging
from pathlib import Path
from typing import Optional, Any
import shutil # For deleting directories


from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel, AutoPeftModelForCausalLM
import torch

logger = logging.getLogger(__name__)

class FineTunedModelManager:
    """
    Manages saving, loading, and deleting fine-tuned LLM models/LoRA adapters
    from the filesystem.
    """
    def __init__(self, models_base_path: Path):
        self.models_base_path = models_base_path
        self.models_base_path.mkdir(parents=True, exist_ok=True) # Ensure the base directory exists
        logger.info(f"FineTunedModelManager initialized. Models will be stored in: {self.models_base_path}")

    def save_adapter(self, adapter_id: str, adapter: Any, tokenizer: Any = None) -> str:
        """
        Saves the fine-tuned LoRA adapter weights (or a full fine-tuned model)
        to a specified path derived from its unique 'adapter_id'.
        Returns the absolute path where the model was saved.
        """
        save_path = self.models_base_path / adapter_id
        save_path.mkdir(parents=True, exist_ok=True)  # Create the specific directory for this adapter

        try:
            # Assumes adapter is a PeftModel or a full Hugging Face model
            adapter.save_pretrained(str(save_path))
            if tokenizer:
                tokenizer.save_pretrained(str(save_path))
            logger.info(f"LoRA adapter/model '{adapter_id}' saved to {save_path}")
            return str(save_path)  # Return the path for database storage
        except Exception as e:
            logger.error(f"Failed to save adapter '{adapter_id}' to {save_path}: {e}", exc_info=True)
            raise

    def load_adapter(self, adapter_path: str, base_model_name: str) -> Optional[Any]:
        """
        Loads a fine-tuned model or LoRA adapter from a given path.
        If it's a LoRA adapter, it typically loads the base model first and then applies the adapter weights.
        Returns the loaded Hugging Face model instance (e.g., PeftModel or AutoModelForCausalLM).
        """
        model_path = Path(adapter_path)
        if not model_path.exists():
            logger.warning(f"Model path not found for adapter '{model_path}'. Cannot load.")
            return None

        try:
            # Determine device: Prioritize MPS for Apple Silicon
            device = "cpu"
            if torch.backends.mps.is_available():
                device = "mps"
                logger.info("Loading model on Apple Metal Performance Shaders (MPS).")
            elif torch.cuda.is_available():  # Keep CUDA check for broader compatibility
                device = "cuda"
                logger.info(f"Loading model on NVIDIA GPU (CUDA): {torch.cuda.get_device_name(0)}")
            else:
                logger.warning("No GPU (CUDA or MPS) found for loading. Loading on CPU.")

            # Define common loading arguments based on the determined device and dtype
            # Use bfloat16 for MPS if built and available, for CUDA if available, else float32
            load_dtype = torch.float32
            if device == "mps" and torch.backends.mps.is_built():  # Check if MPS bfloat16 is truly available
                # As of PyTorch 2.0+, MPS supports bfloat16 for many ops
                load_dtype = torch.bfloat16
            elif device == "cuda" and hasattr(torch, 'bfloat16'):  # Check if CUDA bfloat16 is available
                load_dtype = torch.bfloat16

            base_model_load_kwargs = {
                "device_map": device,
                "torch_dtype": load_dtype,
            }
            logger.info(
                f"Loading base model '{base_model_name}' with device_map='{device}' and torch_dtype='{load_dtype}'.")

            base_model = AutoModelForCausalLM.from_pretrained(
                base_model_name,
                **base_model_load_kwargs
            )

            # Try loading as a PEFT model (adapter)
            try:
                # PeftModel.from_pretrained expects the base model to be on the correct device already.
                peft_model = PeftModel.from_pretrained(base_model, str(model_path))
                # For deployment, often merge and unload to get a single, runnable model
                merged_model = peft_model.merge_and_unload()
                # Move the merged model to the correct device explicitly if device_map="auto" didn't fully place it.
                # Usually, device_map handles this, but an explicit .to(device) can be a fallback.
                if str(merged_model.device) != device:
                    merged_model = merged_model.to(device)
                    logger.info(f"Explicitly moved merged model to device: {device}")

                logger.info(
                    f"Successfully loaded and merged LoRA adapter from {model_path} for base '{base_model_name}'.")
                return merged_model
            except Exception as peft_e:
                logger.warning(
                    f"Could not load as PEFT adapter from {model_path}: {peft_e}. Attempting to load as full model.")
                # Fallback: Try loading as a full fine-tuned model (if it was saved that way)
                full_model = AutoModelForCausalLM.from_pretrained(
                    str(model_path),
                    **base_model_load_kwargs
                )
                if str(full_model.device) != device:
                    full_model = full_model.to(device)
                    logger.info(f"Explicitly moved full model to device: {device}")

                logger.info(f"Successfully loaded full fine-tuned model from {model_path}.")
                return full_model

        except Exception as e:
            logger.error(f"Failed to load model from '{model_path}' with base '{base_model_name}': {e}", exc_info=True)
            return None

    def delete_model(self, adapter_id: str) -> bool:
        """
        Deletes a saved fine-tuned model/adapter directory from the filesystem.
        Returns True if successful, False otherwise.
        """
        model_path = self.models_base_path / adapter_id
        if model_path.exists() and model_path.is_dir():
            shutil.rmtree(model_path) # Recursively remove the directory
            logger.info(f"Fine-tuned model directory '{adapter_id}' deleted from {model_path}.")
            return True
        logger.warning(f"Fine-tuned model directory '{adapter_id}' not found at {model_path} for deletion.")
        return False