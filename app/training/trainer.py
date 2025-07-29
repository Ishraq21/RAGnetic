# app/training/trainer.py
import logging
from pathlib import Path
from typing import Dict, Any, Optional
from datetime import datetime
import uuid
import random
import torch
import os

from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from datasets import Dataset
from trl import SFTTrainer

from app.schemas.fine_tuning import FineTuningJobConfig, FineTuningStatus, FineTunedModel
from app.db import get_sync_db_engine # Function to get a synchronous SQLAlchemy engine
from app.db.models import fine_tuned_models_table # The new fine_tuned_models_table definition


from app.training.data_prep.jsonl_instruction_loader import JsonlInstructionLoader
from app.training.data_prep.conversational_jsonl_loader import ConversationalJsonlLoader
from app.schemas.data_prep import DatasetPreparationConfig

logger = logging.getLogger(__name__)

class LLMFineTuner:
    """
    Orchestrates the LLM fine-tuning process based on a given job configuration.
    Handles database updates for job status and metrics.
    """
    def __init__(self, db_engine: Any):
        self.db_engine = db_engine

    def _update_fine_tune_job_record(self, adapter_id: str,
                                     status: Optional[FineTuningStatus] = None,
                                     logs_path: Optional[str] = None,
                                     final_loss: Optional[float] = None,
                                     validation_loss: Optional[float] = None,
                                     gpu_hours_consumed: Optional[float] = None,
                                     estimated_training_cost_usd: Optional[float] = None):
        """
        Helper method to update the status and metrics of a fine-tuning job
        record in the 'fine_tuned_models_table' in the database.
        """
        with self.db_engine.connect() as connection:
            update_values = {"updated_at": datetime.utcnow()} # Always update timestamp
            if status: update_values["training_status"] = status.value
            if logs_path: update_values["training_logs_path"] = logs_path
            if final_loss is not None: update_values["final_loss"] = final_loss
            if validation_loss is not None: update_values["validation_loss"] = validation_loss
            if gpu_hours_consumed is not None: update_values["gpu_hours_consumed"] = gpu_hours_consumed
            if estimated_training_cost_usd is not None: update_values["estimated_training_cost_usd"] = estimated_training_cost_usd

            connection.execute(
                fine_tuned_models_table.update()
                .where(fine_tuned_models_table.c.adapter_id == adapter_id)
                .values(**update_values)
            )
            connection.commit() # Commit the transaction to save changes
        logger.info(f"Fine-tuning job '{adapter_id}' status updated to {status.value if status else 'no change'}.")

    def fine_tune_llm(
            self,
            job_config: FineTuningJobConfig,
            user_id: int,  # The ID of the user who initiated this training job
    ) -> FineTunedModel:
        """
        Orchestrates the LLM fine-tuning process using the parameters from a FineTuningJobConfig.
        This method uses Hugging Face training pipeline.
        Returns the initial database record of the fine-tuned model.
        """
        # Generate a unique adapter_id for this specific training run
        adapter_id = str(uuid.uuid4())
        job_name = job_config.job_name
        base_model_name = job_config.base_model_name
        dataset_path_str = job_config.dataset_path
        hyperparameters = job_config.hyperparameters.model_dump()

        output_dir = Path(job_config.output_base_dir) / job_name / adapter_id
        output_dir.mkdir(parents=True, exist_ok=True)
        adapter_path = str(output_dir)
        training_logs_path = str(output_dir / "training_logs.txt")

        logger.info(f"Initiating fine-tuning job '{job_name}' (ID: {adapter_id}) for base model '{base_model_name}'.")
        logger.info(f"Dataset: {dataset_path_str}, Model Output Path: {adapter_path}")
        logger.info(f"Configured Hyperparameters: {hyperparameters}")

        # 1. Record the job as PENDING in the database immediately.
        new_job_record_data = {
            "adapter_id": adapter_id,
            "job_name": job_name,
            "base_model_name": base_model_name,
            "adapter_path": adapter_path,
            "training_dataset_id": dataset_path_str,
            "training_status": FineTuningStatus.PENDING.value,
            "training_logs_path": training_logs_path,
            "hyperparameters": hyperparameters,
            "created_by_user_id": user_id,
            "created_at": datetime.utcnow(),
            "updated_at": datetime.utcnow(),
        }
        try:
            with self.db_engine.connect() as connection:
                result = connection.execute(
                    fine_tuned_models_table.insert().values(**new_job_record_data).returning(fine_tuned_models_table)
                )
                created_record_row = result.mappings().first()
                connection.commit()
            initial_db_record = FineTunedModel.model_validate(created_record_row)
            logger.info(f"Fine-tuning job '{job_name}' record created in DB with ID: {adapter_id}.")
        except Exception as e:
            logger.error(f"Failed to record new fine-tuning job in DB: {e}", exc_info=True)
            raise RuntimeError("Failed to create initial fine-tuning job record in database.")

        # 2. Update the job status to RUNNING.
        self._update_fine_tune_job_record(adapter_id, FineTuningStatus.RUNNING)

        try:
            # Determine device: Prioritize user-specified device, then auto-detect
            actual_device = "cpu"  # Default to CPU
            if job_config.device:  # User specified a device
                requested_device = job_config.device.lower()
                if requested_device == "cuda":
                    if torch.cuda.is_available():
                        actual_device = "cuda"
                        logger.info(f"Using NVIDIA GPU (CUDA) as requested: {torch.cuda.get_device_name(0)}")
                    else:
                        logger.warning("User requested 'cuda' but CUDA is not available. Falling back to CPU.")
                elif requested_device == "mps":
                    if torch.backends.mps.is_available():
                        actual_device = "mps"
                        logger.info("Using Apple Metal Performance Shaders (MPS) as requested.")
                    else:
                        logger.warning("User requested 'mps' but MPS is not available. Falling back to CPU.")
                elif requested_device == "cpu":
                    actual_device = "cpu"
                    logger.info("Using CPU as requested by user.")
                else:
                    logger.warning(f"Invalid device '{requested_device}' specified. Falling back to auto-detection.")

            # Auto-detection if no device was specified by user or if requested device was unavailable/invalid
            if actual_device == "cpu":  # If still on CPU after user request, try auto-detect
                if torch.backends.mps.is_available():
                    actual_device = "mps"
                    logger.info("Auto-detect: Using Apple Metal Performance Shaders (MPS) for GPU acceleration.")
                elif torch.cuda.is_available():
                    actual_device = "cuda"
                    logger.info(f"Auto-detect: Using NVIDIA GPU (CUDA): {torch.cuda.get_device_name(0)}")
                else:
                    logger.warning("Auto-detect: No GPU (CUDA or MPS) found. Training will proceed on CPU.")

            # --- Load and prepare the dataset ---
            logger.info(f"Loading training data from: {dataset_path_str}")

            with open(dataset_path_str, 'r', encoding='utf-8') as f:
                first_line = f.readline()
            if "instruction" in first_line and "output" in first_line:
                dataset_format = "jsonl-instruction"
            elif "messages" in first_line and "role" in first_line and "content" in first_line:
                dataset_format = "conversational-jsonl"
            else:
                raise ValueError(
                    "Could not infer dataset format. Ensure it's 'jsonl-instruction' or 'conversational-jsonl'.")

            if dataset_format == "jsonl-instruction":
                loader = JsonlInstructionLoader(dataset_path_str)
                training_data = loader.load()
            elif dataset_format == "conversational-jsonl":
                loader = ConversationalJsonlLoader(dataset_path_str)
                training_data = loader.load()
            else:
                raise ValueError(f"Unsupported dataset format detected: {dataset_format}")

            if not training_data:
                raise ValueError(
                    "Training dataset is empty or could not be loaded. Please check the dataset_path and its content.")
            hf_dataset = Dataset.from_list(training_data)
            logger.info(f"Successfully loaded {len(hf_dataset)} samples for fine-tuning in '{dataset_format}' format.")

            # --- Actual fine-tuning logic using Hugging Face libraries ---
            logger.info(f"Loading base model and tokenizer: {base_model_name}")

            tokenizer = AutoTokenizer.from_pretrained(base_model_name)
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
                logger.info(f"Tokenizer pad_token set to eos_token: {tokenizer.pad_token}")

            # Define common model loading kwargs
            model_load_kwargs = {
                "torch_dtype": torch.bfloat16 if (actual_device == "mps" and torch.backends.mps.is_built()) or \
                                                 (actual_device == "cuda" and hasattr(torch,
                                                                                      'bfloat16')) else torch.float32,
                "device_map": "auto" if actual_device == "cuda" else actual_device,
                # Use "auto" for multi-GPU CUDA, else specific device
            }

            # Handle bitsandbytes imports and config conditionally for CUDA
            if actual_device == "cuda":
                # Ensure bitsandbytes is imported and used only for CUDA here
                try:
                    import bitsandbytes as bnb  # LOCAL IMPORT to avoid warning on MPS/CPU
                    quantization_config = bnb.BitsAndBytesConfig(
                        load_in_4bit=True,
                        bnb_4bit_quant_type="nf4",
                        bnb_4bit_compute_dtype=torch.bfloat16,
                        bnb_4bit_use_double_quant=True,
                    )
                    model_load_kwargs["quantization_config"] = quantization_config
                    logger.info("Using bitsandbytes 4-bit quantization for CUDA device.")
                except ImportError:
                    logger.warning(
                        "bitsandbytes not found or not compiled for CUDA. Loading model in full precision on CUDA.")
                except Exception as e:
                    logger.warning(f"Error with bitsandbytes on CUDA: {e}. Loading model in full precision.")

            model = AutoModelForCausalLM.from_pretrained(
                base_model_name,
                **model_load_kwargs
            )

            # Prepare model for PEFT training
            model.gradient_checkpointing_enable()  # Saves memory during training
            # `prepare_model_for_kbit_training` is specifically for bitsandbytes k-bit,
            # so only call it if bitsandbytes was successfully applied.
            if actual_device == "cuda" and "quantization_config" in model_load_kwargs:
                from peft import prepare_model_for_kbit_training  # Local import to avoid issues
                model = prepare_model_for_kbit_training(model)
                logger.info("Model prepared for k-bit training on CUDA.")
            else:
                logger.info("Model prepared for gradient checkpointing (non-kbit).")

            # Configure LoRA
            lora_config = LoraConfig(
                r=hyperparameters.get('lora_rank', 8),
                lora_alpha=hyperparameters.get('lora_alpha', 16),
                target_modules=hyperparameters.get('target_modules', None),
                lora_dropout=hyperparameters.get('lora_dropout', 0.05),
                bias="none",
                task_type="CAUSAL_LM",
            )
            peft_model = get_peft_model(model, lora_config)
            peft_model.print_trainable_parameters()

            use_fp16 = False
            use_bf16 = False
            if hyperparameters.get('mixed_precision_dtype') == 'fp16' and actual_device == "cuda":
                use_fp16 = True
                logger.info("Enabling fp16 mixed precision for CUDA device as requested.")
            elif hyperparameters.get('mixed_precision_dtype') == 'bf16':
                if actual_device == "mps" and torch.backends.mps.is_built():
                    use_bf16 = True
                    logger.info("Enabling bf16 mixed precision for MPS device as requested.")
                elif actual_device == "cuda" and hasattr(torch, 'bfloat16'):
                    use_bf16 = True
                    logger.info("Enabling bf16 mixed precision for CUDA device as requested.")
                else:
                    logger.warning(f"Requested bf16 mixed precision but device '{actual_device}' does not fully support it. Falling back to float32.")
            elif hyperparameters.get('mixed_precision_dtype') != 'no':
                logger.warning(f"Requested mixed precision '{hyperparameters.get('mixed_precision_dtype')}' not supported on device '{actual_device}'. Falling back to float32.")

            # Define TrainingArguments
            training_args = TrainingArguments(
                output_dir=adapter_path,
                per_device_train_batch_size=hyperparameters.get("batch_size", 4),
                gradient_accumulation_steps=hyperparameters.get("gradient_accumulation_steps", 1),
                num_train_epochs=hyperparameters.get("epochs", 3),
                learning_rate=hyperparameters.get("learning_rate", 2e-4),
                logging_dir=str(Path(training_logs_path).parent / "hf_logs"),
                logging_steps=hyperparameters.get("logging_steps", 10),
                save_steps=hyperparameters.get("save_steps", 500),
                save_total_limit=hyperparameters.get("save_total_limit", 1),
                report_to="tensorboard",
                fp16=use_fp16,  # fp16 for CUDA
                bf16=use_bf16,
                optim="paged_adamw_8bit" if actual_device == "cuda" and "quantization_config" in model_load_kwargs else "adamw_torch",
                # Use paged_adamw_8bit only if bitsandbytes is active on CUDA
                load_best_model_at_end=False,
            )

            # Prepare evaluation dataset if specified
            eval_dataset = None
            if "eval_dataset_path" in job_config.model_fields and job_config.eval_dataset_path:
                logger.info(f"Loading evaluation data from: {job_config.eval_dataset_path}")
                if dataset_format == "jsonl-instruction":
                    eval_loader = JsonlInstructionLoader(job_config.eval_dataset_path)
                    eval_data = eval_loader.load()
                elif dataset_format == "conversational-jsonl":
                    eval_loader = ConversationalJsonlLoader(job_config.eval_dataset_path)
                    eval_data = eval_loader.load()
                else:
                    raise ValueError(f"Unsupported evaluation dataset format: {dataset_format}")

                eval_dataset = Dataset.from_list(eval_data)
                logger.info(f"Successfully loaded {len(eval_dataset)} evaluation samples.")

            # Determine the 'formatting_func' for SFTTrainer based on dataset_format
            def instruction_formatting_function(example):
                return example["instruction"] + "\n" + example.get("input", "") + "\n" + example["output"]

            def conversational_formatting_function(example):
                text = ""
                for message in example["messages"]:
                    if message["role"] == "user":
                        text += f"### Human: {message['content']}\n"
                    elif message["role"] == "assistant":
                        text += f"### Assistant: {message['content']}\n"
                return text.strip()

            formatting_func = instruction_formatting_function if dataset_format == "jsonl-instruction" else conversational_formatting_function

            trainer = SFTTrainer(
                model=peft_model,
                args=training_args,
                train_dataset=hf_dataset,
                eval_dataset=eval_dataset,
                formatting_func=formatting_func,
            )

            logger.info("Starting actual fine-tuning process...")
            trainer.train()

            trainer.save_model(adapter_path)
            tokenizer.save_pretrained(adapter_path)
            logger.info(f"Fine-tuned adapter saved to {adapter_path}")

            final_metrics = trainer.state.log_history[-1] if trainer.state.log_history else {}
            actual_final_loss = final_metrics.get("train_loss", None)
            actual_validation_loss = final_metrics.get("eval_loss", None)

            start_time_record = initial_db_record.created_at
            end_time = datetime.utcnow()
            duration_seconds = (end_time - start_time_record).total_seconds()
            estimated_gpu_hours = duration_seconds / 3600
            estimated_cost = estimated_gpu_hours * hyperparameters.get("cost_per_gpu_hour", 0.5)

            self._update_fine_tune_job_record(
                adapter_id,
                FineTuningStatus.COMPLETED,
                final_loss=actual_final_loss,
                validation_loss=actual_validation_loss,
                gpu_hours_consumed=estimated_gpu_hours,
                estimated_training_cost_usd=estimated_cost,
            )
            logger.info(
                f"Fine-tuning job '{job_name}' (ID: {adapter_id}) completed successfully. Model saved to {adapter_path}")

        except Exception as e:
            logger.error(f"Fine-tuning job '{job_name}' (ID: {adapter_id}) failed: {e}", exc_info=True)
            self._update_fine_tune_job_record(adapter_id, FineTuningStatus.FAILED)
            raise

        return initial_db_record