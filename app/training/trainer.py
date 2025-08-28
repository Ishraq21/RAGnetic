# app/training/trainer.py
from inspect import signature
from pathlib import Path
from typing import Dict, Any, Optional
from datetime import datetime
import uuid
import random
import os
import json
import hashlib

import numpy as np
import torch

from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from datasets import Dataset
from trl import SFTTrainer

from app.schemas.fine_tuning import FineTuningJobConfig, FineTuningStatus, FineTunedModel
from app.db import get_sync_db_engine
from app.db.models import fine_tuned_models_table

from app.training.data_prep.jsonl_instruction_loader import JsonlInstructionLoader
from app.training.data_prep.conversational_jsonl_loader import ConversationalJsonlLoader
from app.schemas.data_prep import DatasetPreparationConfig

import logging
from transformers import logging as hf_logging
from datasets.utils.logging import disable_progress_bar as ds_disable_pb
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer
from transformers.trainer_callback import TrainerCallback, TrainerControl, TrainerState


logging.getLogger("transformers.trainer").setLevel(logging.ERROR)
hf_logging.set_verbosity_error()

ds_disable_pb()
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"

logger = logging.getLogger(__name__)



class DbProgressCallback(TrainerCallback):
    """
    Periodically persists progress (current_step/eta) and heartbeat updated_at.
    Uses a synchronous engine to avoid event loop entanglement in Trainer thread.
    """
    def __init__(self, db_engine, adapter_id: str, log_every_steps: int = 10):
        self.db_engine = db_engine
        self.adapter_id = adapter_id
        self.log_every_steps = max(1, log_every_steps)

    def on_log(self, args, state: TrainerState, control: TrainerControl, **kwargs):
        # state.global_step increments; log every N steps to reduce DB writes
        if state.global_step is None:
            return
        if state.global_step % self.log_every_steps != 0:
            return

        with self.db_engine.begin() as conn:
            conn.execute(
                fine_tuned_models_table.update()
                .where(fine_tuned_models_table.c.adapter_id == self.adapter_id)
                .values(
                    current_step=int(state.global_step),
                    max_steps=int(state.max_steps) if state.max_steps else None,
                    eta_seconds=float(state.epoch) if isinstance(state.epoch, (float, int)) else None,  # epoch used as coarse progress if ETA unknown
                    updated_at=datetime.utcnow(),
                )
            )


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
            user_id: int,
    ) -> FineTunedModel:
        # --- Reproducibility seed ---
        seed = int(os.getenv("RAGNETIC_SEED", "0")) or random.randint(1, 2_000_000_000)
        random.seed(seed)
        np.random.seed(seed % (2 ** 32 - 1))
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
            # Determinism trade-offs; safer defaults:
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True

        adapter_id = job_config.adapter_id or str(uuid.uuid4())
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
        logger.info(f"Seed fixed to: {seed}")

        # Create/ensure DB record + persist seed (if row exists from API, we update ‘seed’ here)
        created_record_row = None
        with self.db_engine.connect() as connection:
            existing = connection.execute(
                fine_tuned_models_table.select().where(fine_tuned_models_table.c.adapter_id == adapter_id)
            ).mappings().first()

            if existing:
                connection.execute(
                    fine_tuned_models_table.update()
                    .where(fine_tuned_models_table.c.adapter_id == adapter_id)
                    .values(
                        job_name=job_name,
                        base_model_name=base_model_name,
                        adapter_path=adapter_path,
                        training_dataset_id=dataset_path_str,
                        training_logs_path=training_logs_path,
                        hyperparameters=hyperparameters,
                        seed=seed,
                        updated_at=datetime.utcnow(),
                    )
                )
                connection.commit()
                created_record_row = existing
                logger.info(f"Using existing DB record for adapter_id={adapter_id}. Seed persisted.")
            else:
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
                    "seed": seed,
                }
                result = connection.execute(
                    fine_tuned_models_table.insert().values(**new_job_record_data).returning(fine_tuned_models_table)
                )
                created_record_row = result.mappings().first()
                connection.commit()
                logger.info(f"Created DB record for adapter_id={adapter_id} with seed {seed}.")

        initial_db_record = FineTunedModel.model_validate(created_record_row)

        # ---- helpers ----
        def _validate_instruction_rows(rows):
            bad = 0
            for r in rows:
                if "instruction" not in r or "output" not in r:
                    bad += 1
                elif not str(r["instruction"]).strip() or not str(r["output"]).strip():
                    bad += 1
            return bad

        def _validate_conversation_rows(rows):
            bad = 0
            for r in rows:
                msgs = r.get("messages")
                if not isinstance(msgs, list) or not msgs:
                    bad += 1;
                    continue
                # basic structure: alternating roles, text non-empty
                for m in msgs:
                    if "role" not in m or "content" not in m:
                        bad += 1;
                        break
                    if m["role"] not in ("user", "assistant"):
                        bad += 1;
                        break
                    if not str(m["content"]).strip():
                        bad += 1;
                        break
            return bad

        # ensure we can reference this in finally even if attach fails early
        file_handler: Optional[logging.FileHandler] = None

        try:
            # Attach a file logger for this run
            file_handler = logging.FileHandler(training_logs_path, encoding="utf-8")
            file_handler.setLevel(logging.INFO)
            file_handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s"))
            logger.addHandler(file_handler)

            # --- Load and prepare the dataset (NEEDED before trainer) ---
            logger.info(f"Loading training data from: {dataset_path_str}")
            with open(dataset_path_str, "r", encoding="utf-8") as f:
                first_line = f.readline()

            if "instruction" in first_line and "output" in first_line:
                dataset_format = "jsonl-instruction"
                training_data = JsonlInstructionLoader(dataset_path_str).load()
                bad = _validate_instruction_rows(training_data)
            elif "messages" in first_line and "role" in first_line and "content" in first_line:
                dataset_format = "conversational-jsonl"
                training_data = ConversationalJsonlLoader(dataset_path_str).load()
                bad = _validate_conversation_rows(training_data)
            else:
                raise ValueError(
                    "Could not infer dataset format. Expected 'jsonl-instruction' or 'conversational-jsonl'.")

            if not training_data:
                raise ValueError("Training dataset is empty. Check dataset_path and file contents.")

            total = len(training_data)
            if bad > 0:
                logger.warning(
                    f"Dataset validation: {bad}/{total} rows flagged invalid; they will still be included but may truncate.")
            if total < 10:
                logger.warning(f"Very small dataset ({total} samples). OK for smoke-test, risky for production.")

            hf_dataset = Dataset.from_list(training_data)
            logger.info(f"Loaded {len(hf_dataset)} samples in '{dataset_format}' format for fine-tuning.")

            # --- Tokenizer (needed for training + saving) ---
            tokenizer = AutoTokenizer.from_pretrained(base_model_name)
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
                logger.info(f"Tokenizer pad_token set to eos_token: {tokenizer.pad_token}")

            # Decide device/dtype
            actual_device = "cpu"
            if job_config.device:
                req = job_config.device.lower()
                if req == "cuda" and torch.cuda.is_available():
                    actual_device = "cuda"
                elif req == "mps" and torch.backends.mps.is_available():
                    actual_device = "mps"
                elif req == "cpu":
                    actual_device = "cpu"
            else:
                if torch.cuda.is_available():
                    actual_device = "cuda"
                elif torch.backends.mps.is_available():
                    actual_device = "mps"


            logger.info(f"Training device resolved to: {actual_device}")

            # dtype preference
            dtype = torch.float32
            if hyperparameters.get('mixed_precision_dtype') == 'fp16' and actual_device == "cuda":
                dtype = torch.float16
            elif hyperparameters.get('mixed_precision_dtype') == 'bf16' and (
                (actual_device == "cuda" and hasattr(torch, 'bfloat16')) or
                (actual_device == "mps" and torch.backends.mps.is_available())
            ):
                dtype = torch.bfloat16

            model_load_kwargs = {
                "torch_dtype": dtype,
                "low_cpu_mem_usage": True,
            }

            if actual_device == "cuda":
                model_load_kwargs["device_map"] = "auto"

            # Optional: 4-bit quantization on CUDA
            if actual_device == "cuda":
                try:
                    from transformers import BitsAndBytesConfig
                    quantization_config = BitsAndBytesConfig(
                        load_in_4bit=True,
                        bnb_4bit_quant_type="nf4",
                        bnb_4bit_compute_dtype=torch.bfloat16,
                        bnb_4bit_use_double_quant=True,
                    )
                    model_load_kwargs["quantization_config"] = quantization_config
                    logger.info("Using bitsandbytes 4-bit quantization for CUDA device.")
                except Exception as e:
                    logger.info(f"bitsandbytes unavailable; running without 4-bit: {e}")

            model = AutoModelForCausalLM.from_pretrained(
                base_model_name,
                **model_load_kwargs
            )

            if hasattr(model, "config") and hasattr(model.config, "use_cache"):
                model.config.use_cache = False

            # For MPS/CPU, move the model explicitly (device_map is not used)
            if actual_device in ("mps", "cpu"):
                try:
                    model.to(actual_device)
                except Exception as e:
                    logger.warning(f"Could not move model to {actual_device}: {e}")


            # Prepare model for PEFT training
            model.gradient_checkpointing_enable()  # Saves memory during training
            # `prepare_model_for_kbit_training` is specifically for bitsandbytes k-bit,
            # so only call it if bitsandbytes was successfully applied.
            if actual_device == "cuda" and "quantization_config" in model_load_kwargs:
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
                dataloader_pin_memory=False,
                disable_tqdm=True,
                log_level="info",
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


            # --- Build a single-text field and wire up SFTTrainer ---
            def _format_instruction(example):
                instr = example.get("instruction", "")
                inp = example.get("input", "")
                out = example.get("output", "")
                return (f"### Instruction:\n{instr}\n"
                        f"### Input:\n{inp}\n"
                        f"### Response:\n{out}").strip()

            def _format_conversation(example):
                lines = []
                for msg in example["messages"]:
                    if msg["role"] == "user":
                        lines.append(f"### Human: {msg['content']}")
                    elif msg["role"] == "assistant":
                        lines.append(f"### Assistant: {msg['content']}")
                return "\n".join(lines).strip()

            formatting_func = _format_instruction if dataset_format == "jsonl-instruction" else _format_conversation

            trainer = SFTTrainer(
                model=peft_model,
                args=training_args,
                train_dataset=hf_dataset,
                eval_dataset=eval_dataset,
                formatting_func=formatting_func,
                processing_class=tokenizer,
            )

            # Label names quiets Hugging Face warning for PEFT models
            if hasattr(trainer, "label_names"):
                trainer.label_names = ["labels"]

            # Progress heartbeat
            trainer.add_callback(DbProgressCallback(self.db_engine, adapter_id,
                                                    log_every_steps=max(1, hyperparameters.get("logging_steps", 10))))

            logger.info("Starting actual fine-tuning process...")

            def _try_train_with_args(_args: TrainingArguments, retries_left: int = 1):
                nonlocal trainer
                try:
                    trainer.args = _args  # update in place
                    return trainer.train()
                except torch.cuda.OutOfMemoryError as oom:
                    if _args.per_device_train_batch_size <= 1 or retries_left <= 0:
                        logger.error("CUDA OOM and cannot reduce batch size further.", exc_info=True)
                        raise
                    torch.cuda.empty_cache()
                    new_bs = max(1, _args.per_device_train_batch_size // 2)
                    logger.warning(
                        f"OOM caught. Reducing per_device_train_batch_size from {_args.per_device_train_batch_size} to {new_bs} and retrying once.")
                    new_args = TrainingArguments(**{**_args.to_dict(), "per_device_train_batch_size": new_bs})
                    return _try_train_with_args(new_args, retries_left - 1)

            train_output = _try_train_with_args(training_args, retries_left=1)

            # Save adapter + tokenizer
            trainer.save_model(adapter_path)
            tokenizer.save_pretrained(adapter_path)
            logger.info(f"Fine-tuned adapter saved to {adapter_path}")

            # Build manifest with file checksums
            def _sha256_file(p: Path) -> str:
                h = hashlib.sha256()
                with p.open("rb") as f:
                    for chunk in iter(lambda: f.read(8192), b""):
                        h.update(chunk)
                return h.hexdigest()

            files = {}
            for p in Path(adapter_path).glob("*"):
                if p.is_file():
                    files[p.name] = {"sha256": _sha256_file(p), "size": p.stat().st_size}

            manifest = {
                "adapter_id": adapter_id,
                "job_name": job_name,
                "base_model_name": base_model_name,
                "seed": seed,
                "hyperparameters": hyperparameters,
                "device": "cuda" if torch.cuda.is_available() else (
                    "mps" if torch.backends.mps.is_available() else "cpu"),
                "created_at": datetime.utcnow().isoformat() + "Z",
                "files": files,
                "tokenizer_folder": adapter_path,
            }
            (Path(adapter_path) / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
            logger.info("Wrote artifact manifest.json")

            # Quick smoke-generation (uses in-memory fine-tuned model)
            try:
                peft_model.eval()
                prompt = "### Human: Say hello in one short sentence.\n### Assistant:"
                inputs = tokenizer(prompt, return_tensors="pt")
                device_for_gen = "cuda" if torch.cuda.is_available() else (
                    "mps" if torch.backends.mps.is_available() else "cpu")
                inputs = {k: v.to(device_for_gen) for k, v in inputs.items()}
                with torch.no_grad():
                    out_ids = peft_model.generate(**inputs, max_new_tokens=32, do_sample=False)
                text = tokenizer.decode(out_ids[0], skip_special_tokens=True)
                (Path(adapter_path) / "post_train_smoke.txt").write_text(text, encoding="utf-8")
                logger.info("Post-train smoke generation complete.")
            except Exception as gen_e:
                logger.warning(f"Post-train smoke generation failed (non-fatal): {gen_e}")

            # Metrics and accounting
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
        finally:
            # Detach the per-run file handler
            if file_handler:
                try:
                    logger.removeHandler(file_handler)
                except Exception:
                    pass
                try:
                    file_handler.close()
                except Exception:
                    pass

        return initial_db_record