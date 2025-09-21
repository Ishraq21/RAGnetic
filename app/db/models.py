from sqlalchemy import (
    Column, Integer, String, Text, DateTime, ForeignKey, MetaData, Table,
    UniqueConstraint, Float, Boolean, Enum, Index, JSON, BigInteger, ForeignKeyConstraint
)
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

# --- Standardized Setup ---
# Use a standard Python function for timestamps to ensure compatibility
naming_convention = {
    "ix":  "ix_%(table_name)s_%(column_0_name)s",
    "uq":  "uq_%(table_name)s_%(column_0_name)s",
    "ck":  "ck_%(table_name)s_%(constraint_name)s",
    "fk":  "fk_%(table_name)s_%(column_0_name)s_%(referred_table_name)s",
    "pk":  "pk_%(table_name)s"
}
metadata = MetaData(naming_convention=naming_convention)
utc_timestamp = datetime.utcnow
sender_enum = Enum("human", "ai", name="sender_enum")
agent_status_enum = Enum("running", "completed", "failed", name="agent_status_enum")
agent_deployment_status_enum = Enum("created", "deploying", "deployed", "idle", "error", "stopped", name="agent_deployment_status_enum")
api_key_scope_enum = Enum("admin", "editor", "viewer", name="api_key_scope_enum")
benchmark_status_enum = Enum('running', 'completed', 'failed', 'cancelled', name='benchmark_status')


# --- Table Definitions ---

organizations_table = Table(
    "organizations", metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("name", String(255), nullable=False, unique=True),
    Column("created_at", DateTime, default=utc_timestamp, nullable=False),
    Column("updated_at", DateTime, onupdate=utc_timestamp, default=utc_timestamp, nullable=False)
)

roles_table = Table(
    "roles", metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("name", String(50), nullable=False, unique=True),
    Column("description", Text, nullable=True)
)

# Junction table for users and organizations (many-to-many)
user_organizations_table = Table(
    "user_organizations", metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("user_id", Integer, ForeignKey("users.id", ondelete="CASCADE"), nullable=False),
    Column("organization_id", Integer, ForeignKey("organizations.id", ondelete="CASCADE"), nullable=False),
    Column("role_id", Integer, ForeignKey("roles.id"), nullable=False),
    UniqueConstraint("user_id", "organization_id", name="uq_user_organization")
)

# Junction table for roles and permissions (many-to-many)
role_permissions_table = Table(
    "role_permissions", metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("role_id", Integer, ForeignKey("roles.id", ondelete="CASCADE"), nullable=False),
    Column("permission", String(255), nullable=False), # e.g., 'create:agents', 'read:users'
    UniqueConstraint("role_id", "permission", name="uq_role_permission")
)

users_table = Table(
    "users", metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("user_id", String(255), nullable=False, unique=True),
    Column("email", String(255), nullable=True, unique=True),
    Column("first_name", String(100), nullable=True),
    Column("last_name", String(100), nullable=True),
    Column("hashed_password", String(255), nullable=False), # Store hashed passwords
    Column("is_active", Boolean, nullable=False, default=True), # User account status
    Column("is_superuser", Boolean, nullable=False, default=False), # For master admin accounts
    Column("created_at", DateTime, default=utc_timestamp, nullable=False),
    Column("updated_at", DateTime, onupdate=utc_timestamp, default=utc_timestamp, nullable=False),
)

chat_sessions_table = Table(
    "chat_sessions", metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("agent_name", String(255), nullable=False),
    Column("user_id", Integer, ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True),
    Column("topic_name", String(255), nullable=True),
    Column("thread_id", String(255), nullable=False, unique=True),
    Column("created_at", DateTime, default=utc_timestamp, nullable=False),
    Column("updated_at", DateTime, onupdate=utc_timestamp, default=utc_timestamp, nullable=False),
)

chat_messages_table = Table(
    "chat_messages", metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("session_id", Integer, ForeignKey("chat_sessions.id", ondelete="CASCADE"), nullable=False, index=True),
    Column("sender", sender_enum, nullable=False),
    Column("content", Text, nullable=False),
    Column("timestamp", DateTime, default=utc_timestamp, nullable=False),
    Column("meta", JSON, nullable=True)

)

memory_entries_table = Table(
    "memory_entries", metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("session_id", Integer, ForeignKey("chat_sessions.id", ondelete="CASCADE"), nullable=False, index=True),
    Column("key", String(255), nullable=False),
    Column("value", JSON, nullable=False),
    Column("timestamp", DateTime, default=utc_timestamp, nullable=False),
    UniqueConstraint("session_id", "key", name="uq_memory_session_key"),
)

ragnetic_logs_table = Table(
    'ragnetic_logs', metadata,
    Column('id', Integer, primary_key=True, autoincrement=True),
    Column('timestamp', DateTime, default=utc_timestamp, nullable=False),
    Column('level', String(50)),
    Column('message', Text),
    Column('module', String(255)),
    Column('function', String(255)),
    Column('line', Integer),
    Column('exc_info', Text, nullable=True),
    Column('details', JSON, nullable=True)
)

agents_table = Table(
    "agents", metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("name", String(255), nullable=False, unique=True),
    Column("display_name", String(255), nullable=True),
    Column("description", Text, nullable=True),
    Column("model_name", String(255), nullable=False),
    Column("embedding_model", String(255), nullable=True),
    Column("status", agent_deployment_status_enum, nullable=False, default="created", index=True),
    Column("last_updated", DateTime, onupdate=utc_timestamp, default=utc_timestamp, nullable=False),
    Column("last_run", DateTime, nullable=True),
    Column("total_cost", Float(precision=10), nullable=False, default=0.0),
    Column("gpu_instance_id", String(255), nullable=True),
    Column("deployment_type", String(50), nullable=True),
    Column("tags", JSON, nullable=True),
    Column("user_id", Integer, ForeignKey("users.id"), nullable=True),
    Column("created_at", DateTime, default=utc_timestamp, nullable=False),
    Column("updated_at", DateTime, onupdate=utc_timestamp, default=utc_timestamp, nullable=False),
)

agent_tools_table = Table(
    "agent_tools", metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("agent_id", Integer, ForeignKey("agents.id", ondelete="CASCADE"), nullable=False, index=True),
    Column("tool_name", String(100), nullable=False),
    Column("tool_config", JSON, nullable=True),
    UniqueConstraint("agent_id", "tool_name", name="uq_agent_tool"),
)

conversation_metrics_table = Table(
    "conversation_metrics", metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("session_id", Integer, ForeignKey("chat_sessions.id", ondelete="CASCADE"), nullable=True, index=True),
    Column("request_id", String(64), nullable=False, index=True),
    Column("prompt_tokens", Integer, nullable=False, default=0),
    Column("completion_tokens", Integer, nullable=False, default=0),
    Column("total_tokens", Integer, nullable=False, default=0),
    Column("retrieval_time_s", Float(precision=10), nullable=True),
    Column("generation_time_s", Float(precision=10), nullable=True),
    Column("estimated_cost_usd", Float(precision=10), nullable=True),
    Column("llm_model", String(255), nullable=True),
    Column("embedding_cost_usd", Float(precision=10), nullable=True),
    Column("timestamp", DateTime, default=utc_timestamp, nullable=False),
    Column("fine_tuned_model_id", String(255), nullable=True, index=True),
    UniqueConstraint("session_id", "request_id", name="uq_session_request"),
    ForeignKeyConstraint(
        ["fine_tuned_model_id"],
        ["fine_tuned_models.adapter_id"],
        name="fk_conv_metrics_finetuned_adapter",
        ondelete="SET NULL",
    ),

)

user_api_keys_table = Table(
    "user_api_keys", metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("user_id", Integer, ForeignKey("users.id", ondelete="CASCADE"), nullable=False),
    Column("api_key", String(255), nullable=False, unique=True),
    Column("created_at", DateTime, default=utc_timestamp, nullable=False),
    Column("updated_at", DateTime, onupdate=utc_timestamp, default=utc_timestamp, nullable=False),
    Column("revoked", Boolean, nullable=False, default=False),
    Column("scope", api_key_scope_enum, nullable=False, default="viewer"),
    Column("last_used_at", DateTime, nullable=True),
    Column("request_count", BigInteger, nullable=False, default=0),
)

# Table to store a record of each overall agent/graph execution
agent_runs = Table(
    'agent_runs', metadata,
    Column('id', Integer, primary_key=True),
    Column('run_id', String(255), unique=True, nullable=False),
    Column('session_id', Integer, ForeignKey('chat_sessions.id'), nullable=False, index=True),
    Column('parent_run_id', String(255), ForeignKey('agent_runs.run_id'), nullable=True, index=True),

    Column('start_time', DateTime, nullable=False),
    Column('end_time', DateTime, nullable=True),
    Column('status', agent_status_enum, nullable=False, default='running'),
    Column('initial_messages', JSON, nullable=True),
    Column('final_state', JSON, nullable=True)
)

# Table to store the details of each individual step within a run
agent_run_steps = Table(
    'agent_run_steps', metadata,
    Column('id', Integer, primary_key=True),
    Column('agent_run_id', Integer, ForeignKey('agent_runs.id'), nullable=False),
    Column('node_name', String(255), nullable=False),
    Column('parent_run_id', String(255), ForeignKey('agent_runs.run_id'), nullable=True, index=True),
    Column('start_time', DateTime, nullable=False),
    Column('end_time', DateTime, nullable=True),
    Column('inputs', JSON, nullable=True),
    Column('outputs', JSON, nullable=True),
    Column('status', agent_status_enum, nullable=False, default='running')
)




crontab_schedule_table = Table(
    "crontab_schedule", metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("minute", String(64), default="*"),
    Column("hour", String(64), default="*"),
    Column("day_of_week", String(64), default="*"),
    Column("day_of_month", String(64), default="*"),
    Column("month_of_year", String(64), default="*"),
)

interval_schedule_table = Table(
    "interval_schedule", metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("every", Integer, nullable=False),
    Column("period", String(24)),
)

periodic_task_table = Table(
    "periodic_task", metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("name", String(255), unique=True),
    Column("task", String(255)),
    Column("interval_id", Integer, ForeignKey("interval_schedule.id")),
    Column("crontab_id", Integer, ForeignKey("crontab_schedule.id")),
    Column("args", Text, default="[]"),
    Column("kwargs", Text, default="{}"),
    Column("queue", String(255)),
    Column("exchange", String(255)),
    Column("routing_key", String(255)),
    Column("expires", DateTime),
    Column("enabled", Boolean, default=True),
    Column("last_run_at", DateTime),
    Column("total_run_count", Integer, default=0),
    Column("date_changed", DateTime),
    Column("description", Text),
)

periodic_task_changed_table = Table(
    "periodic_task_changed", metadata,
    Column("id", Integer, primary_key=True),
    Column("last_update", DateTime, nullable=False),
)

fine_tuning_status_enum = Enum("pending", "running", "completed", "failed", "paused", name="fine_tuning_status_enum")

fine_tuned_models_table = Table(
    "fine_tuned_models", metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("adapter_id", String(255), nullable=False, unique=True),
    Column("job_name", String(255), nullable=False, index=True),
    Column("base_model_name", String(255), nullable=False),
    Column("adapter_path", String(512), nullable=False),
    Column("training_dataset_id", String(512), nullable=True),
    Column("training_status", fine_tuning_status_enum, nullable=False, default="pending"),
    Column("training_logs_path", String(512), nullable=True),
    Column("hyperparameters", JSON, nullable=True),
    Column("final_loss", Float, nullable=True),
    Column("validation_loss", Float, nullable=True),

    Column("worker_host", String(255), nullable=True),
    Column("worker_pid", Integer, nullable=True),
    Column("device", String(32), nullable=True),
    Column("gpu_name", String(255), nullable=True),
    Column("mixed_precision", String(16), nullable=True),   # 'no'|'fp16'|'bf16'
    Column("bitsandbytes_4bit", Boolean, nullable=True),
    Column("seed", Integer, nullable=True),

    Column("current_step", Integer, nullable=True),
    Column("max_steps", Integer, nullable=True),
    Column("eta_seconds", Float, nullable=True),

    Column("eval_dataset_id", String(512), nullable=True),
    Column("eval_metrics", JSON, nullable=True),
    Column("best_checkpoint_path", String(512), nullable=True),

    Column("gpu_hours_consumed", Float, nullable=True),
    Column("estimated_training_cost_usd", Float, nullable=True),
    Column("created_by_user_id", Integer, ForeignKey("users.id"), nullable=False, index=True),
    Column("created_at", DateTime, default=datetime.utcnow, nullable=False),
    Column("updated_at", DateTime, onupdate=datetime.utcnow, default=datetime.utcnow, nullable=False),

    Index("fine_tuned_models_status_idx", "training_status"),
    Index("fine_tuned_models_base_model_idx", "base_model_name"),
    Index("fine_tuned_models_created_at_idx", "created_at"),  # NEW index
)


temporary_documents_table = Table(
    "temporary_documents", metadata,
    Column("id",           Integer,       primary_key=True, autoincrement=True),
    Column("temp_doc_id",  String(36),    unique=True,      nullable=False),
    Column("user_id",      Integer,       ForeignKey("users.id", ondelete="CASCADE"), index=True),
    Column("thread_id",    String(255),   index=True),
    Column("original_name",Text,          nullable=False),
    Column("file_size",    BigInteger,    nullable=False),
    Column("created_at",   DateTime,      default=utc_timestamp, nullable=False),
    Column("expires_at",   DateTime),                    # optional TTL; nullable
    Column("cleaned_up",   Boolean,       default=False),# set → True after sweep
)

document_chunks_table = Table(
    "document_chunks",
    metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("document_name", String(512), nullable=False, index=True),
    Column("chunk_index", Integer, nullable=False),
    Column("content", Text, nullable=False),
    Column("page_number", Integer),
    Column("row_number", Integer),
    Column("created_at", DateTime, default=utc_timestamp, nullable=False),

    # Link back to a temp doc (nullable)
    Column("temp_document_id", Integer, nullable=True, index=True),

    UniqueConstraint("document_name", "chunk_index", name="uq_document_chunk"),

    ForeignKeyConstraint(
        ["temp_document_id"],
        ["temporary_documents.id"],
        name="fk_document_chunks_temp_document",  # named via convention but explicit is clearer
        ondelete="CASCADE",
    ),
)

citations_table = Table(
    "citations", metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("message_id", Integer, ForeignKey("chat_messages.id", ondelete="CASCADE"), nullable=False, index=True),
    Column("chunk_id", Integer, ForeignKey("document_chunks.id", ondelete="CASCADE"), nullable=False, index=True),
    Column("marker_text", String(255), nullable=False),
    Column("start_char", Integer, nullable=False),
    Column("end_char", Integer, nullable=False),
)

lambda_runs = Table(
    "lambda_runs",
    metadata,
    Column("id", Integer, primary_key=True),
    Column("run_id", String(255), unique=True, nullable=False),
    Column("user_id", Integer, ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True),
    Column("status", String(50), nullable=False, default="pending"),  # pending, running, completed, failed
    Column("initial_request", JSON, nullable=False),
    Column("final_state", JSON, nullable=True),
    Column("start_time", DateTime, default=utc_timestamp, nullable=False),
    Column("end_time", DateTime, nullable=True),
    Column("error_message", Text, nullable=True),
    Column("logs", Text, nullable=True),
    Column("thread_id", String(255), ForeignKey("chat_sessions.thread_id", ondelete="CASCADE"), index=True),  # NEW


)

benchmark_runs_table = Table(
    "benchmark_runs", metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("run_id", String(255), unique=True, nullable=False, index=True),
    Column("agent_name", String(255), nullable=False, index=True),
    Column("dataset_id", String(255), nullable=True, index=True),
    Column("prompt_hash", String(64), nullable=True),
    Column("agent_config_hash", String(64), nullable=True),
    Column("judge_model", String(255), nullable=True),
    Column("config_snapshot", JSON, nullable=True),
    Column("total_items", Integer, default=0),
    Column("completed_items", Integer, default=0),
    Column("started_at", DateTime, default=utc_timestamp, nullable=False),
    Column("ended_at", DateTime, nullable=True),
    Column("status", benchmark_status_enum, nullable=False, default="running"),
    Column("summary_metrics", JSON, nullable=True),
    Column("error", Text, nullable=True),
)

benchmark_items_table = Table(
    "benchmark_items", metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("run_id", String(255), ForeignKey("benchmark_runs.run_id", ondelete="CASCADE"), index=True, nullable=False),
    Column("item_index", Integer, nullable=False),
    Column("question", Text, nullable=False),
    Column("ground_truth_chunk_id", String(255), nullable=True),
    Column("retrieved_ids", JSON, nullable=True),
    Column("retrieval_metrics", JSON, nullable=True),
    Column("context_size", Integer, nullable=True),
    Column("answer", Text, nullable=True),
    Column("judge_scores", JSON, nullable=True),
    Column("token_usage", JSON, nullable=True),
    Column("costs", JSON, nullable=True),
    Column("durations", JSON, nullable=True),
    Column("citations", JSON, nullable=True),
    Column("created_at", DateTime, default=utc_timestamp, nullable=False),
    UniqueConstraint("run_id", "item_index", name="uq_bench_item_run_idx"),
)


# --- GPU Infrastructure Tables ---

# Enum for deployment status
deployment_status_enum = Enum("pending", "active", "inactive", "failed", name="deployment_status_enum")

# Enum for deployment type
deployment_type_enum = Enum("api", "webhook", "streaming", name="deployment_type_enum")

# Enum for transaction type
transaction_type_enum = Enum("credit", "debit", "refund", name="transaction_type_enum")

# Enum for session type
session_type_enum = Enum("training", "inference", "deployment", name="session_type_enum")

# Enum for usage type
usage_type_enum = Enum("training", "inference", "deployment", name="usage_type_enum")



user_credits_table = Table(
    "user_credits", metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("user_id", Integer, ForeignKey("users.id", ondelete="CASCADE"), nullable=False, unique=True),
    Column("balance", Float(precision=10), nullable=False, default=0.0),
    Column("daily_limit", Float(precision=10), nullable=True),
    Column("total_spent", Float(precision=10), nullable=False, default=0.0),
    Column("updated_at", DateTime, onupdate=utc_timestamp, default=utc_timestamp, nullable=False),
    Index("ix_user_credits_user_id", "user_id"),
)

credit_transactions_table = Table(
    "credit_transactions", metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("user_id", Integer, ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True),
    Column("amount", Float(precision=10), nullable=False),
    Column("transaction_type", transaction_type_enum, nullable=False),
    Column("description", Text, nullable=True),
    Column("created_at", DateTime, default=utc_timestamp, nullable=False),
    Index("ix_credit_transactions_user_id_created_at", "user_id", "created_at"),
)



deployments_table = Table(
    "deployments", metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("user_id", Integer, ForeignKey("users.id", ondelete="CASCADE"), nullable=False),
    Column("agent_id", Integer, ForeignKey("agents.id", ondelete="CASCADE"), nullable=False),
    Column("deployment_type", deployment_type_enum, nullable=False, index=True),
    Column("status", deployment_status_enum, nullable=False, default="pending"),
    Column("api_key_id", Integer, ForeignKey("api_keys.id", ondelete="SET NULL"), nullable=True),
    Column("endpoint_path", String(255), nullable=True),
    Column("created_at", DateTime, default=utc_timestamp, nullable=False),
    Column("updated_at", DateTime, onupdate=utc_timestamp, default=utc_timestamp, nullable=False),
)

api_keys_table = Table(
    "api_keys", metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("user_id", Integer, ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True),
    Column("name", String(255), nullable=False),
    Column("key_hash", String(255), nullable=False, unique=True),
    Column("scope", api_key_scope_enum, nullable=False, index=True),
    Column("is_active", Boolean, nullable=False, default=True, index=True),
    Column("created_at", DateTime, default=utc_timestamp, nullable=False),
    Column("last_used_at", DateTime, nullable=True),
    Index("ix_api_keys_user_id_is_active_scope", "user_id", "is_active", "scope"),
)

api_usage_table = Table(
    "api_usage", metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("api_key_id", Integer, ForeignKey("api_keys.id", ondelete="CASCADE"), nullable=False, index=True),
    Column("deployment_id", Integer, ForeignKey("deployments.id", ondelete="CASCADE"), nullable=True),
    Column("request_count", Integer, nullable=False, default=0),
    Column("total_cost", Float(precision=10), nullable=False, default=0.0),
    Column("last_request", DateTime, nullable=True),
    Column("created_at", DateTime, default=utc_timestamp, nullable=False),
    Index("ix_api_usage_api_key_id_created_at", "api_key_id", "created_at"),
)

# --- Indexes for Performance ---
Index("tmp_docs_exp_idx", temporary_documents_table.c.expires_at)

Index("chat_messages_session_ts_idx", chat_messages_table.c.session_id, chat_messages_table.c.timestamp)
Index("conv_metrics_session_ts_idx", conversation_metrics_table.c.session_id, conversation_metrics_table.c.timestamp)

