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
workflow_status_enum = Enum("running", "completed", "failed", "paused", name="workflow_status_enum")
agent_status_enum = Enum("running", "completed", "failed", name="agent_status_enum")


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
    Column("permission", String(255), nullable=False), # e.g., 'read:workflows', 'create:agents'
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
    # Use a Python-level default instead of a server-level default
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
)

user_api_keys_table = Table(
    "user_api_keys", metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("user_id", Integer, ForeignKey("users.id", ondelete="CASCADE"), nullable=False),
    Column("api_key", String(255), nullable=False, unique=True),
    Column("created_at", DateTime, default=utc_timestamp, nullable=False),
    Column("updated_at", DateTime, onupdate=utc_timestamp, default=utc_timestamp, nullable=False),
    Column("revoked", Boolean, nullable=False, default=False),
)

# Table to store a record of each overall agent/graph execution
agent_runs = Table(
    'agent_runs', metadata,
    Column('id', Integer, primary_key=True),
    Column('run_id', String(255), unique=True, nullable=False),
    Column('session_id', Integer, ForeignKey('chat_sessions.id'), nullable=False, index=True),
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
    Column('start_time', DateTime, nullable=False),
    Column('end_time', DateTime, nullable=True),
    Column('inputs', JSON, nullable=True),
    Column('outputs', JSON, nullable=True),
    Column('status', agent_status_enum, nullable=False, default='running')
)


workflows_table = Table(
    "workflows", metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("name", String(255), nullable=False, unique=True),
    Column("agent_name", String(255), nullable=True),
    Column("description", Text, nullable=True),
    Column("definition", JSON, nullable=False),
    Column("last_run_at", DateTime, nullable=True),
    Column("created_at", DateTime, default=utc_timestamp, nullable=False),
    Column("updated_at", DateTime, onupdate=utc_timestamp, default=utc_timestamp, nullable=False),
)

workflow_runs_table = Table(
    "workflow_runs", metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("run_id", String(255), unique=True, nullable=False),
    Column("workflow_id", Integer, ForeignKey("workflows.id"), nullable=False, index=True),
    Column("status", workflow_status_enum, nullable=False, default="running"), # Use the new enum here
    Column("user_id", Integer, ForeignKey("users.id"), nullable=True, index=True),
    Column("start_time", DateTime, default=utc_timestamp, nullable=False),
    Column("end_time", DateTime, nullable=True),
    Column("initial_input", JSON, nullable=True),
    Column("final_output", JSON, nullable=True),
    Column("last_execution_state", JSON, nullable=True) # Stores the state to support pause/resume
)

human_tasks_table = Table(
    "human_tasks", metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("run_id", Integer, ForeignKey("workflow_runs.id", ondelete="CASCADE"), nullable=False),
    Column("task_name", String(255), nullable=False),
    Column("status", String(50), nullable=False, default="pending"), # e.g., pending, completed, cancelled
    Column("created_at", DateTime, default=utc_timestamp, nullable=False),
    Column("updated_at", DateTime, onupdate=utc_timestamp, default=utc_timestamp, nullable=False),
    Column("assigned_to_user_id", Integer, ForeignKey("users.id"), nullable=True),
    Column("payload", JSON, nullable=True), # The data the human needs to review
    Column("resolution_data", JSON, nullable=True), # The data submitted by the human
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
    Column("adapter_id", String(255), nullable=False, unique=True), # System-generated UUID for the adapter
    Column("job_name", String(255), nullable=False, index=True), # User-defined name from YAML for tracking
    Column("base_model_name", String(255), nullable=False),      # e.g., 'ollama/llama2', 'mistral'
    Column("adapter_path", String(512), nullable=False),         # Absolute file path to saved weights
    Column("training_dataset_id", String(512), nullable=True),   # Path or identifier of the dataset used
    Column("training_status", fine_tuning_status_enum, nullable=False, default="pending"),
    Column("training_logs_path", String(512), nullable=True),
    Column("hyperparameters", JSON, nullable=True),              # JSON representation of hyperparameters used
    Column("final_loss", Float, nullable=True),
    Column("validation_loss", Float, nullable=True),
    Column("gpu_hours_consumed", Float, nullable=True),
    Column("estimated_training_cost_usd", Float, nullable=True),
    Column("created_by_user_id", Integer, ForeignKey("users.id"), nullable=False, index=True),
    Column("created_at", DateTime, default=datetime.utcnow, nullable=False),
    Column("updated_at", DateTime, onupdate=datetime.utcnow, default=datetime.utcnow, nullable=False),
    Index("fine_tuned_models_status_idx", "training_status"),
    Index("fine_tuned_models_base_model_idx", "base_model_name"),
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


# --- Indexes for Performance ---
Index("tmp_docs_exp_idx", temporary_documents_table.c.expires_at)

Index("chat_messages_session_ts_idx", chat_messages_table.c.session_id, chat_messages_table.c.timestamp)
Index("conv_metrics_session_ts_idx", conversation_metrics_table.c.session_id, conversation_metrics_table.c.timestamp)
Index("workflow_runs_workflow_idx", workflow_runs_table.c.workflow_id)
Index("workflow_runs_user_idx", workflow_runs_table.c.user_id)

