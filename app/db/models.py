# app/db/models.py
from sqlalchemy import (
    Column, Integer, String, Text, DateTime, ForeignKey, MetaData, Table,
    UniqueConstraint, Float, Boolean, Enum, Index, JSON
)
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

# --- Standardized Setup ---
metadata = MetaData()
# Use a standard Python function for timestamps to ensure compatibility
utc_timestamp = datetime.utcnow
sender_enum = Enum("human", "ai", name="sender_enum")

# --- Table Definitions ---

users_table = Table(
    "users", metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("user_id", String(255), nullable=False, unique=True, index=True),
    Column("email", String(255), nullable=True, unique=True),
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
    Column("thread_id", String(255), nullable=False, unique=True, index=True),
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
    Column("name", String(255), nullable=False, unique=True, index=True),
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
    Column("session_id", Integer, ForeignKey("chat_sessions.id", ondelete="CASCADE"), nullable=False, index=True),
    Column("request_id", String(64), nullable=False, index=True),
    Column("prompt_tokens", Integer, nullable=False, default=0),
    Column("completion_tokens", Integer, nullable=False, default=0),
    Column("total_tokens", Integer, nullable=False, default=0),
    Column("retrieval_time_s", Float(precision=10), nullable=True),
    Column("generation_time_s", Float(precision=10), nullable=True),
    Column("estimated_cost_usd", Float(precision=10), nullable=True),
    Column("timestamp", DateTime, default=utc_timestamp, nullable=False),
    UniqueConstraint("session_id", "request_id", name="uq_session_request"),
)

user_api_keys_table = Table(
    "user_api_keys", metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("user_id", Integer, ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True),
    Column("api_key", String(255), nullable=False, unique=True),
    Column("created_at", DateTime, default=utc_timestamp, nullable=False),
    Column("updated_at", DateTime, onupdate=utc_timestamp, default=utc_timestamp, nullable=False),
    Column("revoked", Boolean, nullable=False, default=False),
)

# --- Indexes for Performance ---
Index("chat_messages_session_ts_idx", chat_messages_table.c.session_id, chat_messages_table.c.timestamp)
Index("conv_metrics_session_ts_idx", conversation_metrics_table.c.session_id, conversation_metrics_table.c.timestamp)