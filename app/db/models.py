# app/db/models.py
from sqlalchemy import (
    Column, Integer, String, Text, DateTime, ForeignKey, MetaData, Table,
    UniqueConstraint, Float, Boolean, Enum, Index, JSON
)
from sqlalchemy.sql import func
import logging

logger = logging.getLogger(__name__)
# Define metadata for all tables. This will be used by Alembic for migrations.
metadata = MetaData()
# Create a database-level Enum for the sender role
sender_enum = Enum("human", "ai", name="sender_enum")
users_table = Table(
    "users", metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("user_id", String(255), nullable=False, unique=True, index=True), # Your system's unique user ID
    Column("email", String(255), nullable=True, unique=True), # Optional: for OIDC/OAuth
    Column("created_at", DateTime(timezone=True), server_default=func.now(), nullable=False),
    Column("updated_at", DateTime(timezone=True), onupdate=func.now(), server_default=func.now(), nullable=False),
)
chat_sessions_table = Table(
    "chat_sessions", metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("agent_name", String(255), nullable=False),
    Column("user_id", Integer, ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True),
    Column("thread_id", String(255), nullable=False, unique=True, index=True), # Unique identifier for a chat conversation thread
    Column("created_at", DateTime(timezone=True), server_default=func.now(), nullable=False),
    Column("updated_at", DateTime(timezone=True), onupdate=func.now(), server_default=func.now(), nullable=False),
)
chat_messages_table = Table(
    "chat_messages", metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("session_id", Integer, ForeignKey("chat_sessions.id", ondelete="CASCADE"), nullable=False, index=True),
    Column("sender", sender_enum, nullable=False),
    Column("content", Text, nullable=False),
    Column("timestamp", DateTime(timezone=True), server_default=func.now(), nullable=False),
)
memory_entries_table = Table(
    "memory_entries", metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("session_id", Integer, ForeignKey("chat_sessions.id", ondelete="CASCADE"), nullable=False, index=True),
    Column("key", String(255), nullable=False), # Key for the memory entry
    Column("value", JSON, nullable=False), # Store memory data as JSON
    Column("timestamp", DateTime(timezone=True), server_default=func.now(), nullable=False),
    UniqueConstraint("session_id", "key", name="uq_memory_session_key"), # Ensures unique memory keys per session
)
ragnetic_logs_table = Table(
    'ragnetic_logs', metadata,
    Column('id', Integer, primary_key=True, autoincrement=True),
    Column('timestamp', DateTime(timezone=True), default=func.now()),
    Column('level', String(50)),
    Column('message', Text),
    Column('module', String(255)),
    Column('function', String(255)),
    Column('line', Integer),
    Column('exc_info', Text, nullable=True),
    Column("updated_at", DateTime(timezone=True), onupdate=func.now(), server_default=func.now(), nullable=False),
)
agents_table = Table(
    "agents", metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("name", String(255), nullable=False, unique=True, index=True),
    Column("display_name", String(255), nullable=True),
    Column("description", Text, nullable=True),
    Column("model_name", String(255), nullable=False),
    Column("created_at", DateTime(timezone=True), server_default=func.now(), nullable=False),
    Column("updated_at", DateTime(timezone=True), onupdate=func.now(), server_default=func.now(), nullable=False),
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
    Column("timestamp", DateTime(timezone=True), server_default=func.now(), nullable=False),
    UniqueConstraint("session_id", "request_id", name="uq_session_request"),
)
user_api_keys_table = Table(
    "user_api_keys", metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("user_id", Integer, ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True),
    Column("api_key", String(255), nullable=False, unique=True),
    Column("created_at", DateTime(timezone=True), server_default=func.now(), nullable=False),
    Column("updated_at", DateTime(timezone=True), onupdate=func.now(), server_default=func.now(), nullable=False),
    Column("revoked", Boolean, nullable=False, default=False),
)
# Add composite indexes for faster queries
Index("chat_messages_session_ts_idx", chat_messages_table.c.session_id, chat_messages_table.c.timestamp)
Index("conv_metrics_session_ts_idx", conversation_metrics_table.c.session_id, conversation_metrics_table.c.timestamp)