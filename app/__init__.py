"""
RAGnetic - Open-source framework for building AI agents with RAG.

This package provides the core functionality for RAGnetic, including:
- Agent configuration and deployment
- Multi-agent workflows
- Vector store integration
- Fine-tuning capabilities
"""

__version__ = "0.1.0"

# Core exports for public API
from app.core.config import get_api_key, find_project_root
from app.core.security import get_current_api_key

__all__ = [
    "__version__",
    "get_api_key", 
    "find_project_root",
    "get_current_api_key",
]