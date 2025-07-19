import json
from typing import Any, Dict, List
from langchain_core.messages import BaseMessage
from langchain_core.documents import Document

def _serialize_for_db(data: Any) -> Any:
    # Handle Langchain BaseMessage objects, which are the root cause of the error
    if isinstance(data, BaseMessage):
        result = {
            "type": data.__class__.__name__,
            "content": data.content,
        }
        # If the message contains tool calls, serialize them too
        if hasattr(data, "tool_calls"):
            result["tool_calls"] = _serialize_for_db(data.tool_calls)
        # If the messages carry metadata, serialize it recursively
        if hasattr(data, "metadata"):
            result["metadata"] = _serialize_for_db(data.metadata)
        return result
    # Check for dataclasses/Pydantic models with model_dump
    if hasattr(data, 'model_dump'):
        return data.model_dump(mode='json')
    # Handle Langchain Document objects
    if isinstance(data, Document):
        return {"page_content": data.page_content, "metadata": _serialize_for_db(data.metadata)}
    # Handle lists and dictionaries recursively
    if isinstance(data, dict):
        return {key: _serialize_for_db(value) for key, value in data.items()}
    if isinstance(data, list):
        return [_serialize_for_db(item) for item in data]
    # Handle other objects with a .dict() method
    if hasattr(data, 'dict'):
        return data.dict()
    # Default return for primitive types and other serializable objects
    return data