import logging
from typing import Dict, Optional

import tiktoken

logger = logging.getLogger(__name__)

# Pricing Data (in USD per 1 million tokens)
# This data can be updated as provider prices change.
MODEL_PRICING = {
    # OpenAI
    "gpt-4o-mini": {"input": 0.15, "output": 0.60},
    "gpt-4o": {"input": 2.50, "output": 10.00},
    "gpt-4-turbo": {"input": 10.00, "output": 30.00},
    "gpt-3.5-turbo": {"input": 0.50, "output": 1.50},

    # Anthropic
    "claude-3-opus-20240229": {"input": 15.00, "output": 75.00},
    "claude-3-sonnet-20240229": {"input": 3.00, "output": 15.00},
    "claude-3-haiku-20240307": {"input": 0.25, "output": 1.25},

    # Using pricing for prompts <128k tokens for simplicity
    "gemini-1.5-pro-latest": {"input": 1.25, "output": 5.00},
    "gemini-1.5-flash-latest": {"input": 0.075, "output": 0.30},
    "gemini-1.5-flash-8b": {"input": 0.0375, "output": 0.15},

    # Default/Unknown
    "unknown": {"input": 0.0, "output": 0.0}
}

# --- Tokenizer Cache ---
_tokenizer_cache: Dict[str, tiktoken.Encoding] = {}


def get_tokenizer(model_name: str) -> Optional[tiktoken.Encoding]:
    """Get a tokenizer for a given model, using a cache."""
    # Note: This currently only supports tiktoken-compatible models.
    # Gemini token counts are retrieved directly from the API response metadata.
    if model_name in _tokenizer_cache:
        return _tokenizer_cache[model_name]
    try:
        encoding = tiktoken.encoding_for_model(model_name)
        _tokenizer_cache[model_name] = encoding
        return encoding
    except KeyError:
        logger.warning(f"No 'tiktoken' tokenizer found for model '{model_name}'. Token counts will be approximate.")
        return None


def count_tokens(text: str, model_name: str) -> int:
    """
    Counts the number of tokens in a text string for a specific model.
    This provides an *estimate* and is best used for non-critical calculations.
    For exact counts, use the token usage data from the API response.
    """
    # For OpenAI and Anthropic models, tiktoken is accurate.
    if model_name.startswith("gpt-") or model_name.startswith("claude-"):
        tokenizer = get_tokenizer(model_name)
        if tokenizer:
            return len(tokenizer.encode(text))

    # For Gemini and other models, we use an approximation.
    # The LangChain integration for Gemini provides exact counts post-API call.
    return len(text) // 4


def calculate_cost(
        model_name: str,
        prompt_tokens: int = 0,
        completion_tokens: int = 0
) -> float:
    """
    Calculates the estimated cost of an LLM call based on token counts.
    """
    # Find the base model name for pricing lookup
    base_model_name = model_name
    if not base_model_name in MODEL_PRICING:
        # Fallback for versioned models like 'gemini-1.5-pro-001' -> 'gemini-1.5-pro-latest'
        if "gemini-1.5-pro" in base_model_name:
            base_model_name = "gemini-1.5-pro-latest"
        elif "gemini-1.5-flash" in base_model_name:
            base_model_name = "gemini-1.5-flash-latest"

    pricing = MODEL_PRICING.get(base_model_name, MODEL_PRICING["unknown"])

    if pricing["input"] == 0.0 and pricing["output"] == 0.0:
        logger.warning(f"No pricing information for model '{model_name}'. Cost will be reported as 0.")

    # Prices are per 1 million tokens, so we divide by 1,000,000
    input_cost = (prompt_tokens / 1_000_000) * pricing["input"]
    output_cost = (completion_tokens / 1_000_000) * pricing["output"]

    return input_cost + output_cost