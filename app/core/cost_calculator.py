import logging
from typing import Dict, Optional

import tiktoken

logger = logging.getLogger(__name__)

# Pricing Data (in USD per 1 million tokens)
# This data can be updated as provider prices change.
MODEL_PRICING = {
    # --- OpenAI Chat Models ---
    "gpt-4.1-2025-04-14": {"input": 2.00, "output": 8.00},
    "gpt-4.1-mini-2025-04-14": {"input": 0.40, "output": 1.60},
    "gpt-4.1-nano-2025-04-14": {"input": 0.10, "output": 0.40},
    "gpt-4.5-preview-2025-02-27": {"input": 75.00, "output": 150.00},
    "gpt-4o-2024-08-06": {"input": 2.50, "output": 10.00},
    "gpt-4o-audio-preview-2025-06-03": {"input": 2.50, "output": 10.00},  # Text tokens
    "gpt-4o-realtime-preview-2025-06-03": {"input": 5.00, "output": 20.00},  # Text tokens
    "gpt-4o-mini-2024-07-18": {"input": 0.15, "output": 0.60},
    "gpt-4o-mini-audio-preview-2024-12-17": {"input": 0.15, "output": 0.60},  # Text tokens
    "gpt-4o-mini-realtime-preview-2024-12-17": {"input": 0.60, "output": 2.40},  # Text tokens
    "o1-2024-12-17": {"input": 15.00, "output": 60.00},
    "o1-pro-2025-03-19": {"input": 150.00, "output": 600.00},
    "o3-pro-2025-06-10": {"input": 20.00, "output": 80.00},
    "o3-2025-04-16": {"input": 2.00, "output": 8.00},
    "o3-deep-research-2025-06-26": {"input": 10.00, "output": 40.00},
    "o4-mini-2025-04-16": {"input": 1.10, "output": 4.40},
    "o4-mini-deep-research-2025-06-26": {"input": 2.00, "output": 8.00},
    "o3-mini-2025-01-31": {"input": 1.10, "output": 4.40},
    "o1-mini-2024-09-12": {"input": 1.10, "output": 4.40},
    "codex-mini-latest": {"input": 1.50, "output": 6.00},
    "gpt-4o-mini-search-preview-2025-03-11": {"input": 0.15, "output": 0.60},
    "gpt-4o-search-preview-2025-03-11": {"input": 2.50, "output": 10.00},
    "computer-use-preview-2025-03-11": {"input": 3.00, "output": 12.00},
    "gpt-4-turbo-2024-04-09": {"input": 10.00, "output": 30.00},  # Alias: gpt-4-turbo
    "gpt-3.5-turbo-0125": {"input": 0.50, "output": 1.50},  # Alias: gpt-3.5-turbo
    "gpt-3.5-turbo-instruct": {"input": 1.50, "output": 2.00},
    "gpt-3.5-turbo-16k-0613": {"input": 3.00, "output": 4.00},
    "gpt-4-0613": {"input": 30.00, "output": 60.00},  # Alias: gpt-4
    "gpt-4-32k": {"input": 60.00, "output": 120.00},
    "davinci-002": {"input": 2.00, "output": 2.00},  # Text models
    "babbage-002": {"input": 0.40, "output": 0.40},  # Text models
    "chatgpt-4o-latest": {"input": 5.00, "output": 15.00},  # Alias from 'Other models'

    # --- Google (Vertex AI) Chat Models (Token-based pricing where applicable) ---
    "gemini-1.5-pro-latest": {"input": 1.25, "output": 10.00},  # Used as alias for Gemini 2.5 Pro
    "gemini-1.5-flash-latest": {"input": 0.30, "output": 2.50},  # Used as alias for Gemini 2.5 Flash GA (text)
    "gemini-2.5-pro": {"input": 1.25, "output": 10.00},  # <=200K tokens
    "gemini-2.5-flash-ga": {"input": 0.30, "output": 2.50},  # Text Input
    "gemini-2.5-flash-preview": {"input": 0.15, "output": 0.60},  # Text output (no thinking)
    "gemini-2.5-flash-live-api": {"input": 0.50, "output": 2.00},  # Text tokens
    "gemini-2.5-flash-lite": {"input": 0.10, "output": 0.40},  # Text Input
    "gemini-2.0-flash": {"input": 0.15, "output": 0.60},
    "gemini-2.0-flash-lite": {"input": 0.075, "output": 0.30},

    # --- Anthropic Chat Models (from Vertex AI pricing) ---
    "claude-3-opus-20240229": {"input": 15.00, "output": 75.00},  # Based on current RAGnetic config
    "claude-3-sonnet-20240229": {"input": 3.00, "output": 15.00},  # Based on current RAGnetic config
    "claude-3-haiku-20240307": {"input": 0.25, "output": 1.25},  # Based on current RAGnetic config
    "claude-3-5-haiku": {"input": 0.80, "output": 4.00},
    "claude-3-5-sonnet-v2": {"input": 3.00, "output": 15.00},
    "claude-3-5-sonnet": {"input": 3.00, "output": 15.00},

    # --- AI21 Lab's Models ---
    "jamba-1.5-large": {"input": 2.00, "output": 8.00},
    "jamba-1.5-mini": {"input": 0.20, "output": 0.40},

    # --- Deepseek's Models ---
    "deepseek-r1-0528": {"input": 1.35, "output": 5.40},

    # --- Meta's Llama Models ---
    "llama-3.1-405b": {"input": 5.00, "output": 16.00},
    "llama-3.3-70b": {"input": 0.72, "output": 0.72},
    "llama-4-scout": {"input": 0.25, "output": 0.70},
    "llama-4-maverick": {"input": 0.35, "output": 1.15},

    # --- Mistral AI’s Models ---
    "mistral-small-3.1-25.03": {"input": 0.10, "output": 0.30},
    "mistral-large-24.11": {"input": 2.00, "output": 6.00},
    "mistral-nemo": {"input": 0.15, "output": 0.15},
    "codestral-25.01": {"input": 0.30, "output": 0.90},

    # Default/Unknown Chat Model
    "unknown_chat": {"input": 0.0, "output": 0.0}
}

# --- Embedding Model Pricing Data (in USD per 1 million tokens) ---
# Note: For character-based pricing, we assume 1 token = 4 characters for conversion.
EMBEDDING_PRICING = {
    # OpenAI Embedding Models
    "text-embedding-ada-002": 0.10,  # UPDATED from 0.02 to 0.10
    "text-embedding-3-small": 0.02,
    "text-embedding-3-large": 0.13,

    # Google (Vertex AI) Embedding Models
    "gemini-embedding": 0.15,  # $0.00015 per 1k tokens -> $0.15 per 1M tokens
    "embeddings-for-text": 0.00625,
    # $0.000025 per 1k characters. Assuming 1 char = 0.25 tokens -> 0.000025 / 4 * 1M = 0.00625
    "multimodalembedding": 0.008,
    # $0.0002 / 1k characters -> $0.008 / 1M characters. Assuming 1 char = 0.25 tokens -> 0.0002 / 4 * 1M = 0.008

    # Default/Unknown Embedding Model
    "unknown_embedding": 0.0
}

# --- Tokenizer Cache ---
_tokenizer_cache: Dict[str, tiktoken.Encoding] = {}


def get_tokenizer(model_name: str) -> Optional[tiktoken.Encoding]:
    """Get a tokenizer for a given model, using a cache."""
    # Ensure tiktoken attempts for all relevant models
    if (model_name.startswith("gpt-") or
            model_name.startswith("claude-") or
            model_name.startswith("text-embedding-")):
        tokenizer_name = model_name  # Use model_name as tokenizer_name
        # Map specific model names to their tiktoken encoding names if different
        if "gpt-4o-mini" in model_name:
            tokenizer_name = "gpt-4o-mini"  # Specific mapping
        elif "gpt-4o" in model_name:
            tokenizer_name = "gpt-4o"
        elif "gpt-4-turbo" in model_name:
            tokenizer_name = "gpt-4-turbo"
        elif "gpt-3.5-turbo" in model_name:
            tokenizer_name = "gpt-3.5-turbo"
        elif "claude" in model_name:
            tokenizer_name = "claude-3-haiku"  # General Claude mapping for tiktoken if no specific is found
        elif "text-embedding-ada" in model_name:
            tokenizer_name = "text-embedding-ada-002"
        elif "text-embedding-3" in model_name:
            tokenizer_name = "text-embedding-3-small"  # Using smallest as proxy if large not specified

        if tokenizer_name in _tokenizer_cache:
            return _tokenizer_cache[tokenizer_name]
        try:
            encoding = tiktoken.encoding_for_model(tokenizer_name)
            _tokenizer_cache[tokenizer_name] = encoding
            return encoding
        except KeyError:
            logger.warning(f"No 'tiktoken' tokenizer found for model '{model_name}'. Falling back to approximation.")
            return None

    # For models not supported by tiktoken (e.g., some Google models, custom models)
    logger.warning(f"Model '{model_name}' not supported by tiktoken. Falling back to approximation (len(text) // 4).")
    return None


def count_tokens(text: str, model_name: str) -> int:
    """
    Counts the number of tokens in a text string for a specific model.
    This provides an *estimate* and is best used for non-critical calculations.
    For exact counts, use the token usage data from the API response.
    """
    tokenizer = get_tokenizer(model_name)
    if tokenizer:
        return len(tokenizer.encode(text))

    # Fallback for models without a specific tokenizer or if tokenizer init failed
    return len(text) // 4  # Common approximation: 1 token = ~4 characters


async def calculate_cost(
        llm_model_name: Optional[str] = None,
        prompt_tokens: int = 0,
        completion_tokens: int = 0,
        embedding_model_name: Optional[str] = None,
        embedding_tokens: int = 0,
        gpu_type: Optional[str] = None,
        gpu_provider: Optional[str] = None,
        gpu_hours: float = 0.0
) -> float:
    """
    Calculates the estimated cost of LLM, embedding, and GPU usage based on token counts and GPU hours.
    """
    total_cost = 0.0

    # Calculate LLM cost
    if llm_model_name:
        base_llm_model_name = llm_model_name.lower()
        pricing = MODEL_PRICING.get(base_llm_model_name)
        # Provide reasonable defaults for common aliases
        if pricing is None:
            if base_llm_model_name in ("gpt-4", "gpt-4-0613", "gpt-4-turbo"):
                pricing = MODEL_PRICING.get("gpt-4-0613")
            elif base_llm_model_name in ("gpt-3.5-turbo", "gpt-3.5-turbo-0125"):
                pricing = MODEL_PRICING.get("gpt-3.5-turbo-0125")
            else:
                pricing = MODEL_PRICING["unknown_chat"]

        if "gpt-4o-mini" in base_llm_model_name:
            pricing = MODEL_PRICING.get("gpt-4o-mini-2024-07-18", pricing)
        elif "gpt-4o" in base_llm_model_name:
            pricing = MODEL_PRICING.get("gpt-4o-2024-08-06", pricing)
        elif "gpt-4-turbo" in base_llm_model_name:
            pricing = MODEL_PRICING.get("gpt-4-turbo-2024-04-09", pricing)
        elif "gpt-3.5-turbo" in base_llm_model_name:
            pricing = MODEL_PRICING.get("gpt-3.5-turbo-0125", pricing)
        elif "gemini-1.5-pro" in base_llm_model_name:
            pricing = MODEL_PRICING.get("gemini-2.5-pro", pricing)
        elif "gemini-1.5-flash" in base_llm_model_name:
            pricing = MODEL_PRICING.get("gemini-2.5-flash-ga", pricing)
        elif "claude-3-opus" in base_llm_model_name:
            pricing = MODEL_PRICING.get("claude-3-opus-20240229", pricing)
        elif "claude-3-sonnet" in base_llm_model_name:
            pricing = MODEL_PRICING.get("claude-3-sonnet-20240229", pricing)
        elif "claude-3-haiku" in base_llm_model_name:
            pricing = MODEL_PRICING.get("claude-3-haiku-20240307", pricing)

        if pricing["input"] == 0.0 and pricing["output"] == 0.0:
            logger.info(f"No specific pricing for LLM model '{llm_model_name}'. LLM cost will be reported as 0.")

        input_cost = (prompt_tokens / 1_000_000) * pricing["input"]
        output_cost = (completion_tokens / 1_000_000) * pricing["output"]
        total_cost += input_cost + output_cost

    # NEW: Calculate Embedding cost - Simplified lookup and added debug
    if embedding_model_name and embedding_tokens > 0:
        base_embedding_model_name = embedding_model_name.lower()

        # Direct lookup for exact match first
        embedding_price_per_million = EMBEDDING_PRICING.get(base_embedding_model_name)

        # If no direct match, try common prefixes/aliases, otherwise use unknown_embedding
        if embedding_price_per_million is None:
            if "text-embedding-ada" in base_embedding_model_name:
                embedding_price_per_million = EMBEDDING_PRICING.get("text-embedding-ada-002",
                                                                    EMBEDDING_PRICING["unknown_embedding"])
            elif "text-embedding-3-small" in base_embedding_model_name:
                embedding_price_per_million = EMBEDDING_PRICING.get("text-embedding-3-small",
                                                                    EMBEDDING_PRICING["unknown_embedding"])
            elif "text-embedding-3-large" in base_embedding_model_name:
                embedding_price_per_million = EMBEDDING_PRICING.get("text-embedding-3-large",
                                                                    EMBEDDING_PRICING["unknown_embedding"])
            elif "gemini-embedding" in base_embedding_model_name:
                embedding_price_per_million = EMBEDDING_PRICING.get("gemini-embedding",
                                                                    EMBEDDING_PRICING["unknown_embedding"])
            elif "embeddings-for-text" in base_embedding_model_name:
                embedding_price_per_million = EMBEDDING_PRICING.get("embeddings-for-text",
                                                                    EMBEDDING_PRICING["unknown_embedding"])
            elif "multimodalembedding" in base_embedding_model_name:
                embedding_price_per_million = EMBEDDING_PRICING.get("multimodalembedding",
                                                                    EMBEDDING_PRICING["unknown_embedding"])
            else:
                embedding_price_per_million = EMBEDDING_PRICING[
                    "unknown_embedding"]  # Fallback if no specific alias is found

        logger.debug(
            f"[calculate_cost] Embedding model '{embedding_model_name}' (base: '{base_embedding_model_name}') price per M: ${embedding_price_per_million:.6f}")

        if embedding_price_per_million == 0.0:
            logger.info(
                f"No specific pricing or matched alias found for embedding model '{embedding_model_name}'. Embedding cost will be reported as 0.")

        embedding_cost = (embedding_tokens / 1_000_000) * embedding_price_per_million
        total_cost += embedding_cost

    # NEW: Calculate GPU cost
    if gpu_type and gpu_provider and gpu_hours > 0:
        try:
            # Import here to avoid circular imports
            from app.services.gpu_service_factory import get_gpu_service_instance
            
            gpu_service = get_gpu_service_instance()
            providers = await gpu_service.get_gpu_providers()
            
            cost_per_hour = 1.89  # Default fallback
            for provider in providers:
                if provider["gpu_type"] == gpu_type and provider["name"] == gpu_provider:
                    cost_per_hour = provider["cost_per_hour"]
                    break
            
            gpu_cost = gpu_hours * cost_per_hour
            total_cost += gpu_cost
            
            logger.debug(f"GPU cost: {gpu_type} via {gpu_provider} for {gpu_hours} hours = ${gpu_cost:.2f}")
            
        except Exception as e:
            logger.error(f"Error calculating GPU cost: {e}")
            # Continue without GPU cost if there's an error

    return total_cost