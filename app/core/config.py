# app/core/config.py

import os
import configparser
import logging
from typing import Optional, Any, Dict, List

from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import ChatHuggingFace
from langchain_ollama import ChatOllama
from langchain_core.messages import BaseMessage
from tenacity import retry, wait_exponential, stop_after_attempt, retry_if_exception_type

logger = logging.getLogger(__name__)

RAGNETIC_DIR = ".ragnetic"
CONFIG_FILE = os.path.join(RAGNETIC_DIR, "config.ini")

PROVIDER_MAP = {
    "openai": "OPENAI_API_KEY",
    "anthropic": "ANTHROPIC_API_KEY",
    "google_genai": "GOOGLE_API_KEY",
    "pinecone": "PINECONE_API_KEY",
    "mongodb": "MONGODB_CONN_STRING",
}

llm_model_cache = {}


def get_api_key(provider: str) -> str:
    provider_key_name = PROVIDER_MAP.get(provider.lower())
    if not provider_key_name:
        raise ValueError(f"API provider '{provider}' is not supported.")

    if not os.path.exists(CONFIG_FILE):
        raise FileNotFoundError(f"Config file not found at '{CONFIG_FILE}'. Please run 'ragnetic init'.")

    config = configparser.ConfigParser()
    config.read(CONFIG_FILE)
    api_key = config.get('API_KEYS', provider_key_name, fallback=None)

    if not api_key or api_key == "...":
        raise ValueError(f"API key for '{provider}' not set in {CONFIG_FILE}. Please use 'ragnetic set-api'.")

    return api_key


def get_llm_model(
        model_name: str,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        retries: int = 0,
        timeout: Optional[int] = None,
) -> Any:
    cache_key = f"{model_name}-{temperature}-{max_tokens}-{retries}-{timeout}"
    if cache_key in llm_model_cache:
        return llm_model_cache[cache_key]

    logger.info(f"Initializing LLM model: {model_name} (retries={retries}, timeout={timeout})...")
    common_kwargs = {k: v for k, v in {"temperature": temperature, "request_timeout": timeout}.items() if v is not None}

    try:
        if model_name.startswith("gpt-"):
            llm = ChatOpenAI(model=model_name, api_key=get_api_key("openai"), max_tokens=max_tokens, **common_kwargs)
        elif model_name.startswith("gemini-") or model_name.startswith("models/"):
            llm = ChatGoogleGenerativeAI(model=model_name, google_api_key=get_api_key("google_genai"),
                                         max_output_tokens=max_tokens, **common_kwargs)
        elif model_name.startswith("ollama/"):
            ollama_model_name = model_name.split("/", 1)[1]
            logger.info(f"Using local Ollama model: '{ollama_model_name}'.")
            ollama_kwargs = {k: v for k, v in
                             {"model": ollama_model_name, "temperature": temperature, "num_predict": max_tokens,
                              "timeout": timeout}.items() if v is not None}
            llm = ChatOllama(**ollama_kwargs)
        elif model_name.startswith("hf/"):
            hf_model_name = model_name.split("/", 1)[1]
            logger.info(f"Using local Hugging Face model: '{hf_model_name}'.")
            hf_kwargs = {k: v for k, v in
                         {"model_name": hf_model_name, "temperature": temperature, "max_new_tokens": max_tokens}.items()
                         if v is not None}
            llm = ChatHuggingFace(**hf_kwargs)
        else:
            raise ValueError(f"LLM model '{model_name}' is not supported.")

        if retries > 0:
            original_invoke = llm.invoke

            @retry(wait=wait_exponential(min=2, max=10), stop=stop_after_attempt(retries + 1), reraise=True)
            def _retried_invoke(messages: List[BaseMessage], **kwargs: Any) -> BaseMessage:
                logger.debug(f"LLM invocation for '{model_name}'...")
                return original_invoke(messages, **kwargs)

            llm.invoke = _retried_invoke

        logger.info(f"Successfully initialized LLM model '{model_name}'.")
        llm_model_cache[cache_key] = llm
        return llm
    except Exception as e:
        logger.error(f"Could not initialize LLM model '{model_name}': {e}", exc_info=True)
        raise e