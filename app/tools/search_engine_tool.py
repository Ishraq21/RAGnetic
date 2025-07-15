import logging
import os
import requests
import json
import asyncio
from typing import List, Dict, Any, Optional, Type

from langchain_core.tools import BaseTool, ToolException
from pydantic import Field, BaseModel

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import SystemMessage, HumanMessage

from app.schemas.agent import SearchEngineToolInput, AgentConfig
from app.core.config import get_api_key, get_llm_model

logger = logging.getLogger(__name__)

SEARCH_API_PROVIDER = os.environ.get("SEARCH_API_PROVIDER", "BRAVE")
MAX_SNIPPET_LENGTH = 300


class SearchTool(BaseTool):
    name: str = "search_engine"
    description: str = (
        "Performs a web search to find information on the internet. "
        "Provide a 'query' (e.g., 'latest AI research'). "
        "Optional: 'num_results' (default 5), 'time_period' (e.g., 'past_day'), 'region' (e.g., 'US'). "
        "Example: search_engine(query='latest developments in quantum computing', num_results=3)"
    )
    args_schema: Type[SearchEngineToolInput] = SearchEngineToolInput
    return_direct: bool = True

    agent_config: AgentConfig

    def _run(self, *args: Any, **kwargs: Any) -> str:
        """Synchronous entrypoint, runs _arun."""
        try:
            return asyncio.run(self._arun(*args, **kwargs))
        except ToolException as e:
            raise e
        except Exception as e:
            logger.error(f"Unexpected error in SearchTool._run: {e}", exc_info=True)
            raise ToolException(f"SearchTool encountered an unexpected synchronous error: {e}")

    async def _arun(self, *args: Any, **kwargs: Any) -> str:
        """
        Asynchronous execution logic for the Search Tool.
        Performs web search and then synthesizes the results using the agent's configured LLM.
        """
        try:
            tool_input = SearchEngineToolInput(**kwargs)
        except Exception as e:
            logger.error(f"Error parsing SearchEngineTool input arguments: {e}. Kwargs: {kwargs}", exc_info=True)
            raise ToolException(f"Invalid input to SearchTool: {e}. Please check tool arguments.")

        query = tool_input.query
        num_results = tool_input.num_results
        time_period = tool_input.time_period
        region = tool_input.region

        logger.info(f"SearchTool._arun called for query: '{query[:80]}...'")

        search_results_raw = []
        try:
            if SEARCH_API_PROVIDER.upper() == "BRAVE":
                logger.info("Attempting to get Brave Search API key...")
                api_key = get_api_key("brave_search")
                if not api_key:
                    logger.error("BRAVE_SEARCH_API_KEY environment variable not set or key is empty.")
                    raise ToolException("Brave Search API key is not configured. Please run 'ragnetic set-api-key'.")

                logger.info(f"Brave Search API key retrieved (first 5 chars): {api_key[:5]}*****")

                headers = {
                    "Accept": "application/json",
                    "X-Subscription-Token": api_key
                }
                params = {
                    "q": query,
                    "count": num_results,
                    "offset": 0
                }
                if time_period:
                    freshness_map = {
                        'past_day': '24h', 'past_week': '7d', 'past_month': '30d', 'past_year': '365d'
                    }
                    params['freshness'] = freshness_map.get(time_period)
                if region:
                    params['country'] = region

                logger.info(
                    f"Calling Brave Search API for query: '{query}' with {num_results} results. Params: {params}")
                response = await asyncio.to_thread(requests.get, "https://api.search.brave.com/res/v1/web/search",
                                                   headers=headers, params=params, timeout=10)
                response.raise_for_status()
                data = response.json()

                if data and data.get("web") and data["web"].get("results"):
                    logger.info(f"Successfully received {len(data['web']['results'])} results from Brave Search.")
                    for idx, result in enumerate(data["web"]["results"]):
                        snippet = result.get('description', 'N/A')
                        if len(snippet) > MAX_SNIPPET_LENGTH:
                            snippet = snippet[:MAX_SNIPPET_LENGTH] + "..."

                        search_results_raw.append({
                            "title": result.get("title"),
                            "snippet": snippet,
                            "url": result.get("url"),
                            "position": idx + 1
                        })
                else:
                    logger.info(f"Brave Search API returned no web results for query: '{query}'.")

            else:
                logger.error(f"Unsupported SEARCH_API_PROVIDER: {SEARCH_API_PROVIDER}")
                raise ToolException(f"Unsupported search API provider configured: {SEARCH_API_PROVIDER}")

        except ToolException as e:
            logger.error(f"Tool execution error: {e}")
            raise e
        except requests.exceptions.RequestException as e:
            logger.error(f"HTTP/Network error during web search for '{query}': {e}", exc_info=True)
            raise ToolException(f"Failed to perform web search due to network issue: {e}")
        except json.JSONDecodeError as e:
            logger.error(f"Error decoding JSON response from search API for '{query}': {e}", exc_info=True)
            raise ToolException("Failed to parse search API response.")
        except Exception as e:
            logger.error(f"An unexpected error occurred during web search for '{query}': {e}", exc_info=True)
            raise ToolException(f"An unexpected search error occurred: {e}")

        if not search_results_raw:
            logger.info("Search results are empty after tool execution.")
            return "No relevant information found on the web for your query."

        try:
            llm_for_summarization = get_llm_model(
                model_name=self.agent_config.llm_model,
                model_params=self.agent_config.model_params,
                retries=self.agent_config.llm_retries,
                timeout=self.agent_config.llm_timeout
            )

            prompt_messages = [
                SystemMessage(content=SUMMARIZATION_PROMPT_TEMPLATE.format(
                    query=query,
                    # Pass the raw results as a JSON string to the template
                    search_results_context=json.dumps(search_results_raw, indent=2)
                ))
            ]

            logger.info("Invoking LLM within SearchTool to summarize results...")
            llm_response = await asyncio.to_thread(llm_for_summarization.invoke, prompt_messages)

            logger.info("LLM summarization successful within SearchTool.")
            return llm_response.content

        except Exception as e:
            logger.error(f"Error during LLM summarization within SearchTool for query '{query}': {e}", exc_info=True)
            raise ToolException(f"Failed to summarize search results: {e}")


SUMMARIZATION_PROMPT_TEMPLATE = """
You are a highly skilled information synthesizer. Your task is to analyze the provided raw web search results
and present them *in the following order*:

**ORIGINAL USER QUERY:**  
{query}

**RAW WEB SEARCH RESULTS (LIST OF DICTIONARIES):**  
{search_results_context}

1. Summary (a few coherent paragraphs answering the query)  
2. Key Findings (5–7 bullet points)  
3. Sources:  
   For each result:  
   - Authors, Year, *Title*  
   - One-sentence summary  
   - Link  

Use Markdown format. Add inline citations [↩] in your bullets and narrative, then list the matching references under “Sources.”

**Instructions for Formatting:** 

    - Use Markdown for all your responses. 
    - Use headings (`##`, `###`) to structure main topics. 
    - Use bold text (`**text**`) to highlight key terms, figures, or important information. 
    - Use bullet points (`- `) or numbered lists (`1. `) for detailed points or steps.

"""