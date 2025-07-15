import logging
import os
import requests
import json
import asyncio
from typing import List, Dict, Any, Optional, Type

from langchain_core.tools import BaseTool, ToolException
from pydantic import Field, BaseModel

from app.schemas.agent import SearchEngineToolInput
from app.core.config import get_api_key

logger = logging.getLogger(__name__)

SEARCH_API_PROVIDER = os.environ.get("SEARCH_API_PROVIDER", "BRAVE")

class SearchTool(BaseTool):
    name: str = "search_engine"
    description: str = (
        "Performs a web search to find information on the internet. "
        "Provide a 'query' (e.g., 'latest AI research'). "
        "Optional: 'num_results' (default 5), 'time_period' (e.g., 'past_day'), 'region' (e.g., 'US'). "
        "Example: search_engine(query='latest developments in quantum computing', num_results=3)"
    )
    args_schema: Type[SearchEngineToolInput] = SearchEngineToolInput

    def _run(self, *args: Any, **kwargs: Any) -> str:
        """
        Synchronous entrypoint for the tool.
        This will run the asynchronous _arun method in the current event loop.
        """
        try:
            # We must run the async _arun method using asyncio.run
            # This is a common pattern for exposing async tools to sync contexts.
            return asyncio.run(self._arun(*args, **kwargs))
        except ToolException as e:
            # Propagate ToolExceptions which LangChain/LangGraph can handle
            raise e
        except Exception as e:
            # Catch any other unexpected errors during sync execution
            logger.error(f"Unexpected error in SearchTool._run: {e}", exc_info=True)
            raise ToolException(f"SearchTool encountered an unexpected synchronous error: {e}")

    async def _arun(self, **kwargs: Any) -> str:
        """
        Asynchronous execution logic for the Search Tool.
        Accepts keyword arguments and parses them into the Pydantic schema.
        """
        # Parse kwargs into SearchEngineToolInput object
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

        search_results = []
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

                logger.info(f"Calling Brave Search API for query: '{query}' with {num_results} results. Params: {params}")
                response = await asyncio.to_thread(requests.get, "https://api.search.brave.com/res/v1/web/search", headers=headers, params=params, timeout=10)
                response.raise_for_status()
                data = response.json()

                if data and data.get("web") and data["web"].get("results"):
                    logger.info(f"Successfully received {len(data['web']['results'])} results from Brave Search.")
                    for idx, result in enumerate(data["web"]["results"]):
                        search_results.append({
                            "title": result.get("title"),
                            "snippet": result.get("description"),
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

        if not search_results:
            logger.info("Search results are empty after tool execution.")
            return "No relevant information found on the web for your query."

        formatted_results = []
        sources_section = "\n**Sources:**\n"
        for idx, result in enumerate(search_results):
            formatted_results.append(
                f"--- Result {idx + 1} ---\n"
                f"Title: {result.get('title', 'N/A')}\n"
                f"URL: {result.get('url', 'N/A')}\n"
                f"Snippet: {result.get('snippet', 'N/A')}\n"
            )
            sources_section += f"- [{result.get('title', f'Link {idx+1}')}]({result.get('url', '#')})\n"

        return (
            f"Retrieved Web Search Results:\n"
            f"{'\\n'.join(formatted_results)}\n\n"
            f"{sources_section}"
        )