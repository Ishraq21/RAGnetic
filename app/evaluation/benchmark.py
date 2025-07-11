import logging
from typing import List, Dict, Any

import pandas as pd
from tqdm import tqdm

from app.schemas.agent import AgentConfig
from app.tools.retriever_tool import get_retriever_tool

logger = logging.getLogger(__name__)


def run_benchmark(agent_config: AgentConfig, test_set: List[Dict[str, Any]]) -> pd.DataFrame:
    """
    Runs a benchmark on an agent's retriever using a ground truth test set.

    This evaluation is LLM-free and focuses on objective retrieval metrics.
    """
    logger.info(f"--- Starting Benchmark for Agent: '{agent_config.name}' ---")

    try:
        retriever_tool = get_retriever_tool(agent_config)
        # The actual retriever is a function within the tool's 'func' attribute's closure
        retriever = retriever_tool.func.__closure__[0].cell_contents
    except Exception as e:
        logger.error(f"Could not initialize retriever for evaluation: {e}")
        return pd.DataFrame()

    benchmark_results = []
    for item in tqdm(test_set, desc="Running benchmark"):
        question = item.get("question")
        key_fact = item.get("answer")  # The "ground truth" answer/fact

        if not question or not key_fact:
            continue

        try:
            # 1. Get the retrieved documents for the question
            retrieved_docs = retriever.invoke(question)
            retrieved_context = "\n".join([doc.page_content for doc in retrieved_docs])

            # 2. Calculate Key Fact Recall
            # This checks if the essential answer is present in the retrieved context.
            recall_score = 1 if key_fact.lower() in retrieved_context.lower() else 0

            # 3. Calculate Contextual Noise
            # A simple metric for noise is the number of documents retrieved.
            # A good retriever finds the fact in a small number of highly relevant documents.
            noise_score = len(retrieved_docs)

            benchmark_results.append({
                "question": question,
                "key_fact_recalled": bool(recall_score),
                "contextual_noise": noise_score
            })

        except Exception as e:
            logger.error(f"Failed to benchmark question '{question}': {e}")
            benchmark_results.append({
                "question": question,
                "key_fact_recalled": False,
                "contextual_noise": -1,  # Indicates an error
            })

    return pd.DataFrame(benchmark_results)