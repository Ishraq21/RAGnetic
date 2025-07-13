import logging
import hashlib
import time
from typing import List, Dict, Any, Optional

import pandas as pd
from tqdm import tqdm

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.messages import HumanMessage
from langchain_core.documents import Document
from langchain_core.exceptions import OutputParserException

from app.schemas.agent import AgentConfig
from app.tools.retriever_tool import get_retriever_tool
from app.core.config import get_llm_model
from app.core.parsing_utils import normalize_yes_no, safe_json_parse

logger = logging.getLogger(__name__)

# --- Metric Helpers ---------------------------------------------------------

def calculate_retrieval_metrics(
    retrieved_docs: List[Document], ground_truth_id: Optional[str], k: int
) -> Dict[str, float]:
    """
    Compute precision, recall, F1, MRR, and hit@k for retrieval.
    """
    if not retrieved_docs or not ground_truth_id:
        return {m: 0.0 for m in [
            "retrieval_precision", "retrieval_recall", "retrieval_f1",
            "retrieval_mrr", "retrieval_hit_at_k"]}

    ids = [doc.id for doc in retrieved_docs]
    truth = {ground_truth_id}
    tp = len(set(ids) & truth)
    precision = tp / len(ids)
    recall = tp / len(truth)
    f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0

    # Mean Reciprocal Rank
    mrr = 0.0
    for rank, did in enumerate(ids, start=1):
        if did in truth:
            mrr = 1.0 / rank
            break

    hit_at_k = 1.0 if any(d in truth for d in ids[:k]) else 0.0

    return {
        "retrieval_precision": precision,
        "retrieval_recall": recall,
        "retrieval_f1": f1,
        "retrieval_mrr": mrr,
        "retrieval_hit_at_k": hit_at_k,
    }


def calculate_document_uniqueness(docs: List[Document]) -> float:
    """Fraction of unique contents among retrieved docs."""
    if not docs:
        return 0.0
    unique = {d.page_content for d in docs}
    return len(unique) / len(docs)

# --- LLM-as-Judge Prompt ----------------------------------------------------

_JUDGE_CACHE: Dict[str, Dict[str, Any]] = {}
LLM_AS_JUDGE_PROMPT = """
You are an impartial AI evaluator. Assess the quality of the generated answer.
Return exactly one JSON object with keys:
  - faithfulness: "Yes" or "No"
  - answer_relevance: "Yes" or "No"
  - conciseness_score: int (1-5)
  - coherence_score: int (1-5)
  - reasoning: brief explanation

User Question: {question}
Retrieved Context:
{context}
Generated Answer:
{answer}

Respond with raw JSON only.
"""


def run_benchmark(agent_config: AgentConfig, test_set: List[Dict[str, Any]]) -> pd.DataFrame:
    """
    Execute retrieval, generation, and LLM-based evaluation over a test set.
    Returns a DataFrame of metrics and evaluations.
    """
    logger.info("Starting benchmark for '%s'", agent_config.name)

    # Initialize components
    retriever_tool = get_retriever_tool(agent_config)
    retriever = retriever_tool.func.__closure__[0].cell_contents
    agent_llm = get_llm_model(agent_config.llm_model)
    judge_llm = get_llm_model(agent_config.evaluation_llm_model or agent_config.llm_model, temperature=0.0)
    parser = JsonOutputParser()
    prompt = ChatPromptTemplate.from_template(LLM_AS_JUDGE_PROMPT)
    eval_chain = prompt | judge_llm | parser

    records: List[Dict[str, Any]] = []
    for item in tqdm(test_set, desc="Benchmark"):
        q = item["question"]
        gt_id = item.get("retrieval_ground_truth_chunk_id")

        # Retrieval
        t0 = time.perf_counter()
        docs = retriever.invoke(q)
        retrieval_time = time.perf_counter() - t0

        # Generation
        context_str = "\n\n".join(d.page_content for d in docs)
        answer = "N/A"
        gen_time = 0.0
        if item.get("type") != "Out-of-scope":
            t1 = time.perf_counter()
            resp = agent_llm.invoke([HumanMessage(content=f"Context:\n{context_str}\n\nQuestion: {q}\nAnswer:")])
            gen_time = time.perf_counter() - t1
            answer = resp.content

        # Ground-truth recall
        recalled = any(d.id == gt_id for d in docs)

        # Evaluation
        cache_key = hashlib.sha256(f"{q}|{context_str}|{answer}".encode()).hexdigest()
        if cache_key in _JUDGE_CACHE:
            ev = _JUDGE_CACHE[cache_key]
        else:
            t2 = time.perf_counter()
            try:
                out = eval_chain.invoke({"question": q, "context": context_str, "answer": answer})
                ev = {
                    "faithfulness": normalize_yes_no(out.get("faithfulness", "No")),
                    "answer_relevance": normalize_yes_no(out.get("answer_relevance", "No")),
                    "conciseness_score": int(out.get("conciseness_score", -1)),
                    "coherence_score": int(out.get("coherence_score", -1)),
                    "reasoning": out.get("reasoning", ""),
                }
            except Exception as e:
                raw = getattr(e, 'llm_output', str(e))
                logger.warning("Judge parse failed, fallback regex: %s", raw)
                fb = safe_json_parse(raw) or {}
                ev = {
                    "faithfulness": normalize_yes_no(fb.get("faithfulness", "No")),
                    "answer_relevance": normalize_yes_no(fb.get("answer_relevance", "No")),
                    "conciseness_score": int(fb.get("conciseness_score", -1)),
                    "coherence_score": int(fb.get("coherence_score", -1)),
                    "reasoning": fb.get("reasoning", ""),
                }
            judge_time = time.perf_counter() - t2
            ev["llm_judge_time_s"] = judge_time
            _JUDGE_CACHE[cache_key] = ev

        # Combine metrics
        retrieval_metrics = calculate_retrieval_metrics(docs, gt_id, agent_config.vector_store.hit_rate_k_value)
        uniqueness = calculate_document_uniqueness(docs)

        rec = {
            "question": q,
            "generated_answer": answer,
            "key_fact_recalled": recalled,
            "contextual_noise": len(docs),
            "doc_uniqueness": uniqueness,
            "retrieval_time_s": retrieval_time,
            "generation_time_s": gen_time,
        }
        rec.update(retrieval_metrics)
        rec.update(ev)
        records.append(rec)

    return pd.DataFrame(records)
