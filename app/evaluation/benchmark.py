import logging
import os
import re
from typing import List, Dict, Any, Optional
import asyncio, time as _time, math, json, hashlib, uuid, random
from datetime import datetime
from sqlalchemy import select, update
import pandas as pd
import concurrent.futures
import traceback  # <-- for robust failure reporting

from langchain_core.documents import Document

from app.schemas.agent import AgentConfig
from app.core.config import get_llm_model, get_path_settings

# Consistent module-level logger
logger = logging.getLogger("ragnetic.benchmark")


def _token_count_safe(text: str, model_name: Optional[str], fallback_chars_per_token: int = 4) -> int:
    from app.core.cost_calculator import count_tokens
    try:
        return count_tokens(text or "", model_name)
    except Exception:
        # crude but safe fallback
        return max(1, len(text or "") // fallback_chars_per_token)


def _truncate_docs_by_tokens(docs: List[Document], model_name: Optional[str], budget_tokens: int) -> List[Document]:
    """
    Keep docs (in order) until we hit the token budget. Prevents over-long prompts.
    """
    used = 0
    kept: List[Document] = []
    for d in docs:
        t = _token_count_safe(getattr(d, "page_content", "") or "", model_name)
        if used + t > budget_tokens:
            break
        kept.append(d)
        used += t
    return kept


def _arun(coro):
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)
    else:
        with concurrent.futures.ThreadPoolExecutor(1) as ex:
            fut = ex.submit(asyncio.run, coro)
            return fut.result()


def calculate_document_uniqueness(docs: List[Document]) -> float:
    """Fraction of unique contents among retrieved docs."""
    if not docs:
        return 0.0
    unique = {d.page_content for d in docs}
    return len(unique) / len(docs)


def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "")).strip()


def _hash_text(s: str) -> str:
    return hashlib.sha1(_norm(s).encode("utf-8")).hexdigest()


def _first_match_rank(retrieved_docs: List[Document],
                      gt_id: Optional[str],
                      gt_text: Optional[str]) -> Optional[int]:
    """
    Return the rank (0-based) where we first match either by:
      1) metadata.chunk_id == ground-truth-id
      2) content hash equals ground-truth source_text hash
    If neither matches, return None.
    """
    gt_id = str(gt_id) if gt_id is not None else None
    gt_hash = _hash_text(gt_text) if gt_text else None
    for i, d in enumerate(retrieved_docs):
        md = getattr(d, "metadata", {}) or {}
        rid = md.get("chunk_id")
        if rid is not None and gt_id is not None and str(rid) == gt_id:
            return i
        if gt_hash and _hash_text(getattr(d, "page_content", "")) == gt_hash:
            return i
    return None


def _rank_to_metrics(rank: Optional[int], ks=(1, 3, 5, 10)) -> Dict[str, float]:
    """
    Convert a rank into MRR / nDCG@10 / hit@K. If rank is None, return zeros.
    """
    if rank is None:
        return {"mrr": 0.0, "ndcg@10": 0.0, **{f"hit@{k}": 0.0 for k in ks}}
    mrr = 1.0 / (rank + 1)
    ndcg = 1.0 / (math.log2(rank + 2)) if rank < 10 else 0.0  # IDCG = 1
    hits = {f"hit@{k}": 1.0 if rank < k else 0.0 for k in ks}
    return {"mrr": mrr, "ndcg@10": ndcg, **hits}


def _retrieval_metrics_strict(ids: List[Any], truth_id: Optional[Any], ks=(1, 3, 5, 10)) -> Dict[str, float]:
    """
    Legacy strict metrics using only the ground-truth chunk_id (no content-hash fallback).
    Kept for debugging/analysis; not used as the main metric anymore.
    """
    ids = [str(i) for i in ids if i is not None]
    truth = str(truth_id) if truth_id is not None else None

    metrics: Dict[str, float] = {}
    if not ids or not truth:
        for k in ks:
            metrics[f"hit@{k}"] = 0.0
        return {"mrr": 0.0, "ndcg@10": 0.0, **metrics}

    mrr = 1.0 / (ids.index(truth) + 1) if truth in ids else 0.0
    ndcg = 0.0
    if truth in ids[:10]:
        r = ids.index(truth) + 1
        ndcg = 1.0 / (math.log2(r + 1))  # IDCG=1
    for k in ks:
        metrics[f"hit@{k}"] = 1.0 if truth in ids[:k] else 0.0
    return {"mrr": mrr, "ndcg@10": ndcg, **metrics}


def run_benchmark(
    agent_config: AgentConfig,
    test_set: List[Dict[str, Any]],
    *,
    run_id: Optional[str] = None,
    dataset_id: Optional[str] = None,
    sync_engine=None,
    export_csv_path: Optional[str] = None,
) -> pd.DataFrame:
    """
    Robust, production-grade runner:
    - Uses official retriever tool
    - Persists to DB: benchmark_runs / benchmark_items
    - Provenance: dataset_id, prompt_hash, agent_config_hash
    - Retries + multi-K retrieval metrics (Hit@{1,3,5,10}, MRR, nDCG@10)
    - Cost/tokens for agent/judge/embeddings
    - Resumable via run_id
    """
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_core.output_parsers import JsonOutputParser
    from langchain_core.messages import HumanMessage
    from langchain_core.exceptions import OutputParserException

    from app.db import get_sync_db_engine
    from app.db.models import benchmark_runs_table, benchmark_items_table
    from app.tools.retriever_tool import get_retriever_tool
    from app.core.cost_calculator import calculate_cost, count_tokens

    EVAL_PROMPT = """You are a strict evaluator.

    CONTEXT:
    {context}

    QUESTION:
    {question}

    ANSWER:
    {answer}

    Return ONLY a JSON object with keys: faithfulness, relevance, conciseness, coherence (each in [0,1]), and notes (string)."""

    def _sha(obj: Any) -> str:
        return hashlib.sha256(json.dumps(obj, sort_keys=True, default=str).encode("utf-8")).hexdigest()

    def _jsonable(x):
        # Pydantic v2 objects
        if hasattr(x, "model_dump"):
            try:
                return x.model_dump()
            except Exception:
                pass
        # dataclasses
        try:
            import dataclasses
            if dataclasses.is_dataclass(x):
                return dataclasses.asdict(x)
        except Exception:
            pass
        # containers
        if isinstance(x, dict):
            return {k: _jsonable(v) for k, v in x.items()}
        if isinstance(x, (list, tuple, set)):
            return [_jsonable(v) for v in x]
        # primitives or fallback
        return x if isinstance(x, (str, int, float, bool, type(None))) else str(x)

    cfg_snapshot_raw = {
        "agent_name": agent_config.name,
        "llm_model": getattr(agent_config, "llm_model", None),
        "evaluation_llm_model": getattr(agent_config, "evaluation_llm_model", None) or getattr(agent_config, "llm_model", None),
        "embedding_model": getattr(agent_config, "embedding_model", None),
        "vector_store": getattr(agent_config, "vector_store", None),
        "chunking": getattr(agent_config, "chunking", None),
        "tools": getattr(agent_config, "tools", []),
        "benchmark": getattr(agent_config, "benchmark", None),
    }

    cfg_snapshot = _jsonable(cfg_snapshot_raw)

    agent_config_hash = _sha(cfg_snapshot)  # hash the JSON-safe version
    prompt_hash = _sha({"judge_prompt": EVAL_PROMPT, "schema": "ragnetic.eval.v1"})

    run_id = run_id or f"bench_{uuid.uuid4().hex[:12]}"

    paths = get_path_settings()
    bench_dir = paths["BENCHMARK_DIR"]
    os.makedirs(bench_dir, exist_ok=True)

    def _safe_filename(s: str) -> str:
        return re.sub(r'[^A-Za-z0-9_.-]+', '_', s).strip('._')

    agent_slug = _safe_filename(getattr(agent_config, "name", "agent"))
    default_csv_path = os.path.join(bench_dir, f"benchmark_{agent_slug}_{run_id}.csv")
    export_csv_path = export_csv_path or default_csv_path

    # Models
    agent_llm = get_llm_model(
        getattr(agent_config, "llm_model", None),
        retries=getattr(agent_config, "llm_retries", 2),
        timeout=getattr(agent_config, "llm_timeout", 60)
    )
    judge_model_name = getattr(agent_config, "evaluation_llm_model", None) or getattr(agent_config, "llm_model", None)
    judge_llm = get_llm_model(judge_model_name, temperature=0.0, retries=2, timeout=60)

    eval_prompt = ChatPromptTemplate.from_template(EVAL_PROMPT)
    parser = JsonOutputParser()
    eval_chain = eval_prompt | judge_llm

    # Official Tool (sync ctor; async invoke)
    retriever_tool = _arun(get_retriever_tool(agent_config, user_id=-1, thread_id=run_id))
    if asyncio.iscoroutine(retriever_tool):
        retriever_tool = _arun(retriever_tool)

    engine = sync_engine or get_sync_db_engine()

    # Upsert run header
    with engine.begin() as conn:
        existing = conn.execute(
            select(benchmark_runs_table.c.run_id, benchmark_runs_table.c.completed_items)
            .where(benchmark_runs_table.c.run_id == run_id)
        ).mappings().first()
        if not existing:
            conn.execute(benchmark_runs_table.insert().values(
                run_id=run_id,
                agent_name=agent_config.name,
                dataset_id=dataset_id,
                prompt_hash=prompt_hash,
                agent_config_hash=agent_config_hash,
                judge_model=judge_model_name,
                config_snapshot=cfg_snapshot,
                total_items=len(test_set),
                completed_items=0,
                status="running",
                summary_metrics=None,
                error=None,
            ))
        else:
            logger.info(f"Resuming benchmark run_id={run_id} (completed_items={existing['completed_items']})")

    def _retry(call, attempts=4, base_delay=0.7, max_delay=60.0, exc_types=(Exception,)):
        """
        Exponential backoff with jitter. Enhanced rate limiting for OpenAI TPM limits.
        """
        last = None
        for i in range(attempts):
            try:
                return call()
            except exc_types as e:
                last = e
                delay = min(max_delay, base_delay * (2 ** i))
                
                # Enhanced rate limit handling for OpenAI TPM limits
                msg = str(getattr(e, "message", "")) or str(e)
                if "429" in msg or "RateLimit" in msg or "Too Many Requests" in msg:
                    # Try to extract wait time from error message
                    import re
                    wait_match = re.search(r'Please try again in ([\d.]+)s', msg)
                    if wait_match:
                        extracted_wait = float(wait_match.group(1))
                        delay = max(delay, extracted_wait + 1.0)  # Add 1s buffer
                        logger.warning(f"Rate limit detected. Waiting {delay:.2f}s as suggested by API")
                    else:
                        # For TPM limits, use longer delays
                        delay = max(delay, 10.0)  # Minimum 10s for TPM limits
                        logger.warning(f"Rate limit detected (likely TPM). Using extended delay: {delay:.2f}s")
                
                # Add jitter to prevent thundering herd
                delay *= (0.8 + 0.4 * random.random())
                logger.warning(f"Retry {i + 1}/{attempts} after error: {e}. Sleeping {delay:.2f}s")
                _time.sleep(delay)
        raise last

    records: List[Dict[str, Any]] = []

    # Figure out what we've already done for this run_id
    with engine.begin() as conn:
        already = conn.execute(
            select(benchmark_items_table.c.item_index).where(benchmark_items_table.c.run_id == run_id)
        ).mappings().all()
    done_indices = {row["item_index"] for row in already}
    if done_indices:
        logger.info(
            f"[bench] Resuming run_id={run_id}. Will skip {len(done_indices)} already-completed items: {sorted(done_indices)}")

    for idx, item in enumerate(test_set):
        if idx in done_indices:
            continue
        start = _time.perf_counter()

        q = item.get("question") or item.get("query") or item.get("q")
        if not q:
            continue

        gt_id = item.get("retrieval_ground_truth_chunk_id")
        gt_text = item.get("source_text")  # may be None for some samples

        # --- Retrieval ---
        t0 = _time.perf_counter()
        docs: List[Document] = _retry(lambda: _arun(retriever_tool.ainvoke(q)))
        retrieval_time = _time.perf_counter() - t0

        bm = getattr(agent_config, "benchmark", None)
        max_ctx_docs = bm.max_context_docs if bm else 20
        context_docs = (docs or [])[:max_ctx_docs]

        if docs and len(docs) > max_ctx_docs:
            logger.info(f"[bench] doc-cap: capped to {max_ctx_docs} of {len(docs)} retrieved docs")

        if bm and bm.enable_doc_truncation:
            total_ctx = bm.context_window_tokens
            ratio = bm.context_budget_ratio
            reserve = bm.answer_reserve_tokens

            scaffolding = "Context:\n{ctx}\n\nQuestion: {q}\nAnswer:"
            overhead_tokens = (
                _token_count_safe(scaffolding.replace("{ctx}", ""), getattr(agent_config, "llm_model", None))
                + _token_count_safe(q, getattr(agent_config, "llm_model", None))
            )
            budget = max(0, int(total_ctx * ratio) - reserve - overhead_tokens)

            if budget <= 0 and context_docs:
                logger.info(
                    f"[bench] truncation: budget<=0; forcing top-1 "
                    f"(budget≈{budget}, reserve={reserve}, overhead≈{overhead_tokens})"
                )
                context_docs = context_docs[:1]
            else:
                before_n = len(context_docs)
                context_docs = _truncate_docs_by_tokens(
                    context_docs,
                    getattr(agent_config, "llm_model", None),
                    budget
                )
                after_n = len(context_docs)
                logger.info(
                    f"[bench] truncation: kept {after_n}/{before_n} docs "
                    f"(budget≈{budget} tokens)"
                )

        context_uniqueness = calculate_document_uniqueness(context_docs)
        retrieved_ids: List[str] = []
        for d in context_docs:
            md = getattr(d, "metadata", {}) or {}
            rid = md.get("chunk_id") or getattr(d, "id", None)
            if rid is not None:
                retrieved_ids.append(str(rid))
        context_str = "\n\n".join(getattr(d, "page_content", "") for d in context_docs)

        # Rank via (chunk_id match) OR (content-hash match), then metrics from that rank
        rank = _first_match_rank(context_docs, gt_id, gt_text)
        rmetrics = _rank_to_metrics(rank)

        # (Optional) also compute strict ID-only metrics for debugging
        strict_metrics = _retrieval_metrics_strict([rid for rid in retrieved_ids if rid], gt_id)

        logger.info(f"[bench] idx={idx} GT={gt_id} rank={rank} retrieved_top={retrieved_ids[:10]}")

        emb_model = getattr(agent_config, "embedding_model", None)
        embedding_tokens_query = 0
        embedding_cost_query = 0.0
        if emb_model:
            embedding_tokens_query = count_tokens(q, emb_model)
            embedding_cost_query = calculate_cost(
                embedding_model_name=emb_model,
                embedding_tokens=embedding_tokens_query
            )


        t1 = _time.perf_counter()
        resp = _retry(lambda: agent_llm.invoke([HumanMessage(content=f"Context:\n{context_str}\n\nQuestion: {q}\nAnswer:")]))
        gen_time = _time.perf_counter() - t1
        answer = getattr(resp, "content", str(resp))

        agent_token_usage: Dict[str, int] = {}
        if hasattr(resp, "usage_metadata") and resp.usage_metadata:
            agent_token_usage = {
                "prompt_tokens": resp.usage_metadata.get("input_tokens", 0),
                "completion_tokens": resp.usage_metadata.get("output_tokens", 0),
                "total_tokens": resp.usage_metadata.get("total_tokens", 0),
            }
        elif hasattr(resp, "response_metadata"):
            agent_token_usage = resp.response_metadata.get("token_usage", {}) or {}

        # --- Judge ---
        t2 = _time.perf_counter()
        judge_raw = _retry(lambda: eval_chain.invoke({"question": q, "context": context_str, "answer": answer}))
        judge_time = _time.perf_counter() - t2

        judge_answer = getattr(judge_raw, "content", judge_raw)
        try:
            judge_obj = parser.parse(judge_answer) if isinstance(judge_answer, str) else judge_answer
        except OutputParserException:
            judge_obj = {
                "faithfulness": 0.0,
                "relevance": 0.0,
                "conciseness": 0.0,
                "coherence": 0.0,
                "notes": "Judge JSON parse failed."
            }

        # Judge token usage
        judge_token_usage: Dict[str, int] = {}
        if hasattr(judge_raw, "usage_metadata") and judge_raw.usage_metadata:
            judge_token_usage = {
                "prompt_tokens": judge_raw.usage_metadata.get("input_tokens", 0),
                "completion_tokens": judge_raw.usage_metadata.get("output_tokens", 0),
                "total_tokens": judge_raw.usage_metadata.get("total_tokens", 0),
            }
        elif hasattr(judge_raw, "response_metadata"):
            judge_token_usage = judge_raw.response_metadata.get("token_usage", {}) or {}

        judge_llm_cost = calculate_cost(
            llm_model_name=judge_model_name,
            prompt_tokens=judge_token_usage.get("prompt_tokens", 0),
            completion_tokens=judge_token_usage.get("completion_tokens", 0),
        )

        agent_llm_cost = calculate_cost(
            llm_model_name=getattr(agent_config, "llm_model", None),
            prompt_tokens=agent_token_usage.get("prompt_tokens", 0),
            completion_tokens=agent_token_usage.get("completion_tokens", 0),
        )

        total_cost_usd = float(agent_llm_cost + judge_llm_cost + float(embedding_cost_query or 0.0))
        total_duration = _time.perf_counter() - start

        rec: Dict[str, Any] = {
            "request_id": f"{run_id}:{idx}",
            "run_id": run_id,
            "item_index": idx,
            "agent_name": agent_config.name,
            "dataset_id": dataset_id,
            "question": q,
            "ground_truth_chunk_id": gt_id,
            "retrieved_ids": retrieved_ids,
            "context_size": len(context_docs),
            "context_uniqueness": context_uniqueness,
            "generated_answer": answer,
            "judge": judge_obj,
            "retrieval": rmetrics,
            "retrieval_strict": strict_metrics,
            "durations": {
                "retrieval_s": retrieval_time,
                "generation_s": gen_time,
                "judge_s": judge_time,
                "total_s": total_duration
            },
            "token_usage": {
                "agent": agent_token_usage,
                "judge": judge_token_usage,
                "embedding_tokens_query": embedding_tokens_query
            },
            "costs": {
                "agent_llm_usd": agent_llm_cost,
                "judge_llm_usd": judge_llm_cost,
                "embedding_query_usd": embedding_cost_query,
                "total_usd": total_cost_usd
            },
            "agent_llm_model": getattr(agent_config, "llm_model", None),
            "agent_embedding_model": getattr(agent_config, "embedding_model", None),
            "chunking_mode": getattr(getattr(agent_config, "chunking", None), "mode", None),
            "chunk_size": getattr(getattr(agent_config, "chunking", None), "chunk_size", None),
            "chunk_overlap": getattr(getattr(agent_config, "chunking", None), "chunk_overlap", None),
        }
        records.append(rec)

        # Persist per-item
        with engine.begin() as conn:
            conn.execute(benchmark_items_table.insert().values(
                run_id=run_id,
                item_index=idx,
                question=q,
                ground_truth_chunk_id=gt_id,
                retrieved_ids=retrieved_ids,
                retrieval_metrics=rmetrics,
                context_size=len(context_docs),
                answer=answer,
                judge_scores=judge_obj,
                token_usage={
                    "agent": agent_token_usage,
                    "judge": judge_token_usage,
                    "embedding_tokens_query": embedding_tokens_query
                },
                costs={
                    "agent_llm_usd": agent_llm_cost,
                    "judge_llm_usd": judge_llm_cost,
                    "embedding_query_usd": embedding_cost_query,
                    "total_usd": total_cost_usd
                },
                durations={
                    "retrieval_s": retrieval_time,
                    "generation_s": gen_time,
                    "judge_s": judge_time,
                    "total_s": total_duration
                },
                citations=None,
                created_at=datetime.utcnow(),
            ))
            conn.execute(
                update(benchmark_runs_table)
                .where(benchmark_runs_table.c.run_id == run_id)
                .values(completed_items=benchmark_runs_table.c.completed_items + 1)
            )
            
            # Add delay between requests to prevent rate limiting
            # For TPM limits, we need to pace requests to stay under 30K tokens/minute
            if idx < len(test_set) - 1:  # Don't delay after the last item
                bm = getattr(agent_config, "benchmark", None)
                delay_seconds = bm.request_delay_seconds if bm else 2.0
                logger.info(f"[bench] Completed item {idx + 1}/{len(test_set)}. Waiting {delay_seconds}s before next request to avoid rate limits...")
                _time.sleep(delay_seconds)


    status = "completed"
    error_msg = None
    summary: Dict[str, float] = {}
    try:
        df = pd.DataFrame.from_records(records)
        if not df.empty:
            def _judgemean(field: str) -> float:
                try:
                    return float(df["judge"].apply(lambda j: (j or {}).get(field, 0.0)).mean())
                except Exception:
                    return 0.0

            summary = {
                "avg_total_cost_usd": float(df["costs"].apply(lambda c: c["total_usd"]).mean()),
                "avg_retrieval_hit@5": float(df["retrieval"].apply(lambda r: r.get("hit@5", 0.0)).mean()),
                "avg_mrr": float(df["retrieval"].apply(lambda r: r.get("mrr", 0.0)).mean()),
                "avg_ndcg@10": float(df["retrieval"].apply(lambda r: r.get("ndcg@10", 0.0)).mean()),
                "avg_generation_s": float(df["durations"].apply(lambda d: d["generation_s"]).mean()),
                "avg_retrieval_s": float(df["durations"].apply(lambda d: d["retrieval_s"]).mean()),
                "avg_faithfulness": _judgemean("faithfulness"),
                "avg_relevance": _judgemean("relevance"),
                "avg_conciseness": _judgemean("conciseness"),
                "avg_coherence": _judgemean("coherence"),
            }
            if "context_uniqueness" in df.columns:
                summary["avg_context_uniqueness"] = float(df["context_uniqueness"].mean())

        # Export CSV artifact
        try:
            df.to_csv(export_csv_path, index=False)
            logger.info(f"Benchmark CSV exported to: {export_csv_path}")
        except Exception as e:
            logger.warning(f"Failed to export CSV to {export_csv_path}: {e}")

        logger.info(f"Benchmark complete: run_id={run_id} items={len(records)}/{len(test_set)}")
        return df

    except KeyboardInterrupt:
        status = "aborted"
        error_msg = "Aborted by user"
        raise
    except Exception:
        status = "failed"
        error_msg = traceback.format_exc()
        raise
    finally:
        with engine.begin() as conn:
            conn.execute(
                update(benchmark_runs_table)
                .where(benchmark_runs_table.c.run_id == run_id)
                .values(
                    status=status,
                    ended_at=datetime.utcnow(),
                    summary_metrics=summary if status == "completed" else None,
                    error=error_msg,
                )
            )
