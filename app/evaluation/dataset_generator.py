import logging
import random
from typing import List, Dict, Any

from tqdm import tqdm
from langchain_core.documents import Document as LangChainDocument
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser

from app.core.config import get_llm_model
from app.schemas.agent import AgentConfig
from app.pipelines.embed import _get_chunks_from_documents, load_documents_from_source

logger = logging.getLogger(__name__)

QA_PROMPT = """
You are an expert data creator for evaluating Retrieval Augmented Generation (RAG) systems.
Generate exactly a single JSON array of 1–2 question/answer objects based *only* on the context.

Each object must have keys:
  • question (string)
  • answer   (string)
  • type     (Factoid|Analytical|Out-of-scope)

For Out-of-scope:
  answer must be "UNANSWERABLE: The answer is not contained in the provided text."

Do **NOT** include any other text.

**Context:**
---
{context}
---
"""


class DatasetGenerator:
    """
    Generates a question-answer dataset from an agent's configuration.
    Loads documents, chunks them, and asks an LLM to create QA pairs
    from a (shuffled) sample of those chunks.
    """

    def __init__(self, cfg: AgentConfig):
        self.cfg = cfg
        self.llm = get_llm_model(cfg.llm_model, temperature=0.2)
        logger.info(f"Generating test set for '{cfg.name}' with LLM={cfg.llm_model}")

        self.prompt = ChatPromptTemplate.from_template(QA_PROMPT)
        self.parser = JsonOutputParser()

    def _qa_for_chunk(self, chunk: LangChainDocument) -> List[Dict[str, Any]]:
        chain = self.prompt | self.llm | self.parser
        try:
            qa_list = chain.invoke({"context": chunk.page_content})
            if not isinstance(qa_list, list):
                return []

            md = chunk.metadata or {}
            cid = md.get("chunk_id")
            if cid is None:
                logger.warning(f"Chunk missing chunk_id; skipping. metadata={md}")
                return []

            out: List[Dict[str, Any]] = []
            for qa in qa_list:
                q_text = (qa.get("question") or "").strip()
                a_text = (qa.get("answer") or "").strip()
                q_type = (qa.get("type") or "").strip()

                if not q_text:
                    continue

                # Normalize/guard the type
                allowed = {"Factoid", "Analytical", "Out-of-scope"}
                if q_type not in allowed:
                    q_type = "Out-of-scope" if a_text.startswith("UNANSWERABLE:") else "Factoid"

                out.append({
                    "question": q_text,
                    "answer": a_text,
                    "type": q_type,
                    "retrieval_ground_truth_chunk_id": str(cid),
                    "source_doc_name": md.get("doc_name"),
                    "source_chunk_index": md.get("chunk_index"),
                    "original_doc_id": md.get("original_doc_id"),
                    "source_text": chunk.page_content,  # handy when inspecting failures
                })
            return out
        except Exception as e:
            logger.error(f"Chunk {getattr(chunk, 'id', 'unknown')} QA generation failed: {e}", exc_info=True)
            return []

    async def generate(self, n: int) -> List[Dict[str, Any]]:
        """
        Asynchronously generate up to n QA items across chunks.
        """
        docs: List[LangChainDocument] = []

        # Load all sources
        for source in self.cfg.sources:
            loaded_docs = await load_documents_from_source(
                source,
                self.cfg,
                self.cfg.reproducible_ids
            )
            docs.extend(loaded_docs)

        # Sort, chunk, shuffle, and build QA
        docs.sort(key=lambda d: d.metadata.get("original_doc_id", ""))
        chunks = _get_chunks_from_documents(
            docs,
            self.cfg.chunking,
            self.cfg.embedding_model,
            self.cfg.reproducible_ids
        )
        logger.info(f"→ {len(docs)} docs → {len(chunks)} chunks")

        random.shuffle(chunks)
        all_qa_pairs: List[Dict[str, Any]] = []
        for chunk in tqdm(chunks, desc="QA chunks"):
            if len(all_qa_pairs) >= n:
                break
            all_qa_pairs.extend(self._qa_for_chunk(chunk))

        return all_qa_pairs[:n]


async def generate_test_set(agent_cfg: AgentConfig, num_questions: int) -> List[Dict[str, Any]]:
    """
    High-level helper: generate a test set for a given agent configuration.
    """
    generator = DatasetGenerator(agent_cfg)
    return await generator.generate(num_questions)
