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
    Handles the generation of a question-answer dataset from a given agent's configuration.
    It loads documents, chunks them, and then uses an LLM to generate Q&A pairs
    from a random sample of those chunks.
    """

    def __init__(self, cfg: AgentConfig):
        """
        Initializes the dataset generator.
        Args:
            cfg: The agent configuration object.
        """
        self.cfg = cfg
        self.llm = get_llm_model(cfg.llm_model, temperature=0.2)
        logger.info(f"Generating test set for '{cfg.name}' with LLM={cfg.llm_model}")

        self.prompt = ChatPromptTemplate.from_template(QA_PROMPT)
        self.parser = JsonOutputParser()

    def _qa_for_chunk(self, chunk: LangChainDocument) -> List[Dict[str, Any]]:
        """
        Generates a list of question-answer pairs for a single document chunk.
        This remains synchronous as it calls synchronous LLM inference.
        """
        chain = self.prompt | self.llm | self.parser
        try:
            qa_list = chain.invoke({"context": chunk.page_content})
            if not isinstance(qa_list, list):
                return []
            for qa in qa_list:
                qa["retrieval_ground_truth_chunk_id"] = chunk.id
                qa["source_text"] = chunk.page_content
            return qa_list
        except Exception as e:
            logger.error(f"Chunk {chunk.id} QA generation failed: {e}", exc_info=True)
            return []

    async def generate(self, n: int) -> List[Dict[str, Any]]:
        """
        Generates the full dataset of n questions asynchronously.
        """
        docs: List[LangChainDocument] = []

        # --- correctly supply (source, agent_config, reproducible_ids) ---
        for source in self.cfg.sources:
            loaded_docs = await load_documents_from_source(
                source,
                self.cfg,
                self.cfg.reproducible_ids
            )
            docs.extend(loaded_docs)

        # sort, chunk, sample, and then run QA on each chunk
        docs.sort(key=lambda d: d.metadata.get("original_doc_id", ""))
        chunks = _get_chunks_from_documents(
            docs,
            self.cfg.chunking,
            self.cfg.embedding_model,
            self.cfg.reproducible_ids
        )
        logger.info(f"→ {len(docs)} docs → {len(chunks)} chunks")

        selected = random.sample(chunks, min(n, len(chunks))) if chunks else []
        all_qa_pairs: List[Dict[str, Any]] = []
        for chunk in tqdm(selected, desc="QA chunks"):
            if len(all_qa_pairs) >= n:
                break
            all_qa_pairs.extend(self._qa_for_chunk(chunk))

        return all_qa_pairs[:n]


async def generate_test_set(agent_cfg: AgentConfig, num_questions: int) -> List[Dict[str, Any]]:  # MODIFIED: async def
    """
    High-level function to generate a test set for a given agent configuration asynchronously.
    """
    generator = DatasetGenerator(agent_cfg)
    return await generator.generate(num_questions)