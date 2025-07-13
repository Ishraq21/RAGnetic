# app/evaluation/dataset_generator.py

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
        # Initialize the LLM based on the agent's configuration for generation
        self.llm = get_llm_model(cfg.llm_model, temperature=0.2)
        logger.info(f"Generating test set for '{cfg.name}' with LLM={cfg.llm_model}")

        # Set up the prompt and parser for generating structured Q&A data
        self.prompt = ChatPromptTemplate.from_template(QA_PROMPT)
        self.parser = JsonOutputParser()

    def _qa_for_chunk(self, chunk: LangChainDocument) -> List[Dict[str, Any]]:
        """
        Generates a list of question-answer pairs for a single document chunk.
        Args:
            chunk: The document chunk to generate Q&A from.
        Returns:
            A list of dictionaries, where each dictionary is a Q&A pair.
        """
        # Create a processing chain: prompt -> LLM -> JSON parser
        chain = self.prompt | self.llm | self.parser
        try:
            # Invoke the chain with the chunk's content
            qa_list = chain.invoke({"context": chunk.page_content})
            if not isinstance(qa_list, list):
                return []
            # Augment each generated Q&A pair with the ground truth chunk ID and source text
            for qa in qa_list:
                qa["retrieval_ground_truth_chunk_id"] = chunk.id
                qa["source_text"] = chunk.page_content
            return qa_list
        except Exception as e:
            logger.error(f"Chunk {chunk.id} QA generation failed: {e}", exc_info=True)
            return []

    def generate(self, n: int) -> List[Dict[str, Any]]:
        """
        Generates the full dataset of n questions.
        Args:
            n: The total number of questions to generate.
        Returns:
            A list of generated Q&A pairs.
        """
        # Load documents from all sources specified in the agent config
        docs = []
        for src in self.cfg.sources:
            docs.extend(load_documents_from_source(src, self.cfg.reproducible_ids))

        # Sort documents by their original ID to ensure deterministic processing.
        # This is crucial for making sure chunk IDs are consistent between
        # test set generation and agent deployment, which solves the 0% recall issue.
        docs.sort(key=lambda d: d.metadata.get("original_doc_id", ""))

        # Split the loaded documents into smaller chunks for processing
        chunks = _get_chunks_from_documents(
            docs, self.cfg.chunking, self.cfg.embedding_model, self.cfg.reproducible_ids
        )
        logger.info(f"→ {len(docs)} docs → {len(chunks)} chunks")

        # Select a random sample of chunks to generate questions from
        selected_chunks = random.sample(chunks, min(n, len(chunks)))

        all_qa_pairs = []
        # Iterate over the selected chunks with a progress bar
        for chunk in tqdm(selected_chunks, desc="QA chunks"):
            if len(all_qa_pairs) >= n:
                break
            all_qa_pairs.extend(self._qa_for_chunk(chunk))

        # Return the requested number of Q&A pairs
        return all_qa_pairs[:n]


def generate_test_set(agent_cfg: AgentConfig, num_questions: int) -> List[Dict[str, Any]]:
    """
    High-level function to generate a test set for a given agent configuration.
    Args:
        agent_cfg: The configuration of the agent.
        num_questions: The number of question-answer pairs to generate.
    Returns:
        A list of generated Q&A pairs.
    """
    generator = DatasetGenerator(agent_cfg)
    return generator.generate(num_questions)
