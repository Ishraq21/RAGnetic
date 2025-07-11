import logging
import json
from typing import List, Dict, Any

from tqdm import tqdm
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from app.schemas.agent import AgentConfig
from app.pipelines.embed import load_documents_from_source

logger = logging.getLogger(__name__)

QA_GENERATION_PROMPT = """
Your task is to write a factoid question and a concise answer given a context.
The question should be answerable with a specific, brief piece of factual information from the context.
The question should be formulated in the same style as questions users could ask a search engine.
This means the question MUST NOT mention "according to the passage" or "in the context".

Provide your answer as follows:

Output:::
Factoid question: (your factoid question)
Answer: (your answer to the factoid question)

Now here is the context.

Context: {context}
Output:::
"""


def generate_test_set(agent_config: AgentConfig, num_questions: int) -> List[Dict[str, Any]]:
    """
    Generates a synthetic test set of questions and answers (Key Facts)
    from an agent's source documents.
    """
    logger.info(f"--- Starting Test Set Generation for Agent: '{agent_config.name}' ---")

    # 1. Load all documents from the agent's sources
    all_docs = []
    for source in agent_config.sources:
        all_docs.extend(load_documents_from_source(source))

    if not all_docs:
        logger.error("No documents found for the agent. Cannot generate a test set.")
        return []

    logger.info(f"Loaded {len(all_docs)} documents. Generating {num_questions} questions...")

    # 2. Use a local LLM for generation
    # For this task, a smaller, fast model is sufficient.
    llm = ChatOllama(model="llama3")

    prompt_template = ChatPromptTemplate.from_template(QA_GENERATION_PROMPT)
    chain = prompt_template | llm | StrOutputParser()

    # 3. Generate Question/Answer pairs
    generated_qa_pairs = []
    for doc in tqdm(all_docs, desc="Generating Q&A Pairs"):
        if len(generated_qa_pairs) >= num_questions:
            break

        # We only need the page_content for generation
        context = doc.page_content
        try:
            qa_output = chain.invoke({"context": context})

            question = qa_output.split("Factoid question:")[-1].split("Answer:")[0].strip()
            answer = qa_output.split("Answer:")[-1].strip()

            if question and answer:
                generated_qa_pairs.append({
                    "question": question,
                    "answer": answer,  # This is our "Key Fact"
                    "source": doc.metadata.get("source", "N/A")
                })
        except Exception as e:
            logger.warning(f"Failed to generate QA pair for a document chunk: {e}")
            continue

    logger.info(f"Successfully generated {len(generated_qa_pairs)} question/answer pairs.")
    return generated_qa_pairs