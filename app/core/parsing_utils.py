# app/core/parsing_utils.py

import json
import re
import logging
from typing import Any, Dict, Optional, List

logger = logging.getLogger(__name__)


def normalize_yes_no(response: str) -> Optional[bool]:
    """Normalizes a string to a boolean, handling common variations."""
    if not isinstance(response, str):
        return None
    resp_lower = response.strip().lower()
    if re.search(r'^(yes|true)\b', resp_lower):
        return True
    if re.search(r'^(no|false)\b', resp_lower):
        return False
    return None


def safe_json_parse(json_string: str) -> Optional[Dict[str, Any]]:
    """
    Safely parses a JSON string, handling markdown fences and providing a robust regex fallback.
    """
    # Regex to find a JSON object within ```json ... ``` or a standalone { ... }
    match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```|(\{.*?\})', json_string, re.DOTALL)

    if match:
        clean_json_str = match.group(1) if match.group(1) else match.group(2)
        try:
            return json.loads(clean_json_str)
        except (json.JSONDecodeError, TypeError):
            logger.warning(f"Found JSON-like string but failed to parse. Content: {clean_json_str[:100]}...")

    # If no JSON object is found or parsing fails, fall back to regex for all keys
    logger.warning("Could not parse JSON directly. Resorting to regex fallback for all evaluation keys.")
    faithfulness_match = re.search(r'["\']faithfulness["\']\s*:\s*["\']?(Yes|No)["\']?', json_string, re.IGNORECASE)
    relevance_match = re.search(r'["\']answer_relevance["\']\s*:\s*["\']?(Yes|No)["\']?', json_string, re.IGNORECASE)
    conciseness = re.search(r'["\']conciseness_score["\']\s*:\s*(\d)', json_string, re.IGNORECASE)
    coherence = re.search(r'["\']coherence_score["\']\s*:\s*(\d)', json_string, re.IGNORECASE)
    reasoning = re.search(r'["\']reasoning["\']\s*:\s*["\'](.*?)["\']', json_string, re.DOTALL | re.IGNORECASE)

    if any([faithfulness_match, relevance_match, conciseness, coherence]):
        fallback_data = {
            "faithfulness": faithfulness_match.group(1) if faithfulness_match else "No",
            "answer_relevance": relevance_match.group(1) if relevance_match else "No",
            "conciseness_score": int(conciseness.group(1)) if conciseness else -1,
            "coherence_score": int(coherence.group(1)) if coherence else -1,
            "reasoning": reasoning.group(1).strip() if reasoning else "Scores extracted via regex; no reasoning found."
        }
        logger.info(f"Successfully extracted partial scores via regex: {fallback_data}")
        return fallback_data

    logger.error(f"Could not parse any data from LLM output: {json_string[:200]}...")
    return None


def extract_qa_pairs_from_text(text: str) -> List[Dict[str, str]]:
    """Fallback to extract Q&A pairs if strict JSON parsing fails."""
    qa_pairs = []
    pattern = re.compile(
        r"question[\"\']?\s*:\s*[\"\'](.*?)[\"\'].*?answer[\"\']?\s*:\s*[\"\'](.*?)[\"\'].*?type[\"\']?\s*:\s*[\"\'](.*?)[\"\']",
        re.DOTALL | re.IGNORECASE)

    for match in pattern.finditer(text):
        qa_pairs.append({
            "question": match.group(1).strip(),
            "answer": match.group(2).strip(),
            "type": match.group(3).strip()
        })
    return qa_pairs


