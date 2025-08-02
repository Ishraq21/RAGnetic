import logging
import re
from typing import List, Optional, Tuple
from app.schemas.agent import DataPolicy, PIIConfig, KeywordFilterConfig

logger = logging.getLogger(__name__)


def apply_data_policies(text: str, policies: List[DataPolicy], policy_context: str = "document") -> Tuple[str, bool]:
    """
    Applies a list of data policies (PII redaction and keyword filtering) to text content.

    This function serves as a centralized, non-duplicated implementation for all loaders.

    Args:
        text (str): The input text content to be processed.
        policies (List[DataPolicy]): A list of data policies to apply.
        policy_context (str): The context of the text being processed (e.g., "document", "page", "row").
                              Used for more informative logging.

    Returns:
        Tuple[str, bool]: A tuple containing the processed text and a boolean flag.
                          The boolean is True if the document/text was blocked by a policy,
                          and False otherwise.
    """
    processed_text = text
    document_blocked = False

    for policy in policies:
        if policy.type == 'pii_redaction' and policy.pii_config:
            pii_config: PIIConfig = policy.pii_config
            for pii_type in pii_config.types:
                pattern = None
                replacement = None
                # Regex patterns for various PII types
                if pii_type == 'email':
                    pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
                    replacement = pii_config.redaction_placeholder or (pii_config.redaction_char * 8)
                elif pii_type == 'phone':
                    pattern = r'\b(?:\+\d{1,3}[-. ]?)?\(?\d{3}\)?[-. ]?\d{3}[-. ]?\d{4}\b'
                    replacement = pii_config.redaction_placeholder or (pii_config.redaction_char * 8)
                elif pii_type == 'ssn':
                    pattern = r'\b\d{3}[- ]?\d{2}[- ]?\d{4}\b'
                    replacement = pii_config.redaction_placeholder or (pii_config.redaction_char * 9)
                elif pii_type == 'credit_card':
                    pattern = r'\b(?:4\d{3}[- ]?\d{4}[- ]?\d{4}[- ]?\d{4}|5[1-5]\d{2}[- ]?\d{4}[- ]?\d{4}[- ]?\d{4}|3[47]\d{13}|6(?:011|5\d{2})[- ]?\d{4}[- ]?\d{4}[- ]?\d{4})\b'
                    replacement = pii_config.redaction_placeholder or (pii_config.redaction_char * 16)
                elif pii_type == 'name':
                    logger.warning(
                        f"PII type '{pii_type}' (name) is complex and not fully implemented for regex-based redaction. Skipping.")
                    continue

                if pattern:
                    processed_text = re.sub(pattern, replacement, processed_text)
                    logger.debug(f"Applied {pii_type} redaction policy. Replaced with: {replacement}")

        elif policy.type == 'keyword_filter' and policy.keyword_filter_config:
            kw_config: KeywordFilterConfig = policy.keyword_filter_config
            for keyword in kw_config.keywords:
                if keyword in processed_text:
                    if kw_config.action == 'redact':
                        replacement = kw_config.redaction_placeholder or (kw_config.redaction_char * len(keyword))
                        processed_text = processed_text.replace(keyword, replacement)
                        logger.debug(f"Applied keyword redaction for '{keyword}'. Replaced with: {replacement}")
                    elif kw_config.action == 'block_chunk':
                        # This policy is applied at the "document" level (e.g., file, page, row),
                        # so we'll redact it here and log a warning. The upstream chunking
                        # logic in embed.py will handle chunk-level blocking if configured
                        # differently there.
                        replacement = kw_config.redaction_placeholder or (kw_config.redaction_char * len(keyword))
                        processed_text = processed_text.replace(keyword, replacement)
                        logger.warning(
                            f"Keyword '{keyword}' found. This {policy_context} contains content that should ideally be blocked at chunk level. Currently redacting.")
                    elif kw_config.action == 'block_document':
                        logger.warning(
                            f"Keyword '{keyword}' found. {policy_context.capitalize()} is marked for blocking by policy. Content will be discarded.")
                        document_blocked = True
                        return "", document_blocked

    return processed_text, document_blocked