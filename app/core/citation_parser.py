import os
import re
import logging
from typing import List, Dict, Any

logger = logging.getLogger(__name__)

CITATION_PATTERN = re.compile(r'\[↩:(?P<doc>[^:\]]+?)(?::(?P<page>\d+))?\]')

def extract_citations_from_text(
    llm_response_text: str,
    retrieved_documents_metadata: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    """
    Finds citation markers and maps them back to chunk_ids,
    using page numbers when available, otherwise falling back
    to the first chunk of that document.
    """
    citations_found = []

    # build lookup: basename(filename) -> [chunk_id,…]
    # and optionally "basename:page" -> [that_exact_chunk]
    lookup: Dict[str, List[int]] = {}
    for meta in retrieved_documents_metadata:
        cid = meta.get("chunk_id")
        # try doc_name first, else derive from source_path
        doc = meta.get("doc_name") or os.path.basename(meta.get("source_path",""))
        if not cid or not doc:
            continue
        key = doc.lower()
        lookup.setdefault(key, []).append(cid)

        pg = meta.get("page_number")
        if pg is not None:
            lookup[f"{key}:{pg}"] = [cid]

    seen = set()
    for m in CITATION_PATTERN.finditer(llm_response_text):
        full = m.group(0)
        if full in seen:
            continue
        seen.add(full)

        name = m.group("doc").strip()
        # skip the Unknown Document placeholders
        if name.lower() == "unknown document":
            continue

        page = m.group("page")
        key = name.lower()
        chunk_id = None

        # try page‐specific first
        if page:
            chunk_id = lookup.get(f"{key}:{page}", [None])[0]

        # fallback to any chunk of that file
        if chunk_id is None:
            chunk_id = lookup.get(key, [None])[0]

        if chunk_id:
            start, end = m.span()
            citations_found.append({
                "marker_text": full,
                "doc_name": name,
                "page": int(page) if page else None,
                "chunk_id": chunk_id,
                "start_char": start,
                "end_char": end
            })
        else:
            logger.warning(f"Could not map citation marker '{full}' to any retrieved chunk.")

    return citations_found