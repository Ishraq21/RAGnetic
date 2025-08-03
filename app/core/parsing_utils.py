# app/core/parsing_utils.py

import json
import re
import logging
from typing import Any, Dict, Optional, List
from pathlib import Path
import csv
from io import StringIO
import yaml
import asyncio

from langchain_core.documents import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Conditional Imports based on available libraries
try:
    from pypdf import PdfReader
    PDF_LOADER_AVAILABLE = True
except ImportError:
    PDF_LOADER_AVAILABLE = False
    logging.getLogger(__name__).warning("PyPDF not found. PDF parsing will be skipped.")

try:
    from docx import Document as DocxDocument
    DOCX_LOADER_AVAILABLE = True
except ImportError:
    DOCX_LOADER_AVAILABLE = False
    logging.getLogger(__name__).warning("python-docx not found. DOCX parsing will be skipped.")

logger = logging.getLogger(__name__)

# --- Existing Functions (from your provided file) ---

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

# --- NEW Functions for Document Parsing and Chunking ---

# Default Chunking Parameters (can be made configurable if needed)
DEFAULT_CHUNK_SIZE = 1000
DEFAULT_CHUNK_OVERLAP = 100

async def parse_document_to_chunks(file_path: Path) -> List[Document]:
    """
    Parses a document from the given file path, extracts its text content,
    and splits it into chunks suitable for embedding.
    Adds 'doc_name' and 'page_number' (if applicable) to chunk metadata.
    """
    file_extension = file_path.suffix.lower()
    # Store original file name for metadata
    original_file_name = file_path.name

    # --- Step 1: Extract text content based on file type ---
    try:
        if file_extension == '.txt':
            content = await asyncio.to_thread(file_path.read_text, encoding='utf-8') # Wrap sync file read
            return _chunk_text(content, original_file_name)
        elif file_extension == '.pdf':
            if not PDF_LOADER_AVAILABLE:
                raise ImportError("PyPDF is not installed. Cannot parse PDF files.")
            reader = await asyncio.to_thread(PdfReader, file_path) # Wrap sync PdfReader init
            all_chunks = []
            for i, page in enumerate(reader.pages):
                page_content = await asyncio.to_thread(page.extract_text) or "" # Wrap sync extract_text
                if page_content.strip(): # Only process pages with content
                    page_chunks = _chunk_text(page_content, original_file_name, page_number=i + 1)
                    all_chunks.extend(page_chunks) # This is now fine as page_chunks is List[Document]
            if not all_chunks:
                logger.warning(f"No text extracted from PDF: {file_path.name}")
                return []
            return all_chunks
        elif file_extension == '.docx':
            if not DOCX_LOADER_AVAILABLE:
                raise ImportError("python-docx is not installed. Cannot parse DOCX files.")
            doc = await asyncio.to_thread(DocxDocument, file_path) # Wrap sync DocxDocument init
            content = await asyncio.to_thread("\n".join, [paragraph.text for paragraph in doc.paragraphs if paragraph.text.strip()]) # Wrap sync join
            return _chunk_text(content, original_file_name)
        elif file_extension == '.csv':
            all_chunks = []
            csv_string = await asyncio.to_thread(file_path.read_text, encoding='utf-8') # Wrap sync file read
            csv_file = StringIO(csv_string)
            reader = await asyncio.to_thread(csv.reader, csv_file) # Wrap sync csv reader init
            header = await asyncio.to_thread(next, reader, None) # Wrap sync next

            for i, row in enumerate(reader):
                row_text = ", ".join(row)
                if row_text.strip():
                    chunk = Document(page_content=row_text, metadata={
                        "doc_name": original_file_name,
                        "row_number": i + 1,
                        "original_header": header
                    })
                    all_chunks.append(chunk)
            if not all_chunks:
                logger.warning(f"No rows extracted from CSV: {file_path.name}")
                return []
            return all_chunks
        elif file_extension in ['.json', '.yaml', '.yml', '.hcl', '.tf']:
            raw_content = await asyncio.to_thread(file_path.read_text, encoding='utf-8') # Wrap sync file read
            try:
                if file_extension == '.json':
                    parsed_data = await asyncio.to_thread(json.loads, raw_content) # Wrap sync json.loads
                else:
                    parsed_data = await asyncio.to_thread(yaml.safe_load, raw_content) # Wrap sync yaml.safe_load

                content = await asyncio.to_thread(json.dumps, parsed_data, indent=2) # Wrap sync json.dumps
                return _chunk_text(content, original_file_name)
            except (json.JSONDecodeError, yaml.YAMLError) as e:
                logger.error(f"Failed to parse structured file {file_path.name}: {e}")
                raise ValueError(f"Invalid {file_extension.lstrip('.').upper()} format.")
        elif file_extension == '.ipynb':
            content = await asyncio.to_thread(file_path.read_text, encoding='utf-8') # Wrap sync file read
            try:
                notebook_data = await asyncio.to_thread(json.loads, content) # Wrap sync json.loads
                extracted_cells = []
                for cell in notebook_data.get('cells', []):
                    if 'source' in cell:
                        extracted_cells.append("".join(cell['source']))
                content = "\n\n".join(extracted_cells)
            except json.JSONDecodeError:
                logger.warning(f"Could not parse {file_path.name} as JSON, treating as raw text.")
            return _chunk_text(content, original_file_name)
        elif file_extension == '.md':
            content = await asyncio.to_thread(file_path.read_text, encoding='utf-8') # Wrap sync file read
            return _chunk_text(content, original_file_name)
        elif file_extension == '.log':
            content = await asyncio.to_thread(file_path.read_text, encoding='utf-8') # Wrap sync file read
            return _chunk_text(content, original_file_name)
        else:
            raise ValueError(f"Unsupported file type for parsing: {file_extension}")

    except Exception as e:
        logger.error(f"Error reading or parsing file {file_path.name}: {e}", exc_info=True)
        raise

def _chunk_text(text: str, doc_name: str, page_number: Optional[int] = None) -> List[Document]:
    """
    Splits raw text content into Langchain Documents (chunks) with metadata.
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=DEFAULT_CHUNK_SIZE,
        chunk_overlap=DEFAULT_CHUNK_OVERLAP,
        length_function=len,
        is_separator_regex=False,
    )

    chunks = text_splitter.create_documents([text])

    # Add common metadata to each chunk
    for i, chunk in enumerate(chunks):
        chunk.metadata["doc_name"] = doc_name
        if page_number is not None:
            chunk.metadata["page_number"] = page_number
        chunk.metadata["chunk_index"] = i
    return chunks