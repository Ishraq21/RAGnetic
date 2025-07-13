import pytest
from langchain_core.documents import Document
from app.evaluation.benchmark import (
    calculate_retrieval_metrics,
    calculate_document_uniqueness
)

# --- Test Data ---

# Mock documents for testing
DOC_A = Document(id="doc_a", page_content="This is document A.")
DOC_B = Document(id="doc_b", page_content="This is document B.")
DOC_C = Document(id="doc_c", page_content="This is document C.")
DOC_A_DUPLICATE = Document(id="doc_a_dup", page_content="This is document A.")


# --- Tests for calculate_retrieval_metrics ---

def test_retrieval_metrics_perfect_match():
    """Test when the ground truth ID is the first one retrieved."""
    retrieved = [DOC_A, DOC_B, DOC_C]
    ground_truth_id = "doc_a"
    metrics = calculate_retrieval_metrics(retrieved, ground_truth_id, k=3)

    assert metrics["retrieval_precision"] == 1 / 3
    assert metrics["retrieval_recall"] == 1.0
    assert pytest.approx(metrics["retrieval_f1"]) == 0.5
    assert metrics["retrieval_mrr"] == 1.0
    assert metrics["retrieval_hit_at_k"] == 1.0

def test_retrieval_metrics_match_at_k():
    """Test when the ground truth ID is retrieved but not first."""
    retrieved = [DOC_B, DOC_C, DOC_A]
    ground_truth_id = "doc_a"
    metrics = calculate_retrieval_metrics(retrieved, ground_truth_id, k=3)

    assert metrics["retrieval_precision"] == 1 / 3
    assert metrics["retrieval_recall"] == 1.0
    assert pytest.approx(metrics["retrieval_f1"]) == 0.5
    assert metrics["retrieval_mrr"] == 1 / 3
    assert metrics["retrieval_hit_at_k"] == 1.0

def test_retrieval_metrics_no_match():
    """Test when the ground truth ID is not in the retrieved docs."""
    retrieved = [DOC_B, DOC_C]
    ground_truth_id = "doc_a"
    metrics = calculate_retrieval_metrics(retrieved, ground_truth_id, k=3)

    assert metrics["retrieval_precision"] == 0.0
    assert metrics["retrieval_recall"] == 0.0
    assert metrics["retrieval_f1"] == 0.0
    assert metrics["retrieval_mrr"] == 0.0
    assert metrics["retrieval_hit_at_k"] == 0.0

def test_retrieval_metrics_empty_retrieval():
    """Test behavior with empty retrieved docs."""
    retrieved = []
    ground_truth_id = "doc_a"
    metrics = calculate_retrieval_metrics(retrieved, ground_truth_id, k=3)

    for key in metrics:
        assert metrics[key] == 0.0

def test_retrieval_metrics_no_ground_truth():
    """Test behavior when ground_truth_id is None."""
    retrieved = [DOC_A, DOC_B]
    metrics = calculate_retrieval_metrics(retrieved, None, k=3)

    for key in metrics:
        assert metrics[key] == 0.0


# --- Tests for calculate_document_uniqueness ---

def test_document_uniqueness_all_unique():
    """Test when all documents have unique content."""
    docs = [DOC_A, DOC_B, DOC_C]
    assert calculate_document_uniqueness(docs) == 1.0

def test_document_uniqueness_with_duplicates():
    """Test with some duplicate content."""
    docs = [DOC_A, DOC_B, DOC_A_DUPLICATE]
    assert calculate_document_uniqueness(docs) == 2 / 3

def test_document_uniqueness_all_duplicates():
    """Test when all documents have the same content."""
    docs = [DOC_A, DOC_A_DUPLICATE]
    assert calculate_document_uniqueness(docs) == 0.5

def test_document_uniqueness_empty_list():
    """Test with an empty list of documents."""
    assert calculate_document_uniqueness([]) == 0.0

