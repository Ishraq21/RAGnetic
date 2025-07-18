import pytest
import os
import shutil
import yaml
from pathlib import Path

from app.schemas.agent import AgentConfig
from app.pipelines.embed import embed_agent_data
from app.agents.config_manager import load_agent_config
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from app.core.config import get_api_key

# Define paths relative to the project root for testing purposes
TEST_AGENT_NAME = "test-reproducible-agent-temp"
TEST_AGENTS_DIR = "agents"  # Relative to where cli.py is run
TEST_DATA_DIR = "data"  # Relative to where cli.py is run
TEST_VECTORSTORE_DIR = "vectorstore"  # Relative to where cli.py is run


@pytest.fixture(scope="module")
def setup_test_environment_for_reproducibility():
    """Sets up a clean test environment for reproducible ID embedding tests."""
    # Define actual paths relative to current working directory (project root)
    agent_config_dir = Path(TEST_AGENTS_DIR)
    test_data_dir = Path(TEST_DATA_DIR)
    test_vectorstore_dir = Path(TEST_VECTORSTORE_DIR)

    # Clean up any existing test artifacts before running tests
    for d in [agent_config_dir, test_data_dir, test_vectorstore_dir]:
        if d.exists() and d.is_dir():
            # Only remove if it's the specific test agent's dir or content
            # Check for direct agent name in path or a directory with that name
            if TEST_AGENT_NAME in str(d) or any(f.name == TEST_AGENT_NAME for f in d.iterdir() if f.is_dir()):
                shutil.rmtree(d)
                print(f"Cleaned up {d}")
        os.makedirs(d, exist_ok=True)  # Ensure they exist for the test

    # Create dummy data file
    dummy_data_path = test_data_dir / "repro_test_doc.txt"
    with open(dummy_data_path, "w") as f:
        f.write("This is a consistent dummy document for reproducible ID testing. " * 50)
        f.write("It contains enough content to be split into multiple chunks by the splitter. " * 50)
        f.write("Each sentence is unique enough to generate distinct chunk IDs if not reproducible. " * 50)
    print(f"Created dummy data at {dummy_data_path}")

    # Create dummy agent config file
    agent_config_data = {
        "name": TEST_AGENT_NAME,
        "display_name": "Test Reproducible Agent",
        "description": "Agent for reproducible ID testing.",
        "persona_prompt": "You are a helpful assistant.",
        "sources": [
            {"type": "local", "path": str(dummy_data_path)}
        ],
        "tools": ["retriever"],
        "embedding_model": "text-embedding-3-small",
        "llm_model": "gpt-4o-mini",
        "reproducible_ids": True,  # This is the key setting for this test
        "vector_store": {"type": "faiss"},  # Using FAISS for simplicity in testing load_local
        "chunking": {"mode": "default", "chunk_size": 100, "chunk_overlap": 20},
    }
    agent_config_path = agent_config_dir / f"{TEST_AGENT_NAME}.yaml"
    with open(agent_config_path, "w") as f:
        yaml.dump(agent_config_data, f)
    print(f"Created dummy agent config at {agent_config_path}")

    # Yield control to tests
    yield  # Test runs here

    # Clean up after all tests in this module are done
    try:
        if (test_data_dir / dummy_data_path.name).exists():
            os.remove(test_data_dir / dummy_data_path.name)
            print(f"Removed dummy data at {test_data_dir / dummy_data_path.name}")
        if agent_config_path.exists():
            os.remove(agent_config_path)
            print(f"Removed dummy agent config at {agent_config_path}")
        vectorstore_agent_path = test_vectorstore_dir / TEST_AGENT_NAME
        if vectorstore_agent_path.exists():
            shutil.rmtree(vectorstore_agent_path)
            print(f"Removed vector store at {vectorstore_agent_path}")
    except Exception as e:
        print(f"Error during test environment cleanup: {e}")


def get_faiss_chunk_ids(agent_name: str, vectorstore_base_path: Path) -> set[str]:
    """Loads a FAISS vector store and returns the set of chunk IDs (Document.id)."""
    vectorstore_path = vectorstore_base_path / agent_name

    try:
        # Attempt to get API key, but allow dummy for test if not available
        api_key_val = get_api_key("openai")
    except Exception:
        api_key_val = "dummy_api_key_for_test"  # Fallback for local testing
        print(
            "Warning: OpenAI API key not found for test. Using dummy key. Embeddings might fail if actual API call is made.")

    embeddings = OpenAIEmbeddings(model="text-embedding-3-small", api_key=api_key_val)

    faiss_db = FAISS.load_local(str(vectorstore_path), embeddings, allow_dangerous_deserialization=True)

    # Retrieve all Document.id values from the FAISS docstore
    chunk_ids = set()
    for doc_id_in_store in faiss_db.index_to_docstore_id.values():
        doc = faiss_db.docstore.search(doc_id_in_store)
        if doc and hasattr(doc, 'id'):  # Document.id should now be the chunk_id
            chunk_ids.add(doc.id)

    return chunk_ids


# This test will run once per module due to the fixture scope
@pytest.mark.asyncio
async def test_reproducible_embedding_ids(setup_test_environment_for_reproducibility): # MODIFIED: Add async
    """
    Tests that chunk IDs are reproducible when reproducible_ids is set to True.
    """
    # Load agent config
    # We use TEST_AGENT_NAME directly as load_agent_config expects the name
    agent_config_1 = load_agent_config(TEST_AGENT_NAME)

    # --- First embedding pass ---
    print(f"Running first embedding pass for agent '{TEST_AGENT_NAME}'...")
    await embed_agent_data(agent_config_1) # MODIFIED: Add await

    # Assert vectorstore exists
    vectorstore_path_agent = Path(TEST_VECTORSTORE_DIR) / TEST_AGENT_NAME
    assert vectorstore_path_agent.exists(), "Vector store was not created in the first pass."

    # Retrieve chunk IDs from first pass
    chunk_ids_pass1 = get_faiss_chunk_ids(TEST_AGENT_NAME, Path(TEST_VECTORSTORE_DIR))
    assert len(chunk_ids_pass1) > 0, "No chunks were generated in the first pass."
    print(f"First pass generated {len(chunk_ids_pass1)} unique chunks.")

    # --- Clean up for second pass ---
    # Remove only the agent's vector store directory for a clean re-embed
    if vectorstore_path_agent.exists():
        shutil.rmtree(vectorstore_path_agent)
    print("Vector store removed for second pass.")

    # --- Second embedding pass with the same data and config ---
    # Reload config to simulate a fresh run, though content should be identical
    agent_config_2 = load_agent_config(TEST_AGENT_NAME)
    print(f"Running second embedding pass for agent '{TEST_AGENT_NAME}'...")
    await embed_agent_data(agent_config_2) # MODIFIED: Add await

    # Assert vectorstore exists for second pass
    assert vectorstore_path_agent.exists(), "Vector store was not created in the second pass."

    # Retrieve chunk IDs from second pass
    chunk_ids_pass2 = get_faiss_chunk_ids(TEST_AGENT_NAME, Path(TEST_VECTORSTORE_DIR))
    print(f"Second pass generated {len(chunk_ids_pass2)} unique chunks.")

    # --- Assert that the sets of chunk IDs are identical ---
    assert chunk_ids_pass1 == chunk_ids_pass2, "Chunk IDs are not reproducible between embedding passes."
    assert len(chunk_ids_pass1) == len(chunk_ids_pass2), "Number of chunks differs between passes."

    print(f"Successfully verified reproducible IDs for agent '{TEST_AGENT_NAME}'. All chunk IDs match between passes.")
