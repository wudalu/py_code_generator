# -*- coding: utf-8 -*-
import logging
import os
import sys
import asyncio
import pytest
import pytest_asyncio
from pathlib import Path

# Add project root to path for sibling imports (config, embedding)
# Adjusted path for new location: tests -> project root
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

import config # Import global config
# Update imports: get functions from their new locations
from embedding import EmbeddingPipeline, VectorStore, Splitter, get_embedding_model
from embedding.splitter import AstPythonSplitter, FallbackSplitter
from embedding.storage import ChromaVectorStore
from utils import setup_logging # Import setup_logging from utils

# Import the Streamable HTTP client instead of the SSE one
# from mcp.mcp_client_sse import CodeEmbeddingClient as SseClient
from mcp.mcp_client_streamablehttp import StreamableHttpClient

# Configuration (Adjust as needed)
# Use environment variables or a config file for real applications
SERVER_URL = os.environ.get("MCP_SERVER_URL", "http://localhost:8080") # Point to Streamable HTTP port
TEST_COLLECTION_NAME = "test_pipeline_collection"

# Use tests directory as base for test files
TESTS_DIR = Path(__file__).parent
TEST_CODE_FILE_1 = TESTS_DIR / "test_code_files/sample_code_1.py"
TEST_CODE_FILE_2 = TESTS_DIR / "test_code_files/sample_code_2.py"

# Create dummy test code files if they don't exist
@pytest.fixture(scope="session")
def create_test_files():
    test_code_dir = TESTS_DIR / "test_code_files"
    test_code_dir.mkdir(exist_ok=True)

    # Define content for file 1
    file1_content = '''"""This is sample code file 1."""
def hello(name: str):
    print(f"Hello, {name}!")

class Calculator:
    def add(self, a, b):
        return a + b
'''
    # Define content for file 2
    file2_content = '''import os

# This is sample code file 2.
def process_data(data):
    # Some complex logic here
    if 'value' in data:
        return data['value'] * 10
    return None
'''

    if not TEST_CODE_FILE_1.exists():
        with open(TEST_CODE_FILE_1, "w") as f:
            f.write(file1_content)
    if not TEST_CODE_FILE_2.exists():
        with open(TEST_CODE_FILE_2, "w") as f:
            f.write(file2_content)

    yield # Let tests run
    # Teardown is optional

@pytest_asyncio.fixture(scope="function")
async def mcp_client():
    """Provides an initialized StreamableHttpClient instance for tests."""
    print(f"\nAttempting to connect to MCP server at: {SERVER_URL}")
    client = StreamableHttpClient(SERVER_URL)
    try:
        # Initialize explicitly
        init_result = await client.initialize()
        print(f"MCP Client initialized successfully. Capabilities: {init_result}")
        
        # --- Clean up test collection before yielding --- 
        print(f"Attempting to clean up collection '{TEST_COLLECTION_NAME}' before test...")
        delete_result = await client.delete_collection(TEST_COLLECTION_NAME)
        # Log deletion result, but don't fail the fixture if deletion fails (collection might not exist)
        if delete_result.get("status") == "success":
             print(f"Pre-test cleanup successful for '{TEST_COLLECTION_NAME}'.")
        elif "Collection not found" in delete_result.get("message", ""):
             print(f"Collection '{TEST_COLLECTION_NAME}' not found during pre-test cleanup (OK).")
        else:
            logger.warning(f"Pre-test cleanup potentially failed for '{TEST_COLLECTION_NAME}': {delete_result}")
            
        yield client
    except Exception as e:
        pytest.fail(f"Failed to initialize MCP client: {e}", pytrace=True)
    finally:
        # Disconnect explicitly
        if client.is_connected:
            print("Closing MCP Client connection...")
            await client.close()
            print("MCP Client connection closed.")
        else:
            print("MCP Client was not connected, skipping close.")

@pytest.mark.asyncio
async def test_embedding_pipeline_integration(mcp_client: StreamableHttpClient, create_test_files):
    """Tests the full process: process files, list, get info, search."""
    client = mcp_client # Alias for clarity

    # --- 1. Process Files --- 
    print(f"\nTesting: Process Files ({TEST_CODE_FILE_1.name}, {TEST_CODE_FILE_2.name})")
    file_paths = [str(TEST_CODE_FILE_1), str(TEST_CODE_FILE_2)]
    process_result = await client.process_files(file_paths, TEST_COLLECTION_NAME)
    print(f"Process files result: {process_result}")
    assert process_result["status"] == "success"
    assert process_result["collection_name"] == TEST_COLLECTION_NAME
    assert process_result["file_count"] == 2
    assert process_result["chunk_count"] > 0 # Should generate some chunks

    # --- 2. List Collections --- 
    print("\nTesting: List Collections")
    collections_result = await client.list_collections()
    print(f"List collections result: {collections_result}")
    assert TEST_COLLECTION_NAME in collections_result.get("collections", [])

    # --- 3. Get Collection Info --- 
    print("\nTesting: Get Collection Info ({TEST_COLLECTION_NAME})")
    info_result = await client.get_collection_info(TEST_COLLECTION_NAME)
    print(f"Get collection info result: {info_result}")
    assert info_result.get("name") == TEST_COLLECTION_NAME
    # Add more assertions based on expected info (e.g., count)
    # ChromaDB might return `count` or similar in metadata
    assert "count" in info_result, f"Expected 'count' key in collection info: {info_result}"
    assert info_result.get("count") == process_result["chunk_count"] 

    # --- 4. Search Code --- 
    search_query = "calculator class"
    print("\nTesting: Search Code ('{search_query}')")
    search_results = await client.search_code(search_query, k=1, collection_name=TEST_COLLECTION_NAME)
    print(f"Search results: {search_results}")
    assert len(search_results) > 0
    # Check if the content seems relevant
    first_result = search_results[0]
    assert "Calculator" in first_result.get("content", "")
    assert first_result.get("metadata").get("source") == str(TEST_CODE_FILE_1)

    # --- 5. Search with different query --- 
    search_query_2 = "process data function"
    print("\nTesting: Search Code ('{search_query_2}')")
    search_results_2 = await client.search_code(search_query_2, k=1, collection_name=TEST_COLLECTION_NAME)
    print(f"Search results: {search_results_2}")
    assert len(search_results_2) > 0
    first_result_2 = search_results_2[0]
    assert "process_data" in first_result_2.get("content", "")
    assert first_result_2.get("metadata").get("source") == str(TEST_CODE_FILE_2)

    # Add more specific tests if needed
    print("\nEmbedding pipeline integration test completed successfully.")

# Note: This test assumes the chroma_db_store directory is cleaned up
#       between test runs or uses a unique name if run in parallel.
#       For simplicity, it reuses the same collection name.

# Select and initialize embedding model based on config
# from langchain_core.embeddings import Embeddings

# def get_embedding_model() -> Embeddings:
#     ...
# Function implementation removed ...

# def setup_logging():
#     ...
# Function implementation removed ...

# Call the setup function - Now imported
setup_logging()

# Configure logging --> Removed basicConfig call
# logging.basicConfig(level=logging.INFO,
#                     format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__) # Get logger after setup

def main():
    logger.info("--- Starting Code Embedding Script ---")

    # --- 1. Initialize Components using Config --- 
    try:
        embedding_model = get_embedding_model()
    except Exception as e:
        logger.error(f"Failed to initialize embedding model: {e}", exc_info=True)
        return

    # Initialize splitters (FallbackSplitter uses config defaults)
    splitters: Dict[str, Splitter] = {
        'python': AstPythonSplitter(),
        'fallback': FallbackSplitter() # Uses config for defaults
    }

    # Initialize vector store
    vector_store: VectorStore = ChromaVectorStore(
        persist_directory=config.PERSIST_DIRECTORY,
        embedding_function=embedding_model,
        # collection_name=config.DEFAULT_COLLECTION_NAME # Let pipeline handle default/passed name
    )

    # --- 2. Initialize Pipeline --- 
    pipeline = EmbeddingPipeline(
        embedding_model=embedding_model,
        vector_store=vector_store,
        splitters=splitters
    )

    # --- 3. Define Files to Process --- 
    # Example: Use the same test files as before
    # Adjust the path relative to this script or use absolute paths
    test_file_dir = os.path.join(project_root, 'rag', 'data_src', 'fastapi', 'fastapi')
    files_to_process = [
        os.path.join(test_file_dir, 'exceptions.py'),
        os.path.join(test_file_dir, 'concurrency.py'),
        # Add a non-python file for fallback testing
        # os.path.join(project_root, 'README.md') 
    ]
    files_to_process = [f for f in files_to_process if os.path.exists(f)]
    if not files_to_process:
        logger.error("No valid files found to process.")
        return

    collection_name = config.DEFAULT_COLLECTION_NAME

    # --- 4. Run Processing --- 
    logger.info(f"Processing {len(files_to_process)} files into collection '{collection_name}'...")
    result = pipeline.process_files(
        file_paths=files_to_process,
        collection_name=collection_name
    )

    logger.info(f"Processing finished. Result: {result}")

    # --- 5. (Optional) Test Search --- 
    if result['status'] == 'success' and result['chunk_count'] > 0:
        try:
            search_query = "HTTPException"
            logger.info(f"Performing a test search for: '{search_query}'")
            search_results = vector_store.search(search_query, k=2, collection_name=collection_name)
            logger.info(f"Found {len(search_results)} search results:")
            for i, res in enumerate(search_results):
                logger.info(f"  Result {i+1}: Metadata={res['metadata']}")
                # logger.info(f"  Content: {res['content'][:100]}...") 
        except Exception as e:
            logger.error(f"Test search failed: {e}", exc_info=True)

    logger.info("--- Code Embedding Script Finished ---")

if __name__ == "__main__":
    main() 