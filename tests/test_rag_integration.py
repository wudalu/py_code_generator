# -*- coding: utf-8 -*-
import logging
import os
import sys

# Ensure the project root is in the Python path
# This allows running the script from the root directory
# Adjust path calculation assuming the script is now in tests/
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

import config
from rag import create_rag_service
# Adjust imports based on actual embedding module structure
try:
    # Use the function exported from the embedding package
    from embedding import get_embedding_model
    # Import ChromaVectorStore from its specific module path
    from embedding.storage.chroma import ChromaVectorStore
except ImportError as e:
    print(f"Error importing embedding/vector store components: {e}")
    # Provide more specific guidance based on the new paths
    print("Please ensure embedding/__init__.py exports get_embedding_model and embedding/storage/chroma.py exists with ChromaVectorStore.")
    sys.exit(1)

# --- Configuration ---
# Configure logging to show debug messages from RAG components
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    stream=sys.stdout # Print logs to console
)
# Optionally reduce verbosity of libraries like sentence-transformers
logging.getLogger("sentence_transformers").setLevel(logging.WARNING)

# --- Test Query ---
# Define a query relevant to the indexed data (e.g., FastAPI)
# Adjust this query based on the content you have indexed
# TEST_QUERY = "How can I run background tasks in FastAPI?"
TEST_QUERY = "Compare FastAPI BackgroundTasks with Celery"
# TEST_QUERY = "FastAPI task queue database"
# TEST_QUERY = "How to handle errors in background jobs fastapi?"


# --- Main Test Execution ---
def run_test():
    logger = logging.getLogger(__name__) # Get logger instance
    logger.info("--- Starting RAG Integration Test ---")

    # 1. Create Embedding Function
    try:
        logger.info(f"Creating embedding function (Provider: {config.EMBEDDING_PROVIDER})...")
        # Call the correctly imported function
        # We assume get_embedding_model() returns the embedding function object directly
        embedding_func = get_embedding_model()
        logger.info("Embedding function created.")
    except Exception as e:
        logger.error(f"Failed to create embedding function: {e}", exc_info=True)
        return

    # 2. Instantiate Vector Store
    persist_dir = config.PERSIST_DIRECTORY
    if not os.path.exists(persist_dir) or not os.listdir(persist_dir):
         logger.error(f"Vector store directory '{persist_dir}' does not exist or is empty.")
         logger.error("Please ensure you have indexed data before running the test.")
         return
    try:
        logger.info(f"Connecting to Vector Store (ChromaDB) at: {persist_dir}")
        # Assuming ChromaVectorStore takes persist_directory and embedding_function
        vector_store = ChromaVectorStore(
            persist_directory=persist_dir,
            embedding_function=embedding_func # Pass the created embedding function
        )
        logger.info("Vector Store connected.")
    except Exception as e:
        logger.error(f"Failed to connect to Vector Store: {e}", exc_info=True)
        return

    # 3. Create RAG Service using the factory function
    # The factory function handles LLM and optional ReRanker initialization based on config
    try:
        logger.info("Creating RAG Service...")
        rag_service = create_rag_service(vector_store=vector_store)
        logger.info("RAG Service created.")
        # Log whether reranker is active based on the service's internal state (if possible)
        # This relies on RAGService having a way to check, e.g., service._reranker is not None
        is_reranker_active = getattr(rag_service, '_reranker', None) is not None
        logger.info(f"Reranker Active in this run: {is_reranker_active}")

    except ValueError as e:
         logger.error(f"Configuration error creating RAG service: {e}")
         logger.error("Please check your config.py and environment variables (e.g., OPENAI_API_KEY).")
         return
    except Exception as e:
        logger.error(f"Failed to create RAG Service: {e}", exc_info=True)
        return

    # 4. Perform RAG Generation
    logger.info(f"--- Sending Query: '{TEST_QUERY}' ---")
    try:
        response = rag_service.generate(query=TEST_QUERY, k=config.DEFAULT_RETRIEVAL_K) # Use k from config
        logger.info("--- RAG Response Received ---")
        print("\nGenerated Response:")
        print("--------------------")
        print(response)
        print("--------------------\n")
    except Exception as e:
        logger.error(f"Error during RAG generation: {e}", exc_info=True)

    logger.info("--- RAG Integration Test Finished ---")

if __name__ == "__main__":
    # Check for necessary environment variables (example for OpenAI)
    # Use the LLM provider configured in config.py for the check
    provider_keys = {
        "openai": "OPENAI_API_KEY",
        "deepseek": "DEEPSEEK_API_KEY",
        "ark": "ARK_API_KEY", # Assuming ARK uses ARK_API_KEY from env
        # Add other providers and their respective API key env var names here
    }
    required_key = provider_keys.get(config.LLM_PROVIDER.lower())

    if required_key and not getattr(config, required_key, None):
        print(f"Error: {required_key} environment variable is not set, but LLM_PROVIDER is '{config.LLM_PROVIDER}'.")
        print(f"Please set the {required_key} environment variable.")
    else:
        run_test() 