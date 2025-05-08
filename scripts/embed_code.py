# -*- coding: utf-8 -*-
import logging
import os
import sys

# Add project root to path for sibling imports (config, embedding)
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

import config # Import global config
from embedding import EmbeddingPipeline, VectorStore, Splitter
from embedding.splitter import AstPythonSplitter, FallbackSplitter
from embedding.storage import ChromaVectorStore

# Select and initialize embedding model based on config
from langchain_core.embeddings import Embeddings

def get_embedding_model() -> Embeddings:
    provider = config.EMBEDDING_PROVIDER.lower()
    if provider == 'huggingface':
        from langchain_community.embeddings import HuggingFaceEmbeddings
        print(f"Using HuggingFace embedding model: {config.HUGGINGFACE_MODEL_NAME}")
        return HuggingFaceEmbeddings(model_name=config.HUGGINGFACE_MODEL_NAME)
    elif provider == 'openai':
        from langchain_openai import OpenAIEmbeddings
        if not config.OPENAI_API_KEY:
            raise ValueError("OPENAI_API_KEY must be set in config or environment for OpenAI embeddings.")
        print(f"Using OpenAI embedding model: {config.OPENAI_EMBEDDING_MODEL}")
        return OpenAIEmbeddings(
            openai_api_key=config.OPENAI_API_KEY,
            model=config.OPENAI_EMBEDDING_MODEL
        )
    elif provider == 'fake':
        from langchain_community.embeddings import FakeEmbeddings
        print("Using Fake embeddings for testing.")
        return FakeEmbeddings(size=768) # Example size
    else:
        raise ValueError(f"Unsupported embedding provider in config: {config.EMBEDDING_PROVIDER}")

def setup_logging():
    """Configures logging to output to both console and file based on config."""
    log_level = getattr(logging, config.LOG_LEVEL, logging.INFO)
    formatter = logging.Formatter(config.LOG_FORMAT)

    # Get root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)

    # Remove existing handlers to avoid duplication if script is run multiple times
    # (though basicConfig usually handles this, explicit removal is safer)
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)

    # Console Handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)

    # File Handler
    try:
        os.makedirs(config.LOG_DIR, exist_ok=True)
        file_handler = logging.FileHandler(config.LOG_FILE, mode='a', encoding='utf-8')
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)
        print(f"Logging to file: {config.LOG_FILE}") # Indicate file logging setup
    except Exception as e:
        logging.error(f"Failed to set up file logging to {config.LOG_FILE}: {e}", exc_info=True)

# Call the setup function
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