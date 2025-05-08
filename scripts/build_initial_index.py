# -*- coding: utf-8 -*-
import logging
import os
import sys
import time

# Add project root to path for sibling imports (config, embedding)
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

import config # Import global config
from embedding import EmbeddingPipeline, VectorStore, Splitter, get_embedding_model
from embedding.splitter import AstPythonSplitter, FallbackSplitter
from embedding.storage import ChromaVectorStore
from utils import setup_logging # Import from new location

# Configure logging (reuse setup from test script)
setup_logging()
logger = logging.getLogger(__name__)

# --- Files to Index ---
# List of files selected from codes_for_embedding/fastapi/fastapi
# Excluding tests, docs, examples etc.
FASTAPI_CORE_FILES = [
    "codes_for_embedding/fastapi/fastapi/middleware/gzip.py",
    "codes_for_embedding/fastapi/fastapi/middleware/cors.py",
    "codes_for_embedding/fastapi/fastapi/middleware/__init__.py",
    "codes_for_embedding/fastapi/fastapi/middleware/httpsredirect.py",
    "codes_for_embedding/fastapi/fastapi/middleware/trustedhost.py",
    "codes_for_embedding/fastapi/fastapi/middleware/wsgi.py",
    "codes_for_embedding/fastapi/fastapi/params.py",
    "codes_for_embedding/fastapi/fastapi/responses.py",
    "codes_for_embedding/fastapi/fastapi/templating.py", # Potentially less relevant? Keeping for now.
    "codes_for_embedding/fastapi/fastapi/security/open_id_connect_url.py",
    "codes_for_embedding/fastapi/fastapi/security/oauth2.py",
    "codes_for_embedding/fastapi/fastapi/security/__init__.py",
    "codes_for_embedding/fastapi/fastapi/security/api_key.py",
    "codes_for_embedding/fastapi/fastapi/security/utils.py",
    "codes_for_embedding/fastapi/fastapi/security/http.py",
    "codes_for_embedding/fastapi/fastapi/security/base.py",
    "codes_for_embedding/fastapi/fastapi/exception_handlers.py",
    "codes_for_embedding/fastapi/fastapi/websockets.py",
    "codes_for_embedding/fastapi/fastapi/applications.py",
    "codes_for_embedding/fastapi/fastapi/concurrency.py",
    "codes_for_embedding/fastapi/fastapi/background.py",
    "codes_for_embedding/fastapi/fastapi/dependencies/models.py",
    "codes_for_embedding/fastapi/fastapi/dependencies/__init__.py",
    "codes_for_embedding/fastapi/fastapi/dependencies/utils.py",
    "codes_for_embedding/fastapi/fastapi/__init__.py",
    "codes_for_embedding/fastapi/fastapi/encoders.py",
    "codes_for_embedding/fastapi/fastapi/types.py",
    "codes_for_embedding/fastapi/fastapi/logger.py",
    "codes_for_embedding/fastapi/fastapi/openapi/models.py",
    "codes_for_embedding/fastapi/fastapi/openapi/constants.py",
    "codes_for_embedding/fastapi/fastapi/openapi/__init__.py",
    "codes_for_embedding/fastapi/fastapi/openapi/docs.py",
    "codes_for_embedding/fastapi/fastapi/openapi/utils.py",
    "codes_for_embedding/fastapi/fastapi/staticfiles.py", # Potentially less relevant? Keeping for now.
    "codes_for_embedding/fastapi/fastapi/cli.py",
    "codes_for_embedding/fastapi/fastapi/utils.py",
    "codes_for_embedding/fastapi/fastapi/routing.py",
    # "codes_for_embedding/fastapi/fastapi/testclient.py", # Excluded
    "codes_for_embedding/fastapi/fastapi/exceptions.py",
    "codes_for_embedding/fastapi/fastapi/param_functions.py",
    "codes_for_embedding/fastapi/fastapi/_compat.py",
    "codes_for_embedding/fastapi/fastapi/requests.py",
    "codes_for_embedding/fastapi/fastapi/datastructures.py",
    "codes_for_embedding/fastapi/fastapi/__main__.py",
]

def main():
    logger.info("--- Starting Initial Index Build Script ---")
    start_time = time.time()

    # --- 1. Initialize Components using Config ---
    logger.info("Initializing components...")
    try:
        embedding_model = get_embedding_model()
        splitters: Dict[str, Splitter] = {
            'python': AstPythonSplitter(),
            'fallback': FallbackSplitter() # Uses config defaults
        }
        vector_store: VectorStore = ChromaVectorStore(
            persist_directory=config.PERSIST_DIRECTORY,
            embedding_function=embedding_model,
        )
        pipeline = EmbeddingPipeline(
            embedding_model=embedding_model,
            vector_store=vector_store,
            splitters=splitters
        )
    except Exception as e:
        logger.error(f"Failed to initialize pipeline components: {e}", exc_info=True)
        return

    # --- 2. Prepare File List ---
    logger.info("Preparing file list...")
    abs_file_paths = []
    missing_files = 0
    for rel_path in FASTAPI_CORE_FILES:
        abs_path = os.path.join(project_root, rel_path)
        if os.path.exists(abs_path):
            abs_file_paths.append(abs_path)
        else:
            logger.warning(f"File not found, skipping: {abs_path}")
            missing_files += 1

    if not abs_file_paths:
        logger.error("No valid files found in the specified list. Exiting.")
        return

    if missing_files > 0:
         logger.warning(f"Skipped {missing_files} missing files.")

    collection_name = config.DEFAULT_COLLECTION_NAME

    # --- 3. Run Processing ---
    logger.info(f"Processing {len(abs_file_paths)} files into collection '{collection_name}'...")
    try:
        result = pipeline.process_files(
            file_paths=abs_file_paths,
            collection_name=collection_name
        )
        logger.info(f"Processing finished. Result: {result}")
    except Exception as e:
        logger.error(f"Error during pipeline processing: {e}", exc_info=True)
        result = {"status": "error", "message": str(e)}

    # --- 4. Log Summary ---
    end_time = time.time()
    duration = end_time - start_time
    logger.info(f"Total time taken: {duration:.2f} seconds.")
    if result.get("status") == "success":
        logger.info(f"Successfully indexed {result.get('chunk_count', 0)} chunks from {result.get('file_count', 0)} files into collection '{collection_name}'.")
        # Optionally verify count
        try:
            info = vector_store.get_collection_info(collection_name)
            logger.info(f"Collection '{collection_name}' info: {info}")
        except Exception as e:
            logger.warning(f"Could not retrieve collection info for verification: {e}")
    else:
        logger.error(f"Indexing failed. Status: {result.get('status')}, Message: {result.get('message')}")

    logger.info("--- Initial Index Build Script Finished ---")


if __name__ == "__main__":
    main() 