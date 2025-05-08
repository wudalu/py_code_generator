# -*- coding: utf-8 -*-
import logging
import os
import sys
import pprint

# Add project root to path for sibling imports (config, embedding, utils)
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

import config # Import global config
from embedding import VectorStore, get_embedding_model
from embedding.storage import ChromaVectorStore
from utils import setup_logging

# Configure logging
setup_logging()
logger = logging.getLogger(__name__)

def main():
    logger.info("--- Starting Index Query Script ---")

    # --- 1. Initialize Components --- 
    logger.info("Initializing components...")
    try:
        # Need the embedding model to potentially embed the query (though Chroma might do it)
        # Crucially, needs to be the *same* model used for indexing.
        embedding_model = get_embedding_model()

        # Initialize vector store - pointing to existing persisted data
        vector_store: VectorStore = ChromaVectorStore(
            persist_directory=config.PERSIST_DIRECTORY,
            embedding_function=embedding_model,
            # No need to specify collection name here if we pass it to search
        )
        logger.info(f"Vector store initialized from: {config.PERSIST_DIRECTORY}")

    except Exception as e:
        logger.error(f"Failed to initialize components: {e}", exc_info=True)
        return

    # --- 2. Get Query and Collection --- 
    collection_name = config.DEFAULT_COLLECTION_NAME
    # Load the collection to ensure it exists and potentially initialize client
    logger.info(f"Attempting to load collection: '{collection_name}'")
    if not vector_store.load_collection(collection_name):
        logger.error(f"Collection '{collection_name}' not found or could not be loaded.")
        # Check if the collection exists at all
        try:
            available_collections = vector_store.get_collections()
            logger.info(f"Available collections: {available_collections}")
        except Exception:
            logger.warning("Could not retrieve list of available collections.")
        return

    # --- Interactive Query Loop --- 
    while True:
        # Get query from user input
        try:
            query = input(f"\nEnter search query (or 'q' to quit): ")
        except EOFError:
            break # Exit if input stream is closed

        if not query:
            continue # Ask again if input is empty

        query_lower = query.strip().lower()
        if query_lower == 'q':
            logger.info("Exiting interactive query session.")
            break # Exit loop

        k = config.DEFAULT_RETRIEVAL_K

        # --- Perform Search ---
        logger.info(f"Searching collection '{collection_name}' for '{query}' (k={k})...")
        try:
            search_results = vector_store.search(query, k=k, collection_name=collection_name)
        except Exception as e:
            logger.error(f"Search failed: {e}", exc_info=True)
            continue # Continue to next query prompt

        # --- Display Results ---
        logger.info(f"Found {len(search_results)} results:")
        if not search_results:
            logger.info("No results found for this query.")
        else:
            for i, res in enumerate(search_results):
                print("-" * 30)
                print(f"Result {i+1}/{len(search_results)}")
                print("Metadata:")
                pprint.pprint(res['metadata'])
                print("Content Snippet:")
                # Print first few lines or characters
                content_lines = res['content'].splitlines()
                snippet = "\n".join(content_lines[:10]) # Print first 10 lines
                if len(content_lines) > 10:
                    snippet += "\n... (truncated)"
                print(snippet)
                print("-" * 30)

    logger.info("--- Index Query Script Finished ---")

if __name__ == "__main__":
    main() 