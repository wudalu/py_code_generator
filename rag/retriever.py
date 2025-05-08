# -*- coding: utf-8 -*-
import logging
from typing import List, Dict, Any
# Attempt to import VectorStore from the storage package directly
from embedding.storage import VectorStore
# Removed incorrect import of DocumentChunk
# from .models import DocumentChunk

logger = logging.getLogger(__name__)

class Retriever:
    """Handles retrieving relevant document chunks from the vector store."""
    def __init__(self, vector_store: VectorStore):
        self._vector_store = vector_store
        logger.info(f"Retriever initialized with vector store: {type(vector_store).__name__}")

    def retrieve(self, query_for_retrieval: str, k: int, collection_name: str) -> List[Dict[str, Any]]:
        """
        Retrieves the top k most relevant document chunks for the query.

        Args:
            query_for_retrieval: The query text (potentially optimized) to use for embedding and search.
            k: The number of chunks to retrieve.
            collection_name: The name of the collection to search within.

        Returns:
            A list of dictionaries, each containing 'content' and 'metadata'.
        """
        logger.info(f"Retrieving top {k} documents for query from collection '{collection_name}'.")
        logger.debug(f"Query text used for embedding: '{query_for_retrieval[:100]}...'")

        try:
            # VectorStore search should return List[Dict[str, Any]] as per interface
            # Chroma implementation needs to align if it doesn't already
            results = self._vector_store.search(query=query_for_retrieval, k=k, collection_name=collection_name)

            if not results:
                logger.warning(f"Vector store returned no results for the query in collection '{collection_name}'.")
                return []

            # The results should already be in the correct format List[Dict[str, Any]]
            logger.info(f"Retrieved {len(results)} documents.")
            return results

        except Exception as e:
            logger.error(f"Error retrieving documents from vector store: {e}", exc_info=True)
            return [] # Return empty list on error

__all__ = ["Retriever"] 