# -*- coding: utf-8 -*-
from typing import List, Dict, Any, Optional
import logging
import os

# Assuming interface and LangChain types are importable
from ..storage.interface import VectorStore
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_community.vectorstores import Chroma
# Need access to Chroma's base client API for listing collections
import chromadb

logger = logging.getLogger(__name__)

class ChromaVectorStore: # No need to inherit Protocol
    """ChromaDB implementation of the VectorStore interface."""

    _client: Optional[Chroma] = None # LangChain Chroma wrapper instance
    _persist_directory: str
    _embedding_function: Embeddings
    _collection_name: Optional[str] = None
    # Store the base chromadb client for administrative tasks like listing collections
    _db_client: Optional[chromadb.Client] = None

    def __init__(self, persist_directory: str, embedding_function: Embeddings, collection_name: Optional[str] = None):
        """Initialize the ChromaDB vector store wrapper."""
        self._persist_directory = persist_directory
        self._embedding_function = embedding_function
        # Ensure the persist directory exists
        os.makedirs(persist_directory, exist_ok=True)

        # Initialize the base chromadb client
        try:
            self._db_client = chromadb.PersistentClient(path=persist_directory)
        except Exception as e:
            logger.error(f"Failed to initialize chromadb persistent client at {persist_directory}: {e}", exc_info=True)
            # Decide if this is a fatal error for the constructor
            raise

        logger.info(f"ChromaVectorStore initialized. Persist directory: {persist_directory}")
        # Try to load the default collection immediately if specified
        if collection_name:
            self.load_collection(collection_name)
        else:
            # If no default collection specified, maybe load the first one? Or require explicit load/add.
            # For now, _client remains None until load_collection or add_documents initializes it.
            logger.info("No default collection specified. Use load_collection() or add_documents() to initialize.")

    def _init_chroma_client(self, collection_name: str) -> Optional[Chroma]:
        """Helper to initialize the LangChain Chroma client wrapper."""
        try:
            client = Chroma(
                collection_name=collection_name,
                embedding_function=self._embedding_function,
                persist_directory=self._persist_directory,
                client=self._db_client # Pass the base client
            )
            return client
        except Exception as e:
            # Catch specific exceptions? e.g., collection not found if that's relevant here.
            logger.error(f"Failed to initialize LangChain Chroma wrapper for collection '{collection_name}': {e}", exc_info=True)
            return None

    def add_documents(self, documents: List[Document], collection_name: Optional[str] = None) -> None:
        """Add documents to the specified collection. Creates collection if it doesn't exist."""
        target_collection = collection_name or self._collection_name
        if not target_collection:
            raise ValueError("Collection name must be specified either during init or in add_documents call.")

        logger.info(f"Adding {len(documents)} documents to collection '{target_collection}'...")
        try:
            # Chroma.from_documents handles creation if collection doesn't exist
            # It reuses existing if it finds one with the same name.
            # Pass the base client to potentially speed up connection/reuse.
            self._client = Chroma.from_documents(
                documents=documents,
                embedding=self._embedding_function,
                collection_name=target_collection,
                persist_directory=self._persist_directory,
                client=self._db_client
            )
            # Persist changes explicitly if needed (depends on Chroma version/behavior)
            # self._client.persist() # May not be needed with PersistentClient
            self._collection_name = target_collection # Update the active collection name
            logger.info(f"Successfully added documents to collection '{target_collection}'.")
        except Exception as e:
            logger.error(f"Failed to add documents to Chroma collection '{target_collection}': {e}", exc_info=True)
            # Re-raise or handle as appropriate
            raise

    def load_collection(self, collection_name: str) -> bool:
        """Load an existing collection. Returns True if successful, False otherwise."""
        logger.info(f"Attempting to load Chroma collection: {collection_name}")
        client = self._init_chroma_client(collection_name)
        if client:
            self._client = client
            self._collection_name = collection_name
            logger.info(f"Successfully loaded collection '{collection_name}'.")
            return True
        else:
            self._client = None # Ensure client is None if loading failed
            self._collection_name = None
            return False

    def search(self, query: str, k: int = 5, collection_name: Optional[str] = None) -> List[Dict[str, Any]]:
        """Search for similar documents in the specified collection."""
        target_collection = collection_name or self._collection_name
        if not target_collection:
            logger.error("Search failed: No collection name specified or loaded.")
            return []

        # Ensure the client is loaded for the target collection
        if not self._client or self._collection_name != target_collection:
            if not self.load_collection(target_collection):
                logger.error(f"Search failed: Could not load collection '{target_collection}'.")
                return []

        if not self._client:
             logger.error(f"Search failed: Client not initialized for collection '{target_collection}'.")
             return []

        logger.debug(f"Searching in collection '{target_collection}' for '{query}' (k={k}).")
        try:
            results: List[Document] = self._client.similarity_search(query, k=k)
            logger.info(f"Search completed. Found {len(results)} results.")
            # Format results as specified in the interface
            formatted_results = [
                {"content": doc.page_content, "metadata": doc.metadata}
                for doc in results
            ]
            return formatted_results
        except Exception as e:
            logger.error(f"Search failed in collection '{target_collection}': {e}", exc_info=True)
            return []

    def get_collections(self) -> List[str]:
        """Get a list of all available collection names."""
        if not self._db_client:
            logger.error("Cannot get collections: Base DB client not initialized.")
            return []
        try:
            collections = self._db_client.list_collections()
            return [collection.name for collection in collections]
        except Exception as e:
            logger.error(f"Failed to list collections: {e}", exc_info=True)
            return []

    def get_collection_info(self, collection_name: str) -> Dict[str, Any]:
        """Get information about a specific collection (e.g., count)."""
        if not self._db_client:
            logger.error(f"Cannot get collection info: Base DB client not initialized.")
            return {"error": "DB client not initialized"}
        try:
            collection = self._db_client.get_collection(collection_name)
            count = collection.count()
            logger.info(f"Collection '{collection_name}' count: {count}")
            return {
                "name": collection_name,
                "count": count,
                "persist_directory": self._persist_directory
            }
        except Exception as e:
            # Catch specific exception for collection not found if possible
            logger.error(f"Failed to get info for collection '{collection_name}': {e}", exc_info=True)
            return {"name": collection_name, "error": str(e)}

    def delete_collection(self, collection_name: str) -> None:
        """Delete a specific collection."""
        if not self._db_client:
            logger.error(f"Cannot delete collection '{collection_name}': Base DB client not initialized.")
            raise RuntimeError("DB client not initialized") # Or return False/error dict
        try:
            logger.warning(f"Attempting to delete collection: {collection_name}")
            self._db_client.delete_collection(collection_name)
            logger.info(f"Successfully deleted collection '{collection_name}'.")
            # If the deleted collection was the currently loaded one, reset the client
            if self._collection_name == collection_name:
                self._client = None
                self._collection_name = None
        except Exception as e:
            # Catch specific exception for collection not found? Chroma might raise ValueError
            # For now, log and re-raise or return specific error
            logger.error(f"Failed to delete collection '{collection_name}': {e}", exc_info=True)
            raise # Re-raise the exception 