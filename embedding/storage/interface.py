# -*- coding: utf-8 -*-
from typing import List, Dict, Any, Protocol, Optional
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

class VectorStore(Protocol):
    """Interface for vector store operations."""

    def __init__(self, persist_directory: str, embedding_function: Embeddings, collection_name: Optional[str] = None):
        """Initialize the vector store."""
        pass

    def add_documents(self, documents: List[Document], collection_name: Optional[str] = None) -> None:
        """Add documents to the specified collection."""
        pass

    def load_collection(self, collection_name: str) -> bool:
        """Load an existing collection. Returns True if successful, False otherwise."""
        pass

    def search(self, query: str, k: int = 5, collection_name: Optional[str] = None) -> List[Dict[str, Any]]:
        """Search for similar documents in the specified collection.

        Returns:
            List of dictionaries, each containing 'content' and 'metadata'.
        """
        pass

    def get_collections(self) -> List[str]:
        """Get a list of all available collection names."""
        pass

    def get_collection_info(self, collection_name: str) -> Dict[str, Any]:
        """Get information about a specific collection (e.g., count)."""
        pass

    # Add other necessary methods if needed, e.g., delete_collection, update_document, etc. 