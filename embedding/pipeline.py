# -*- coding: utf-8 -*-
import logging
import os
from typing import List, Dict, Optional

# Assuming interfaces and types are importable
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from .loader import load_code_files
from .splitter.interface import Splitter
from .storage.interface import VectorStore
# Placeholder for future config object
# from ..config import AppConfig 

logger = logging.getLogger(__name__)

class EmbeddingPipeline:
    """Coordinates the process of loading, splitting, embedding, and storing code files."""

    # Dependencies injected during initialization
    _embedding_model: Embeddings
    _vector_store: VectorStore
    _splitters: Dict[str, Splitter] # Language (e.g., 'python', 'fallback') -> Splitter instance
    # _config: AppConfig # Future config object

    def __init__(self,
                 embedding_model: Embeddings,
                 vector_store: VectorStore,
                 splitters: Dict[str, Splitter],
                 # config: AppConfig # Future config object
                 ):
        """Initializes the embedding pipeline.

        Args:
            embedding_model: An initialized embedding model instance.
            vector_store: An initialized vector store instance.
            splitters: A dictionary mapping language identifiers (e.g., 'python',
                       'fallback') to initialized splitter instances.
            # config: The application configuration object.
        """
        self._embedding_model = embedding_model
        self._vector_store = vector_store
        self._splitters = splitters
        # self._config = config
        logger.info("EmbeddingPipeline initialized.")

    def _get_splitter(self, language: str) -> Splitter:
        """Gets the appropriate splitter for a given language, defaulting to fallback."""
        return self._splitters.get(language, self._splitters['fallback'])

    def _extract_repo_name(self, file_path: str) -> str:
        """Helper to extract repository name from file path (example logic)."""
        repo_name = "unknown_repo"
        try:
            parts = file_path.split(os.sep)
            if "data_src" in parts:
                repo_index = parts.index("data_src")
                if repo_index + 1 < len(parts):
                    repo_name = parts[repo_index + 1]
        except Exception:
            logger.warning(f"Could not extract repo name from path: {file_path}")
        return repo_name

    def process_files(self, file_paths: List[str], collection_name: str) -> Dict:
        """Loads, splits, embeds, and stores a list of code files.

        Args:
            file_paths: List of paths to code files.
            collection_name: Name of the vector store collection to use.

        Returns:
            A dictionary summarizing the processing results.
        """
        logger.info(f"Starting processing for {len(file_paths)} files into collection '{collection_name}'.")
        all_chunks = []
        processed_files_count = 0

        try:
            # 1. Load files
            documents = load_code_files(file_paths)
            if not documents:
                return {"status": "warning", "message": "No documents were successfully loaded.", "file_count": len(file_paths), "chunk_count": 0}
            processed_files_count = len(documents) # Count successfully loaded

            # 2. Split documents
            for doc in documents:
                file_path = doc.metadata.get("source", "unknown")
                language = os.path.splitext(file_path)[1].lstrip('.').lower() or "unknown"
                # Use specific language splitter if available, else fallback
                splitter = self._get_splitter(language if language == 'py' else 'fallback')

                try:
                    file_chunks = splitter.split_document(doc)
                    # Add/Update common metadata after splitting
                    repo_name = self._extract_repo_name(file_path)
                    for chunk in file_chunks:
                        chunk.metadata['repository'] = repo_name
                        chunk.metadata['file_path'] = file_path # Ensure it's present
                        if 'language' not in chunk.metadata:
                            chunk.metadata['language'] = language if language == 'py' else 'unknown' # Or use splitter info?
                    all_chunks.extend(file_chunks)
                    logger.debug(f"Split {file_path} ({language}) into {len(file_chunks)} chunks.")
                except Exception as e:
                    logger.error(f"Failed to split document {file_path}: {e}", exc_info=True)
                    # Optionally, add the original doc as a chunk?
                    # all_chunks.append(doc) # Decide on error handling

            if not all_chunks:
                return {"status": "warning", "message": "Splitting resulted in zero chunks.", "file_count": processed_files_count, "chunk_count": 0}

            # 3. Add to vector store (Embedding happens inside add_documents)
            self._vector_store.add_documents(all_chunks, collection_name=collection_name)

            logger.info(f"Successfully processed {processed_files_count} files, generating {len(all_chunks)} chunks for collection '{collection_name}'.")
            return {
                "status": "success",
                "message": f"Successfully processed {processed_files_count} files.",
                "file_count": processed_files_count,
                "chunk_count": len(all_chunks),
                "collection_name": collection_name
            }

        except Exception as e:
            logger.error(f"Error during embedding pipeline processing: {e}", exc_info=True)
            return {"status": "error", "message": str(e), "file_count": processed_files_count, "chunk_count": len(all_chunks)} 