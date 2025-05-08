# -*- coding: utf-8 -*-
from typing import List, Optional
import logging

from langchain_core.documents import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from .interface import Splitter
import config

logger = logging.getLogger(__name__)

class FallbackSplitter:
    """A wrapper around RecursiveCharacterTextSplitter for fallback.

    Implements the Splitter interface.
    """
    _splitter: RecursiveCharacterTextSplitter

    def __init__(self, 
                 chunk_size: Optional[int] = None, 
                 chunk_overlap: Optional[int] = None, 
                 **kwargs):
        """Initializes the fallback splitter.

        Args:
            chunk_size: Target size of chunks. Defaults to config.DEFAULT_CHUNK_SIZE.
            chunk_overlap: Overlap between chunks. Defaults to config.DEFAULT_CHUNK_OVERLAP.
            **kwargs: Other arguments for RecursiveCharacterTextSplitter.
        """
        final_chunk_size = chunk_size if chunk_size is not None else config.DEFAULT_CHUNK_SIZE
        final_chunk_overlap = chunk_overlap if chunk_overlap is not None else config.DEFAULT_CHUNK_OVERLAP

        self._splitter = RecursiveCharacterTextSplitter(
            chunk_size=final_chunk_size,
            chunk_overlap=final_chunk_overlap,
            **kwargs
        )
        logger.debug(f"FallbackSplitter initialized with chunk_size={final_chunk_size}, chunk_overlap={final_chunk_overlap}")

    def split_document(self, document: Document) -> List[Document]:
        """Split document using RecursiveCharacterTextSplitter."""
        file_path = document.metadata.get("source", "unknown")
        logger.debug(f"Using fallback splitter for {file_path}")
        try:
            return self._splitter.split_documents([document])
        except Exception as e:
            logger.error(f"Error during fallback splitting for {file_path}: {e}", exc_info=True)
            # Return the original document as a single chunk in case of error?
            # Or return empty list?
            return [document] # Safest option might be to return the original 