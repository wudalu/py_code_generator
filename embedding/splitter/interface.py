# -*- coding: utf-8 -*-
from typing import List, Protocol
from langchain_core.documents import Document

class Splitter(Protocol):
    """Interface for splitting a document into smaller chunks."""

    def split_document(self, document: Document) -> List[Document]:
        """Split a single document into a list of smaller documents (chunks)."""
        pass 