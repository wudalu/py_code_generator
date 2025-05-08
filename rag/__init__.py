# -*- coding: utf-8 -*-
# rag/__init__.py

from .service import RAGService, create_rag_service
# Optionally export other components if needed directly
# from .retriever import Retriever
# from .prompt_builder import PromptBuilder
# from .llm_interface import LLMInterface

__all__ = [
    "RAGService",
    "create_rag_service",
] 