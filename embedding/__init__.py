# -*- coding: utf-8 -*-
# Export the main pipeline class and potentially interfaces/protocols
from .pipeline import EmbeddingPipeline
from .storage.interface import VectorStore
from .splitter.interface import Splitter
# Export the model loader function
from .model_loader import get_embedding_model

# You might also want to export concrete implementations if they are commonly used directly,
# but usually, it's better to configure and inject them via the pipeline.
# from .storage.chroma import ChromaVectorStore
# from .splitter.ast_python import AstPythonSplitter
# from .splitter.fallback import FallbackSplitter

__all__ = [
    "EmbeddingPipeline",
    "VectorStore",
    "Splitter",
    "get_embedding_model",
] 