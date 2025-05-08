# -*- coding: utf-8 -*-
import logging
from typing import List, Dict, Any, Optional

logger = logging.getLogger(__name__)

# Simple cache for loaded models
_reranker_model_cache = {}

class ReRanker:
    """Handles re-ranking of retrieved documents using a CrossEncoder model."""

    def __init__(self, model_name: str):
        """Initializes the ReRanker with a specific CrossEncoder model name."""
        if not model_name:
            raise ValueError("model_name cannot be empty for ReRanker.")
        self.model_name = model_name
        self._model = self._load_model()
        logger.info(f"ReRanker initialized with model: {self.model_name}")

    def _load_model(self):
        """Loads the CrossEncoder model, using a cache."""
        if self.model_name in _reranker_model_cache:
            logger.debug(f"Using cached CrossEncoder model: {self.model_name}")
            return _reranker_model_cache[self.model_name]

        model = None
        try:
            from sentence_transformers.cross_encoder import CrossEncoder
            logger.info(f"Loading CrossEncoder model: {self.model_name}...")
            model = CrossEncoder(self.model_name)
            _reranker_model_cache[self.model_name] = model
            logger.info(f"Successfully loaded CrossEncoder model: {self.model_name}")
            return model
        except ImportError:
            logger.error(
                "sentence-transformers library not found. "
                "Please install it: pip install sentence-transformers"
            )
            raise
        except Exception as e:
            logger.error(f"Failed to load CrossEncoder model '{self.model_name}': {e}", exc_info=True)
            # Ensure model remains None if loading fails
            raise

    def rerank(
        self,
        query: str,
        chunks: List[Dict[str, Any]],
        top_n: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """Re-ranks the provided chunks based on their relevance to the query.

        Args:
            query: The user query string.
            chunks: The list of retrieved document chunks (dictionaries).
            top_n: If specified, only returns the top N reranked results.

        Returns:
            A new list of chunks, sorted by rerank score in descending order,
            optionally truncated to top_n results.
        """
        if not chunks:
            return []
        if self._model is None:
             logger.error("ReRanker model is not loaded. Cannot rerank.")
             # Return original chunks if model failed to load earlier
             return chunks

        logger.debug(f"Reranking {len(chunks)} chunks for query: '{query[:50]}...'")

        # Prepare pairs for the model: (query, chunk_content)
        pairs = [(query, chunk.get('content', '')) for chunk in chunks]

        try:
            # Get scores from the model
            # Ensure model is loaded before predict is called
            if self._model is None:
                 raise RuntimeError("Reranker model is not available.") # Should not happen based on check above, but for safety

            scores = self._model.predict(pairs, show_progress_bar=False)
            logger.debug(f"Received {len(scores)} scores from reranker model.")

            # Add scores to chunks and sort
            # Make a copy to avoid modifying the original list directly if passed by reference
            chunks_with_scores = list(chunks)
            for i, score in enumerate(scores):
                # Ensure score is a standard float if it comes from numpy etc.
                chunks_with_scores[i]['rerank_score'] = float(score)

            # Sort chunks by the new score, highest first
            reranked_chunks = sorted(chunks_with_scores, key=lambda x: x.get('rerank_score', -float('inf')), reverse=True)

            # Optionally truncate to top_n
            if top_n is not None and top_n > 0:
                 final_chunks = reranked_chunks[:top_n]
                 logger.debug(f"Returning top {len(final_chunks)} reranked chunks (truncated from {len(reranked_chunks)})." )
            else:
                 final_chunks = reranked_chunks
                 logger.debug(f"Returning {len(final_chunks)} reranked chunks (no truncation)." )

            return final_chunks

        except Exception as e:
            logger.error(f"Error during reranking with model '{self.model_name}': {e}", exc_info=True)
            # Fallback: return original chunks if reranking fails
            return chunks

__all__ = ["ReRanker"] 