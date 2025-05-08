# -*- coding: utf-8 -*-
import logging
from abc import ABC, abstractmethod
from rag.llm_interface import LLMInterface  # Assuming LLMInterface is the base class/protocol
import config # To access configuration like templates if needed later

logger = logging.getLogger(__name__)

# --- Base Class ---

class BaseQueryOptimizer(ABC):
    """Abstract base class for query optimizers."""

    @abstractmethod
    def optimize(self, query: str) -> str:
        """
        Optimizes the given query for retrieval.

        Args:
            query: The original user query.

        Returns:
            A potentially modified query string intended for vector store retrieval.
        """
        pass

# --- Implementations ---

class NoOpOptimizer(BaseQueryOptimizer):
    """Optimizer that performs no operation (returns the original query)."""

    def optimize(self, query: str) -> str:
        logger.debug("Using NoOpOptimizer. Query remains unchanged.")
        return query

class HyDEOptimizer(BaseQueryOptimizer):
    """
    Optimizer using Hypothetical Document Embeddings (HyDE).
    Generates a hypothetical document based on the query and returns it
    for embedding and retrieval.
    """
    DEFAULT_HYDE_PROMPT = (
        "You are a helpful assistant. Given the user query below, generate a concise hypothetical code snippet "
        "or technical explanation that is likely to contain the answer to the query. "
        "Focus on relevance and keywords. Output only the hypothetical text, nothing else.\n\n"
        "USER QUERY: {query}\n\nHypothetical Answer:"
    )


    def __init__(self, llm_interface: LLMInterface, prompt_template: str = DEFAULT_HYDE_PROMPT):
        """
        Initializes the HyDE optimizer.

        Args:
            llm_interface: An instance of LLMInterface to generate the hypothetical document.
            prompt_template: The prompt template to use for generating the hypothetical document.
                               Must contain '{query}'.
        """
        if "{query}" not in prompt_template:
            raise ValueError("HyDE prompt_template must include '{query}'.")
        self._llm = llm_interface
        self._prompt_template = prompt_template
        logger.info(f"HyDEOptimizer initialized with LLM: {type(llm_interface).__name__}")


    def optimize(self, query: str) -> str:
        """Generates a hypothetical document for the query."""
        logger.debug(f"Using HyDEOptimizer to generate hypothetical document for query: '{query[:50]}...'")
        hyde_prompt = self._prompt_template.format(query=query)

        try:
            # Generate the hypothetical document using the LLM
            # We assume the LLM interface can handle simple generation without context
            hypothetical_document = self._llm.generate(prompt=hyde_prompt)
            logger.debug(f"Generated hypothetical document: '{hypothetical_document[:100]}...'")
            # Return the generated document text. This text will be embedded for retrieval.
            # The original query is still used later for the final prompt and reranking.
            return hypothetical_document
        except Exception as e:
            logger.error(f"HyDEOptimizer failed to generate hypothetical document: {e}", exc_info=True)
            # Fallback to original query if generation fails
            logger.warning("HyDE generation failed. Falling back to original query for retrieval.")
            return query

# --- Factory Function ---

def create_query_optimizer(config_module: object = config, llm_interface: LLMInterface | None = None) -> BaseQueryOptimizer:
    """
    Factory function to create a query optimizer based on configuration.

    Args:
        config_module: The configuration module (e.g., imported config.py).
        llm_interface: The LLM interface instance, required by some optimizers (like HyDE).

    Returns:
        An instance of a BaseQueryOptimizer subclass.

    Raises:
        ValueError: If an unknown optimizer type is specified or if a required
                    dependency (like llm_interface for HyDE) is missing.
    """
    optimizer_type = getattr(config_module, 'QUERY_OPTIMIZER_TYPE', 'none').lower()
    logger.info(f"Attempting to create query optimizer of type: {optimizer_type}")

    if optimizer_type == "hyde":
        if llm_interface is None:
            raise ValueError("LLMInterface instance is required for HyDEOptimizer.")
        # TODO: Add configuration for custom HyDE prompt template later if needed
        return HyDEOptimizer(llm_interface=llm_interface)
    elif optimizer_type == "none":
        return NoOpOptimizer()
    # Add other optimizers here (e.g., decomposition)
    # elif optimizer_type == "decomposition":
    #     if llm_interface is None:
    #         raise ValueError("LLMInterface instance is required for DecompositionOptimizer.")
    #     return DecompositionOptimizer(llm_interface=llm_interface)
    else:
        raise ValueError(f"Unknown query optimizer type specified in config: {optimizer_type}") 