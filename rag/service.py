# -*- coding: utf-8 -*-
import logging
from typing import Iterator, Optional, Dict, Any
import config
import os # Import os

# Assuming component interfaces are importable
from .retriever import Retriever
from .prompt_builder import PromptBuilder
from .llm_interface import LLMInterface, OpenAILLM, ArkLLM, DeepSeekLLM # Import specific implementation for now
from .reranker import ReRanker # Import the new ReRanker
from .tokenizer_util import get_tokenizer # Import the new util
from .optimizer import BaseQueryOptimizer, create_query_optimizer # Import optimizer components

logger = logging.getLogger(__name__)

class RAGService:
    """Orchestrates the RAG process: retrieve, re-rank, build prompt, generate."""
    _retriever: Retriever
    _prompt_builder: PromptBuilder
    _llm_interface: LLMInterface
    _reranker: Optional[ReRanker] # Add optional reranker
    _query_optimizer: Optional[BaseQueryOptimizer] # Add query optimizer

    def __init__(
        self,
        retriever: Retriever,
        prompt_builder: PromptBuilder,
        llm_interface: LLMInterface,
        reranker: Optional[ReRanker] = None, # Make reranker optional
        query_optimizer: Optional[BaseQueryOptimizer] = None # Add query optimizer
    ):
        """Initializes the RAG service with necessary components."""
        self._retriever = retriever
        self._prompt_builder = prompt_builder
        self._llm_interface = llm_interface
        self._reranker = reranker
        self._query_optimizer = query_optimizer
        optimizer_name = type(self._query_optimizer).__name__ if self._query_optimizer else "None"
        reranker_enabled = self._reranker is not None
        logger.info(f"RAGService initialized. Query Optimizer: {optimizer_name}. Reranker enabled: {reranker_enabled}")

    def generate(self, query: str, collection_name: Optional[str] = None, k: Optional[int] = None) -> str:
        """Performs RAG to generate a response for the query."""
        logger.info(f"RAG generate called for query: '{query[:50]}...'")
        # 1. Optimize Query (Optional)
        if self._query_optimizer:
            query_for_retrieval = self._query_optimizer.optimize(query)
        else:
            query_for_retrieval = query

        # Determine target k and collection name, using defaults from config if not provided
        target_k = k if k is not None else config.DEFAULT_RETRIEVAL_K
        target_collection = collection_name if collection_name is not None else config.DEFAULT_COLLECTION_NAME

        # Add logging to check the final collection name being used
        logger.debug(f"Attempting retrieval from collection: '{target_collection}' (Type: {type(target_collection)}), k={target_k}")

        # 2. Retrieve context using the determined parameters
        retrieved_chunks = self._retriever.retrieve(
            query_for_retrieval=query_for_retrieval, 
            k=target_k, 
            collection_name=target_collection # Pass the calculated target_collection
        )
        
        if not retrieved_chunks:
             logger.warning("Retriever returned no chunks.")
             # Handle case with no retrieved chunks (e.g., generate without context or return specific message)
             # For now, let's try to generate with empty context
             processed_chunks = [] # Ensure processed_chunks is empty list
        else:
            # 1.5. Re-rank context (if reranker is enabled)
            if self._reranker:
                try:
                    # Use RERANKER_TOP_N from config to potentially limit input to reranker
                    # Note: Reranker itself also has a top_n param for output limit
                    limit_input_to_reranker = config.RERANKER_TOP_N if config.RERANKER_TOP_N > 0 else len(retrieved_chunks)
                    chunks_to_rerank = retrieved_chunks[:limit_input_to_reranker]
                    logger.debug(f"Passing {len(chunks_to_rerank)} chunks to reranker.")
                    processed_chunks = self._reranker.rerank(query, chunks_to_rerank) # Reranker handles output top_n internally if needed
                except Exception as e:
                    logger.error(f"Reranking failed: {e}. Proceeding with original retrieved chunks.", exc_info=True)
                    processed_chunks = retrieved_chunks # Fallback
            else:
                processed_chunks = retrieved_chunks
                logger.debug("Skipping reranking step.")

        if not processed_chunks:
            logger.warning("No chunks remaining after retrieval/reranking.")
            # Decide how to proceed - generate with empty context?

        # 2. Build Prompt, considering context window limit and using tokenizer
        try:
            model_name = self._llm_interface.get_model_name()
            tokenizer = get_tokenizer(model_name)
            max_tokens = self._llm_interface.get_context_window_size()

            prompt = self._prompt_builder.build_prompt(
                query=query,
                context_chunks=processed_chunks, # Use processed (potentially reranked) chunks
                tokenizer=tokenizer,
                max_context_len=max_tokens
            )
            # logger.debug(f"Generated prompt (first 100 chars): {prompt[:100]}...")
            logger.debug(f"Generated prompt): {prompt}")
        except Exception as e:
            logger.error(f"Failed to build prompt: {e}", exc_info=True)
            return f"Error building prompt: {e}"

        # 3. Generate Response
        try:
            response = self._llm_interface.generate(prompt)
            logger.info("Successfully generated response.")
            return response
        except Exception as e:
            logger.error(f"LLM generation failed: {e}", exc_info=True)
            # Return error message or raise?
            return f"Error generating response: {e}"

    def stream_generate(self, query: str, collection_name: Optional[str] = None, k: Optional[int] = None) -> Iterator[str]:
        """Performs RAG to generate a streaming response."""
        logger.info(f"RAG stream_generate called for query: '{query[:50]}...'")
        # Pre-fetch context and build prompt before starting stream generation
        prompt = None
        try:
            # 1. Retrieve context
            retrieved_chunks = self._retriever.retrieve(query, k=k, collection_name=collection_name)
            if not retrieved_chunks:
                 logger.warning("Retriever returned no chunks for streaming.")
                 # Handle empty retrieval - maybe yield an informative message?
                 # Let prompt building handle empty context for now

            # 1.5. Re-rank context (if reranker is enabled)
            if self._reranker and retrieved_chunks:
                try:
                    limit_input_to_reranker = config.RERANKER_TOP_N if config.RERANKER_TOP_N > 0 else len(retrieved_chunks)
                    chunks_to_rerank = retrieved_chunks[:limit_input_to_reranker]
                    logger.debug(f"Passing {len(chunks_to_rerank)} chunks to reranker for streaming.")
                    processed_chunks = self._reranker.rerank(query, chunks_to_rerank)
                except Exception as e:
                    logger.error(f"Reranking failed for streaming: {e}. Proceeding with original retrieved chunks.", exc_info=True)
                    processed_chunks = retrieved_chunks # Fallback
            else:
                processed_chunks = retrieved_chunks
                logger.debug("Skipping reranking step for streaming.")

            if not processed_chunks:
                 logger.warning("No chunks remaining after retrieval/reranking for streaming.")
                 # Let prompt building handle empty context

            # 2. Build Prompt, considering context window limit and using tokenizer
            model_name = self._llm_interface.get_model_name()
            tokenizer = get_tokenizer(model_name)
            max_tokens = self._llm_interface.get_context_window_size()

            prompt = self._prompt_builder.build_prompt(
                query=query,
                context_chunks=processed_chunks, # Use processed chunks
                tokenizer=tokenizer,
                max_context_len=max_tokens
            )
            logger.debug(f"Generated prompt for streaming (first 100 chars): {prompt[:100]}...")

        except Exception as e:
            logger.error(f"Error preparing for streaming generation: {e}", exc_info=True)
            yield f"\nError preparing response: {e}\n"
            return # Stop the generator

        # 3. Generate Response Stream if prompt was built successfully
        if prompt is not None:
            try:
                logger.info("Streaming response from LLM...")
                yield from self._llm_interface.stream_generate(prompt)
                logger.info("Finished streaming response.")
            except Exception as e:
                logger.error(f"Error during streaming generation: {e}", exc_info=True)
                # Yield an error message? Needs careful handling by the caller.
                yield f"\nError during generation: {e}\n"
        else:
            logger.warning("Prompt was not built, skipping LLM stream generation.")

# --- Helper function to create RAGService instance --- 
# This simplifies instantiation in scripts or servers
from embedding import VectorStore # Import VectorStore type

def create_rag_service(vector_store: VectorStore, llm_config: Optional[Dict[str, Any]] = None) -> RAGService:
     """Creates and returns an initialized RAGService instance based on config."""
     logger.info("Creating RAG Service...")
     retriever = Retriever(vector_store=vector_store)
     prompt_builder = PromptBuilder() # Uses default template logic

     # Initialize LLM Interface
     llm_interface = None
     llm_provider = config.LLM_PROVIDER.lower() if hasattr(config, 'LLM_PROVIDER') else 'openai'

     if llm_provider == 'openai':
         if not config.OPENAI_API_KEY:
             raise ValueError("OPENAI_API_KEY must be set for OpenAI LLM")
         llm_interface = OpenAILLM(
             api_key=config.OPENAI_API_KEY,
             model=config.OPENAI_LLM_MODEL if hasattr(config, 'OPENAI_LLM_MODEL') else 'gpt-3.5-turbo',
             **(llm_config or {})
         )
         logger.info(f"Initialized OpenAI LLM interface for model: {llm_interface.get_model_name()}")
     elif llm_provider == 'ark':
         from .llm_interface import ArkLLM
         ark_api_key = os.environ.get(config.ARK_API_KEY_ENV_VAR)
         if not ark_api_key:
             raise ValueError(f"Environment variable '{config.ARK_API_KEY_ENV_VAR}' must be set for ARK LLM")
         if not config.ARK_BASE_URL:
              raise ValueError("ARK_BASE_URL must be set in config for ARK LLM")
         llm_interface = ArkLLM(
             api_key=ark_api_key,
             base_url=config.ARK_BASE_URL,
             model=config.ARK_LLM_MODEL,
             **(llm_config or {})
         )
         logger.info(f"Initialized ARK LLM interface for model: {llm_interface.get_model_name()}")
     elif llm_provider == 'deepseek':
         deepseek_api_key = os.environ.get(config.DEEPSEEK_API_KEY_ENV_VAR)
         if not deepseek_api_key:
             raise ValueError(f"Environment variable '{config.DEEPSEEK_API_KEY_ENV_VAR}' must be set for DeepSeek LLM")
         # base_url defaults in config, model defaults in config
         llm_interface = DeepSeekLLM(
             api_key=deepseek_api_key,
             base_url=config.DEEPSEEK_BASE_URL,
             model=config.DEEPSEEK_LLM_MODEL,
             **(llm_config or {})
         )
         logger.info(f"Initialized DeepSeek LLM interface for model: {llm_interface.get_model_name()}")
     else:
         raise ValueError(f"Unsupported LLM provider in config: {llm_provider}")

     # Initialize ReRanker (optional)
     reranker = None
     if config.RERANKER_MODEL_NAME:
         try:
             from .reranker import ReRanker # Import locally to keep optional
             reranker = ReRanker(model_name=config.RERANKER_MODEL_NAME)
             logger.info(f"Reranker enabled with model: {config.RERANKER_MODEL_NAME}")
         except Exception as e:
             logger.error(f"Failed to initialize ReRanker: {e}. Reranking will be disabled.", exc_info=True)
             reranker = None # Ensure reranker is None if init fails
     else:
         logger.info("Reranker is disabled (RERANKER_MODEL_NAME not set in config).")

     # Initialize Query Optimizer (depends on LLM)
     query_optimizer = None
     if config.QUERY_OPTIMIZER_TYPE != 'none':
         try:
             query_optimizer = create_query_optimizer(config, llm_interface)
             logger.info(f"Query Optimizer enabled: {type(query_optimizer).__name__}")
         except Exception as e:
             logger.error(f"Failed to initialize Query Optimizer ({config.QUERY_OPTIMIZER_TYPE}): {e}. Query optimization will be disabled.", exc_info=True)
             query_optimizer = None # Ensure query optimizer is None if init fails
     else:
         logger.info(f"Query Optimizer is disabled (QUERY_OPTIMIZER_TYPE is 'none').")

     # Create RAGService instance
     rag_service = RAGService(
         retriever=retriever,
         prompt_builder=prompt_builder,
         llm_interface=llm_interface,
         reranker=reranker, # Pass the initialized reranker (or None)
         query_optimizer=query_optimizer # Pass the initialized query optimizer (or None)
     )
     logger.info("RAG Service created successfully.")
     return rag_service


__all__ = ["RAGService", "create_rag_service"] 