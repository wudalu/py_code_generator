# -*- coding: utf-8 -*-
import logging
import config # Assuming config is importable from project root

from langchain_core.embeddings import Embeddings

logger = logging.getLogger(__name__)

def get_embedding_model() -> Embeddings:
    """Loads and returns an embedding model instance based on configuration."""
    provider = config.EMBEDDING_PROVIDER.lower()
    logger.info(f"Attempting to load embedding model from provider: {provider}")

    if provider == 'huggingface':
        try:
            from langchain_community.embeddings import HuggingFaceEmbeddings
            logger.info(f"Using HuggingFace embedding model: {config.HUGGINGFACE_MODEL_NAME}")
            return HuggingFaceEmbeddings(model_name=config.HUGGINGFACE_MODEL_NAME)
        except ImportError:
             logger.error("langchain-community package not installed. Please install it to use HuggingFace embeddings.")
             raise
        except Exception as e:
            logger.error(f"Failed to load HuggingFace model '{config.HUGGINGFACE_MODEL_NAME}': {e}", exc_info=True)
            raise

    elif provider == 'openai':
        if not config.OPENAI_API_KEY:
            logger.error("OPENAI_API_KEY must be set in config or environment for OpenAI embeddings.")
            raise ValueError("Missing OpenAI API Key")
        try:
            from langchain_openai import OpenAIEmbeddings
            logger.info(f"Using OpenAI embedding model: {config.OPENAI_EMBEDDING_MODEL}")
            return OpenAIEmbeddings(
                openai_api_key=config.OPENAI_API_KEY,
                model=config.OPENAI_EMBEDDING_MODEL
            )
        except ImportError:
            logger.error("langchain-openai package not installed. Please install it to use OpenAI embeddings.")
            raise
        except Exception as e:
             logger.error(f"Failed to initialize OpenAI embeddings: {e}", exc_info=True)
             raise

    elif provider == 'fake':
        try:
            from langchain_community.embeddings import FakeEmbeddings
            logger.info("Using Fake embeddings for testing.")
            # Consider making the size configurable as well
            return FakeEmbeddings(size=768) # Example size
        except ImportError:
             logger.error("langchain-community package not installed. Please install it to use Fake embeddings.")
             raise
        except Exception as e:
             logger.error(f"Failed to initialize Fake embeddings: {e}", exc_info=True)
             raise

    else:
        logger.error(f"Unsupported embedding provider in config: {config.EMBEDDING_PROVIDER}")
        raise ValueError(f"Unsupported embedding provider: {config.EMBEDDING_PROVIDER}")

__all__ = ["get_embedding_model"] 