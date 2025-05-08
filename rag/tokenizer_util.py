# -*- coding: utf-8 -*-
import logging
import tiktoken

logger = logging.getLogger(__name__)

# Cache for loaded tokenizers
_tokenizer_cache = {}

# Default encoding to use if model-specific fails or model is unknown
# cl100k_base is used by gpt-4, gpt-3.5-turbo, text-embedding-ada-002
DEFAULT_ENCODING = "cl100k_base"

def get_tokenizer(model_name: str) -> tiktoken.Encoding:
    """Gets a tiktoken tokenizer for the given model name.

    Uses a cache to avoid reloading tokenizers. Falls back to a default
    encoding if the specific model encoding is not found or fails to load.

    Args:
        model_name: The name of the model (e.g., "gpt-4", "gpt-3.5-turbo").

    Returns:
        A tiktoken Encoding object.
    """
    if model_name in _tokenizer_cache:
        return _tokenizer_cache[model_name]

    encoding: tiktoken.Encoding
    try:
        encoding = tiktoken.encoding_for_model(model_name)
        logger.debug(f"Loaded tiktoken encoding for model: {model_name}")
    except KeyError:
        logger.warning(
            f"Model '{model_name}' not found by tiktoken. "
            f"Falling back to default encoding '{DEFAULT_ENCODING}'. "
            f"Token counts may be inaccurate."
        )
        # Fallback to default encoding
        try:
            encoding = tiktoken.get_encoding(DEFAULT_ENCODING)
        except Exception as e:
            logger.error(f"Failed to load default tiktoken encoding '{DEFAULT_ENCODING}': {e}", exc_info=True)
            raise RuntimeError(f"Could not load tiktoken default encoding '{DEFAULT_ENCODING}'") from e
    except Exception as e:
        logger.error(f"An unexpected error occurred while getting tokenizer for model '{model_name}': {e}", exc_info=True)
        # Attempt fallback for any other error during loading
        try:
            encoding = tiktoken.get_encoding(DEFAULT_ENCODING)
            logger.warning(f"Unexpected error loading tokenizer for '{model_name}', falling back to '{DEFAULT_ENCODING}'.")
        except Exception as fallback_e:
             logger.error(f"Failed to load default tiktoken encoding '{DEFAULT_ENCODING}' during fallback: {fallback_e}", exc_info=True)
             raise RuntimeError(f"Could not load tiktoken default encoding '{DEFAULT_ENCODING}'") from fallback_e

    _tokenizer_cache[model_name] = encoding
    return encoding

__all__ = ["get_tokenizer"] 