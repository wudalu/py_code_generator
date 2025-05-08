# -*- coding: utf-8 -*-
import logging
from typing import Protocol, Iterator, Optional, List, Dict
import config # Import config module
import os # Import os for environment variables
from openai import OpenAI, APIConnectionError, RateLimitError # Import OpenAI client and specific errors

logger = logging.getLogger(__name__)

class LLMInterface(Protocol):
    """Interface for interacting with a Large Language Model."""

    def generate(self, prompt: str) -> str:
        """Generate a response from the LLM based on the final formatted prompt."""
        pass

    def stream_generate(self, prompt: str) -> Iterator[str]:
        """Generate a response from the LLM as a stream based on the final formatted prompt."""
        logger.warning("Streaming not implemented by default for this LLM interface.")
        if False: # This makes it a generator
            yield
        pass

    def get_context_window_size(self) -> int:
        """Returns the maximum context window size (in tokens) for the model."""
        return config.DEFAULT_CONTEXT_WINDOW

    def get_model_name(self) -> str:
        """Returns the name of the underlying LLM model."""
        return "unknown_model"

# Example concrete implementation (Skeleton)
class OpenAILLM: # Not inheriting protocol, using duck typing
    _model: str
    _client: OpenAI

    def __init__(self, api_key: str, model: str, **kwargs):
        """Initialize OpenAI client with default base URL."""
        self._model = model
        try:
            self._client = OpenAI(api_key=api_key, **kwargs) # Standard client
            logger.info(f"OpenAILLM initialized for model: {self._model}")
        except Exception as e:
            logger.error(f"Failed to initialize OpenAI client: {e}", exc_info=True)
            raise

    def generate(self, prompt: str) -> str:
        """Generate response using OpenAI API with the provided final prompt."""
        logger.debug(f"Generating response from OpenAI model: {self._model}")
        messages = [{"role": "user", "content": prompt}] # Pass final prompt as user message
        try:
            completion = self._client.chat.completions.create(
                model=self._model,
                messages=messages,
                stream=False
            )
            response = completion.choices[0].message.content
            logger.debug("Successfully generated response from OpenAI.")
            return response or "" # Return empty string if content is None
        except (APIConnectionError, RateLimitError) as e:
             logger.error(f"OpenAI API Error: {e}")
             raise # Re-raise specific OpenAI errors
        except Exception as e:
            logger.error(f"Error during OpenAI generation: {e}", exc_info=True)
            raise

    def stream_generate(self, prompt: str) -> Iterator[str]:
        """Stream response using OpenAI API with the provided final prompt."""
        logger.debug(f"Streaming response from OpenAI model: {self._model}")
        messages = [{"role": "user", "content": prompt}] # Pass final prompt as user message
        try:
            stream = self._client.chat.completions.create(
                model=self._model,
                messages=messages,
                stream=True
            )
            for chunk in stream:
                content = chunk.choices[0].delta.content
                if content:
                    yield content
            logger.debug("Finished streaming response from OpenAI.")
        except (APIConnectionError, RateLimitError) as e:
             logger.error(f"OpenAI API Streaming Error: {e}")
             raise # Re-raise specific OpenAI errors
        except Exception as e:
            logger.error(f"Error during OpenAI streaming: {e}", exc_info=True)
            raise

    def get_context_window_size(self) -> int:
        """Gets the context window size for the specific OpenAI model."""
        size = config.MODEL_CONTEXT_WINDOWS.get(self._model, config.DEFAULT_CONTEXT_WINDOW)
        if size == config.DEFAULT_CONTEXT_WINDOW:
            logger.warning(
                f"Context window size for model '{self._model}' not found in config. "
                f"Falling back to default: {config.DEFAULT_CONTEXT_WINDOW} tokens."
            )
        return size

    def get_model_name(self) -> str:
        """Returns the configured OpenAI model name."""
        return self._model

# --- ARK LLM Implementation (ByteDance) --- 
class ArkLLM:
    _model: str
    _client: OpenAI

    def __init__(self, api_key: str, model: str, base_url: str, **kwargs):
        """Initialize OpenAI client with custom ARK base URL."""
        self._model = model
        if not base_url:
            raise ValueError("base_url must be provided for ArkLLM")
        try:
            # Use the OpenAI library but configure it for ARK's endpoint
            self._client = OpenAI(api_key=api_key, base_url=base_url, **kwargs)
            logger.info(f"ArkLLM initialized for model: {self._model} at {base_url}")
        except Exception as e:
            logger.error(f"Failed to initialize ARK client: {e}", exc_info=True)
            raise

    # generate method accepts the final prompt string
    def generate(self, prompt: str) -> str:
        """Generate response using ARK API with the provided final prompt."""
        logger.debug(f"Generating response from ARK model: {self._model}")
        # The final prompt from PromptBuilder is used as the user message
        messages = [{"role": "user", "content": prompt}]
        try:
            completion = self._client.chat.completions.create(
                model=self._model,
                messages=messages,
                stream=False
            )
            response = completion.choices[0].message.content
            logger.debug("Successfully generated response from ARK.")
            return response or "" # Return empty string if content is None
        except (APIConnectionError, RateLimitError) as e:
             logger.error(f"ARK API Error: {e}")
             raise
        except Exception as e:
            logger.error(f"Error during ARK generation: {e}", exc_info=True)
            raise

    # stream_generate method accepts the final prompt string
    def stream_generate(self, prompt: str) -> Iterator[str]:
        """Stream response using ARK API with the provided final prompt."""
        logger.debug(f"Streaming response from ARK model: {self._model}")
        messages = [{"role": "user", "content": prompt}]
        try:
            stream = self._client.chat.completions.create(
                model=self._model,
                messages=messages,
                stream=True
            )
            for chunk in stream:
                # Handle potential empty choices list
                if chunk.choices and chunk.choices[0].delta:
                     content = chunk.choices[0].delta.content
                     if content:
                         yield content
            logger.debug("Finished streaming response from ARK.")
        except (APIConnectionError, RateLimitError) as e:
             logger.error(f"ARK API Streaming Error: {e}")
             raise
        except Exception as e:
            logger.error(f"Error during ARK streaming: {e}", exc_info=True)
            raise

    def get_context_window_size(self) -> int:
        """Gets the context window size for the specific ARK model."""
        # Attempt to find in config, otherwise use default
        size = config.MODEL_CONTEXT_WINDOWS.get(self._model, config.DEFAULT_CONTEXT_WINDOW)
        if size == config.DEFAULT_CONTEXT_WINDOW:
            logger.warning(
                f"Context window size for ARK model '{self._model}' not found in config. "
                f"Falling back to default: {config.DEFAULT_CONTEXT_WINDOW} tokens."
            )
        return size

    def get_model_name(self) -> str:
        """Returns the configured ARK model name."""
        return self._model

# --- DeepSeek LLM Implementation --- 
class DeepSeekLLM:
    _model: str
    _client: OpenAI

    def __init__(self, api_key: str, model: str, base_url: str, **kwargs):
        """Initialize OpenAI client with custom DeepSeek base URL."""
        self._model = model
        if not base_url:
            # Use default if not explicitly provided, matching config logic
            base_url = "https://api.deepseek.com"
        try:
            self._client = OpenAI(api_key=api_key, base_url=base_url, **kwargs)
            logger.info(f"DeepSeekLLM initialized for model: {self._model} at {base_url}")
        except Exception as e:
            logger.error(f"Failed to initialize DeepSeek client: {e}", exc_info=True)
            raise

    # generate method accepts the final prompt string
    def generate(self, prompt: str) -> str:
        """Generate response using DeepSeek API with the provided final prompt."""
        logger.debug(f"Generating response from DeepSeek model: {self._model}")
        # The final prompt from PromptBuilder is used as the user message
        # Deepseek might prefer a system prompt, but we keep it user-only for consistency with interface
        messages = [{"role": "user", "content": prompt}]
        try:
            completion = self._client.chat.completions.create(
                model=self._model,
                messages=messages,
                stream=False
            )
            response = completion.choices[0].message.content
            logger.debug("Successfully generated response from DeepSeek.")
            return response or "" # Return empty string if content is None
        except (APIConnectionError, RateLimitError) as e:
             logger.error(f"DeepSeek API Error: {e}")
             raise
        except Exception as e:
            logger.error(f"Error during DeepSeek generation: {e}", exc_info=True)
            raise

    # stream_generate method accepts the final prompt string
    def stream_generate(self, prompt: str) -> Iterator[str]:
        """Stream response using DeepSeek API with the provided final prompt."""
        logger.debug(f"Streaming response from DeepSeek model: {self._model}")
        messages = [{"role": "user", "content": prompt}]
        try:
            stream = self._client.chat.completions.create(
                model=self._model,
                messages=messages,
                stream=True
            )
            for chunk in stream:
                # Handle potential empty choices list
                if chunk.choices and chunk.choices[0].delta:
                     content = chunk.choices[0].delta.content
                     if content:
                         yield content
            logger.debug("Finished streaming response from DeepSeek.")
        except (APIConnectionError, RateLimitError) as e:
             logger.error(f"DeepSeek API Streaming Error: {e}")
             raise
        except Exception as e:
            logger.error(f"Error during DeepSeek streaming: {e}", exc_info=True)
            raise

    def get_context_window_size(self) -> int:
        """Gets the context window size for the specific DeepSeek model."""
        # Attempt to find in config, otherwise use default
        size = config.MODEL_CONTEXT_WINDOWS.get(self._model, config.DEFAULT_CONTEXT_WINDOW)
        if size == config.DEFAULT_CONTEXT_WINDOW:
            logger.warning(
                f"Context window size for DeepSeek model '{self._model}' not found in config. "
                f"Falling back to default: {config.DEFAULT_CONTEXT_WINDOW} tokens. Check DeepSeek documentation for exact size."
            )
        return size

    def get_model_name(self) -> str:
        """Returns the configured DeepSeek model name."""
        return self._model

# Add other implementations like HuggingFaceLLM etc. as needed

__all__ = ["LLMInterface", "OpenAILLM", "ArkLLM", "DeepSeekLLM"] # Add DeepSeekLLM