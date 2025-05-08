# -*- coding: utf-8 -*-
import os
from dotenv import load_dotenv

# Load environment variables from .env file, if it exists
load_dotenv()

# --- Embedding Configuration ---
# Directory to persist ChromaDB data
PERSIST_DIRECTORY = os.environ.get("PERSIST_DIRECTORY", "./chroma_db_store")

# Default collection name for ChromaDB
DEFAULT_COLLECTION_NAME = os.environ.get("DEFAULT_COLLECTION_NAME", "code_collection_v1")

# Embedding model configuration
# Options: 'huggingface' or 'openai' or 'fake' (for testing)
EMBEDDING_PROVIDER = os.environ.get("EMBEDDING_PROVIDER", "huggingface")
# Name of the HuggingFace model (if provider is huggingface)
HUGGINGFACE_MODEL_NAME = os.environ.get("HUGGINGFACE_MODEL_NAME", "sentence-transformers/all-MiniLM-L6-v2")
# Name of the OpenAI model (if provider is openai)
OPENAI_EMBEDDING_MODEL = os.environ.get("OPENAI_EMBEDDING_MODEL", "text-embedding-ada-002")

# Splitting configuration
DEFAULT_CHUNK_SIZE = int(os.environ.get("DEFAULT_CHUNK_SIZE", 1000))
DEFAULT_CHUNK_OVERLAP = int(os.environ.get("DEFAULT_CHUNK_OVERLAP", 100))

# --- Logging Configuration ---
LOG_DIR = "log"
LOG_FILE = os.path.join(LOG_DIR, os.environ.get("LOG_FILE_NAME", "app.log"))
LOG_LEVEL = os.environ.get("LOG_LEVEL", "INFO").upper()
LOG_FORMAT = os.environ.get("LOG_FORMAT", '%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# --- RAG Configuration (Example) ---
# Number of relevant chunks to retrieve
DEFAULT_RETRIEVAL_K = int(os.environ.get("DEFAULT_RETRIEVAL_K", 5))


# --- MCP Server Configuration (Example) ---
# Host and port for MCP SSE server
MCP_SSE_HOST = os.environ.get("MCP_SSE_HOST", "127.0.0.1")
MCP_SSE_PORT = int(os.environ.get("MCP_SSE_PORT", 8001))

# --- LLM Configuration --- (Provider set below)

# --- OpenAI LLM Configuration (if LLM_PROVIDER is 'openai') ---
OPENAI_API_KEY_ENV_VAR = os.environ.get("OPENAI_API_KEY_ENV_VAR", "OPENAI_API_KEY")
OPENAI_API_KEY = os.environ.get(OPENAI_API_KEY_ENV_VAR) # Load the key value
OPENAI_LLM_MODEL = os.environ.get("OPENAI_LLM_MODEL", "gpt-3.5-turbo")

# --- DeepSeek LLM Configuration (if LLM_PROVIDER is 'deepseek') ---
DEEPSEEK_BASE_URL = os.environ.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
DEEPSEEK_API_KEY_ENV_VAR = os.environ.get("DEEPSEEK_API_KEY_ENV_VAR", "DEEPSEEK_API_KEY") # Env var *name*
DEEPSEEK_API_KEY = os.environ.get(DEEPSEEK_API_KEY_ENV_VAR) # Load the key value
DEEPSEEK_LLM_MODEL = os.environ.get("DEEPSEEK_LLM_MODEL", "deepseek-chat") # Default model

# --- Select LLM Provider ---
# Provider for the generation model (e.g., 'openai', 'deepseek', 'ark')
LLM_PROVIDER = os.environ.get("LLM_PROVIDER", "deepseek") # Default to DeepSeek


# --- Reranker Configuration ---
# Set to a model name from sentence-transformers/huggingface for cross-encoder reranking
# Examples: 'cross-encoder/ms-marco-MiniLM-L-6-v2', 'cross-encoder/ms-marco-MiniLM-L-12-v2'
# Set to None or empty string to disable reranking
RERANKER_MODEL_NAME = os.environ.get("RERANKER_MODEL_NAME", "cross-encoder/ms-marco-MiniLM-L-6-v2")
# RERANKER_MODEL_NAME = os.environ.get("RERANKER_MODEL_NAME", None) # Example: disable reranking
# Number of top results from retriever to pass to reranker
# Set to 0 or None to pass all retrieved results
RERANKER_TOP_N = int(os.environ.get("RERANKER_TOP_N", 0)) # Default: rerank all retrieved

# Add other configuration sections as needed (e.g., Logging, SFT, API Keys etc.)

# --- Validation (Optional but recommended) ---
# This validation is now primarily handled in the test script, but could be added here too.
# Example:
# if LLM_PROVIDER == 'openai' and not OPENAI_API_KEY:
#     raise ValueError("OPENAI_API_KEY must be set when LLM_PROVIDER is 'openai'.")


# --- Model Context Window Sizes (in tokens) ---
# Source: Primarily from OpenAI documentation and common knowledge
# Note: Actual usable tokens might be slightly less due to model overhead.
# Using character count as a proxy for token count in prompt building is an approximation.
MODEL_CONTEXT_WINDOWS = {
    # GPT-4 Turbo
    "gpt-4-turbo": 128000,
    "gpt-4-turbo-2024-04-09": 128000,
    "gpt-4-turbo-preview": 128000,
    "gpt-4-0125-preview": 128000,
    "gpt-4-1106-preview": 128000,
    # GPT-4
    "gpt-4": 8192,
    "gpt-4-0613": 8192,
    "gpt-4-32k": 32768,
    "gpt-4-32k-0613": 32768,
    # GPT-3.5 Turbo
    "gpt-3.5-turbo-0125": 16385,
    "gpt-3.5-turbo": 16385, # Often points to the latest 16k version
    "gpt-3.5-turbo-1106": 16385,
    "gpt-3.5-turbo-instruct": 4096, # Instruct models have smaller context
    "gpt-3.5-turbo-16k": 16385, # Older 16k model alias
    "gpt-3.5-turbo-0613": 4096, # Older 4k model
    "gpt-3.5-turbo-16k-0613": 16385, # Older 16k model
    # Add other models as needed
    "ep-20250415151742-n8ctj": 32768, # Added ARK model context size
    "deepseek-chat": 65536, # Added DeepSeek model context size
    "deepseek-coder": 16384, # Example DeepSeek Coder context size
}
DEFAULT_CONTEXT_WINDOW = 4096 # Fallback if model not found

# Simplified print at the end to avoid printing sensitive keys
print(f"Config loaded: PERSIST_DIRECTORY={PERSIST_DIRECTORY}, EMBEDDING_PROVIDER={EMBEDDING_PROVIDER}, RERANKER_MODEL={RERANKER_MODEL_NAME}, LLM_PROVIDER={LLM_PROVIDER}")

# --- Query Optimizer Configuration ---
# Options: "none", "hyde" (add others like "decomposition")
QUERY_OPTIMIZER_TYPE = os.environ.get("QUERY_OPTIMIZER_TYPE", "none").lower() 