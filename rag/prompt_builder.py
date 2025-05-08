# -*- coding: utf-8 -*-
import logging
import os
from typing import List, Dict, Any, Optional
import re # Keep re for analyzing template keys
import tiktoken # Import tiktoken

logger = logging.getLogger(__name__)

DEFAULT_TEMPLATE_DIR = os.path.join(os.path.dirname(__file__), 'templates')
DEFAULT_TEMPLATE_FILE = os.path.join(DEFAULT_TEMPLATE_DIR, 'default_code_gen.tmpl')

# --- Default Prompt Template for Code Generation ---
# Removed DEFAULT_CODE_GEN_TEMPLATE constant

class PromptBuilder:
    """Constructs prompts for the LLM using retrieved context and user query."""
    _template: str
    _template_keys: set # Store expected keys like {'context', 'query'}
    # Remove _template_base_len as it depends on tokenizer

    def __init__(self, template_path: str = DEFAULT_TEMPLATE_FILE):
        """Initializes the PromptBuilder, loading the template from the specified path."""
        try:
            # Ensure the templates directory exists if using the default path
            if template_path == DEFAULT_TEMPLATE_FILE and not os.path.exists(DEFAULT_TEMPLATE_DIR):
                os.makedirs(DEFAULT_TEMPLATE_DIR)
                logger.info(f"Created template directory: {DEFAULT_TEMPLATE_DIR}")
                # Optionally create a default template file if it doesn't exist
                if not os.path.exists(DEFAULT_TEMPLATE_FILE):
                    # You might want to write the old default content here as a fallback
                    default_content = """# ROLE: You are an expert Python programming assistant.
# TASK: Generate Python code based on the user's query and the provided context.
# CONTEXT:
{context}
# USER QUERY:
{query}
# RESPONSE (Python Code only):
```python
"""
                    with open(DEFAULT_TEMPLATE_FILE, 'w', encoding='utf-8') as f:
                        f.write(default_content)
                    logger.info(f"Created default template file: {DEFAULT_TEMPLATE_FILE}")


            with open(template_path, 'r', encoding='utf-8') as f:
                self._template = f.read()
            logger.info(f"PromptBuilder initialized with template from: {template_path}")
            # Analyze template keys
            self._analyze_template_keys()
        except FileNotFoundError:
            logger.error(f"Template file not found at {template_path}. PromptBuilder will use a basic fallback.")
            self._template = "Context:\n{context}\n\nQuery:\n{query}"
            self._analyze_template_keys() # Analyze fallback template
        except Exception as e:
            logger.error(f"Error loading or analyzing template from {template_path}: {e}", exc_info=True)
            self._template = "Context:\n{context}\n\nQuery:\n{query}" # Fallback
            self._analyze_template_keys() # Analyze fallback template
        # Add validation for template placeholders here if needed

    def _analyze_template_keys(self):
        """Helper to find keys in the template."""
        keys = set(re.findall(r'\{([^}]+)\}', self._template))
        self._template_keys = keys
        if 'context' not in keys or 'query' not in keys:
             logger.warning(f"Template might be missing required keys 'context' or 'query'. Found: {keys}")

    def _format_context(
        self,
        retrieved_chunks: List[Dict[str, Any]],
        tokenizer: tiktoken.Encoding,
        available_tokens_for_context: Optional[int] = None
    ) -> str:
        """Formats the retrieved chunks into a string context, respecting token limits."""
        if not retrieved_chunks:
            return "No relevant code snippets found in the knowledge base."

        context_parts = []
        current_tokens = 0
        limit_reached = False
        chunks_used = 0

        # Estimate header tokens (encode once)
        header = "--- Relevant Code Snippets ---"
        header_tokens = len(tokenizer.encode(header))
        context_parts.append(header)
        current_tokens += header_tokens

        for i, chunk in enumerate(retrieved_chunks):
            metadata = chunk.get('metadata', {})
            source = metadata.get('file_path', 'unknown source')
            node_info = f" (Node: {metadata.get('node_type', '')} {metadata.get('node_name', '')})" if metadata.get('node_name') else ""
            content = chunk.get('content', '')

            # Construct the snippet string elements for token calculation
            snippet_header = f"\n--- Snippet {i+1} (from {source}{node_info}) ---\n"
            snippet_code_prefix = "```python\n"
            snippet_code_suffix = "\n```\n"

            # Encode parts separately for better accuracy
            header_part_tokens = len(tokenizer.encode(snippet_header))
            prefix_tokens = len(tokenizer.encode(snippet_code_prefix))
            content_tokens = len(tokenizer.encode(content))
            suffix_tokens = len(tokenizer.encode(snippet_code_suffix))
            snippet_total_tokens = header_part_tokens + prefix_tokens + content_tokens + suffix_tokens

            if available_tokens_for_context is not None and (current_tokens + snippet_total_tokens) > available_tokens_for_context:
                logger.warning(
                    f"Context truncated: Stopping after {chunks_used} snippets ({current_tokens} tokens) due to token limit "
                    f"(available for context: {available_tokens_for_context} tokens). "
                    f"Consider using fewer retrieval results (k) or a model with larger context."
                )
                limit_reached = True
                break # Stop adding chunks

            # Add the snippet parts to the context
            context_parts.append(snippet_header)
            context_parts.append(snippet_code_prefix)
            context_parts.append(content)
            context_parts.append(snippet_code_suffix)
            current_tokens += snippet_total_tokens
            chunks_used += 1

        return "".join(context_parts).strip()

    def build_prompt(
        self,
        query: str,
        context_chunks: List[Dict[str, Any]],
        tokenizer: tiktoken.Encoding,
        max_context_len: Optional[int] = None
    ) -> str:
        """Builds the final prompt string, truncating context based on token limits."""
        if 'context' not in self._template_keys or 'query' not in self._template_keys:
            logger.error(f"Template is missing required keys ('context', 'query'). Cannot build prompt reliably.")
            formatted_context = self._format_context(context_chunks, tokenizer) # Format without limit for fallback
            return f"Context:\n{formatted_context}\n\nQuery:\n{query}" # Basic fallback

        # Calculate template base length in tokens (excluding placeholders)
        base_template = self._template
        for key in self._template_keys:
            base_template = base_template.replace(f'{{{key}}}', '')
        template_base_tokens = len(tokenizer.encode(base_template))

        query_tokens = len(tokenizer.encode(query))

        available_tokens_for_context = None
        if max_context_len is not None:
            available_tokens_for_context = max_context_len - template_base_tokens - query_tokens
            # Add a small buffer (e.g., 10 tokens) for safety/formatting nuances
            available_tokens_for_context -= 10

            if available_tokens_for_context < 0:
                logger.warning(
                    f"Query tokens ({query_tokens}) and template base tokens ({template_base_tokens}) "
                    f"already exceed max context tokens ({max_context_len}). No space for context."
                )
                available_tokens_for_context = 0 # Allow formatting to return empty context message
            else:
                 logger.debug(f"Calculated available tokens for context: {available_tokens_for_context}")

        formatted_context = self._format_context(
            context_chunks,
            tokenizer,
            available_tokens_for_context
        )

        try:
            # Format the final prompt
            final_prompt = self._template.format(context=formatted_context, query=query)
            final_prompt_tokens = len(tokenizer.encode(final_prompt))

            # Final length check
            if max_context_len is not None and final_prompt_tokens > max_context_len:
                 logger.warning(
                     f"Final prompt token count ({final_prompt_tokens}) slightly exceeds max limit ({max_context_len}) "
                     f"despite context truncation. This might happen due to tokenization nuances."
                 )
                 # Optional: Truncate the final prompt string itself if absolutely necessary
                 # final_prompt = tokenizer.decode(tokenizer.encode(final_prompt)[:max_context_len])

            formatted_context_tokens = len(tokenizer.encode(formatted_context))
            logger.debug(
                f"Built prompt. Final tokens: {final_prompt_tokens}. Context tokens: {formatted_context_tokens}. Query tokens: {query_tokens}. Base tokens: {template_base_tokens}"
            )
            return final_prompt
        except KeyError as e:
            logger.error(f"Failed to format prompt template. Missing key: {e}. Template: {self._template[:100]}...")
            return f"Context:\n{formatted_context}\n\nQuery:\n{query}" # Basic fallback
        except Exception as e:
            logger.error(f"Error building prompt: {e}", exc_info=True)
            raise

__all__ = ["PromptBuilder"] 