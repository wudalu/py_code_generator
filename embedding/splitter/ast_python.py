# -*- coding: utf-8 -*-
import ast
import logging
from typing import List

from langchain_core.documents import Document
from .interface import Splitter

logger = logging.getLogger(__name__)

class AstPythonSplitter: # No need to inherit Protocol explicitly
    """Splits Python code based on Abstract Syntax Trees (AST).

    Focuses on splitting into top-level functions and classes.
    """
    # Add __init__ if configuration is needed (e.g., min chunk size, how to handle non-func/class code)
    # def __init__(self, ...):
    #     pass

    def split_document(self, document: Document) -> List[Document]:
        """Split Python document using AST into functions and classes."""
        file_path = document.metadata.get("source", "unknown")
        code_content = document.page_content
        base_metadata = document.metadata.copy() # Important: Copy original metadata
        chunks = []
        lines = code_content.splitlines() # Keep original lines for extraction

        try:
            tree = ast.parse(code_content, filename=file_path)
            logger.debug(f"AST parsing successful for {file_path}")

            for node in tree.body:
                if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
                    start_line = node.lineno
                    end_line = node.end_lineno

                    # Use ast.get_source_segment if available and reliable
                    # Using line numbers as a robust fallback
                    if end_line is None:
                         # Attempt to find the max end line of children if end_line is missing
                         # This can be complex, using line numbers for now
                         logger.warning(f"Node '{node.name}' in {file_path} has missing end_lineno. Estimating based on children or next node might be needed.")
                         # Simple fallback: just use start_line if end is unknown
                         # A better heuristic could be added later.
                         # For now, try to find the max end_lineno from direct children
                         max_child_end = start_line
                         for child in ast.walk(node):
                             if hasattr(child, 'end_lineno') and child.end_lineno is not None:
                                 max_child_end = max(max_child_end, child.end_lineno)
                         end_line = max_child_end

                    # Ensure end_line is within bounds
                    end_line = min(end_line, len(lines))

                    # Extract source segment (1-based to 0-based index)
                    if start_line <= end_line:
                        segment_lines = lines[start_line - 1 : end_line]
                        source_segment = "\n".join(segment_lines)

                        if source_segment.strip(): # Avoid empty chunks
                            metadata = base_metadata.copy()
                            metadata.update({
                                # 'language': 'python', # Already in base metadata?
                                'node_type': 'function' if isinstance(node, ast.FunctionDef) else 'class',
                                'node_name': node.name,
                                'start_line': start_line,
                                'end_line': end_line
                            })
                            chunks.append(Document(page_content=source_segment, metadata=metadata))
                        else:
                             logger.debug(f"Skipping empty segment for node '{node.name}' in {file_path}")
                    else:
                        logger.warning(f"Invalid line numbers for node '{node.name}' in {file_path}: start={start_line}, end={end_line}")
                # else: # Handling top-level code (imports, assignments, etc.)
                    # Option 1: Ignore (current approach)
                    # Option 2: Group consecutive top-level lines into chunks
                    # Option 3: Pass them to fallback splitter
                    # pass

            # If only top-level code exists (no functions/classes found)
            if not chunks and tree.body:
                logger.warning(f"AST splitting for {file_path} found only top-level code. Consider using fallback or specific handling.")
                # Option: Return the whole file as one chunk? Or let pipeline use fallback?
                # Returning empty list signals pipeline to use fallback based on current logic.
                return []

        except SyntaxError as e:
            logger.error(f"AST SyntaxError in {file_path}: {e}. Returning no chunks from AST splitter.")
            return [] # Signal pipeline to use fallback
        except Exception as e:
            logger.error(f"Unexpected error during AST splitting for {file_path}: {e}", exc_info=True)
            return [] # Signal pipeline to use fallback

        logger.info(f"AST splitter generated {len(chunks)} chunks for {file_path}")
        return chunks # Return the list of generated Document chunks 