# -*- coding: utf-8 -*-
import os
import logging
from typing import List

from langchain_community.document_loaders import TextLoader
from langchain_core.documents import Document

logger = logging.getLogger(__name__)

def load_code_files(file_paths: List[str]) -> List[Document]:
    """Loads code files from a list of paths.

    Handles different encodings and logs errors.

    Args:
        file_paths: List of paths to code files.

    Returns:
        List of loaded LangChain Document objects.
    """
    logger.info(f"开始加载{len(file_paths)}个代码文件")
    loaded_docs = []

    for file_path in file_paths:
        try:
            # Check existence and readability
            if not os.path.exists(file_path):
                logger.error(f"文件不存在: {file_path}")
                continue
            if not os.access(file_path, os.R_OK):
                logger.error(f"文件无法读取: {file_path}")
                continue

            # Try different encodings
            encodings = ['utf-8', 'latin-1', 'cp1252']
            loaded = False
            for encoding in encodings:
                try:
                    loader = TextLoader(file_path, encoding=encoding)
                    docs = loader.load()
                    # Add file path to metadata if TextLoader doesn't do it consistently
                    for doc in docs:
                        if 'source' not in doc.metadata:
                            doc.metadata['source'] = file_path
                    loaded_docs.extend(docs)
                    logger.info(f"成功加载文件: {file_path} (编码: {encoding})")
                    loaded = True
                    break
                except UnicodeDecodeError:
                    continue
                except Exception as e:
                    logger.error(f"使用{encoding}编码加载文件{file_path}失败: {e}")
                    break # Stop trying encodings for this file

            if not loaded:
                logger.error(f"无法使用任何已知编码加载文件: {file_path}")
        except Exception as e:
            logger.error(f"加载文件 {file_path} 时发生意外错误: {e}", exc_info=True)

    logger.info(f"共加载了{len(loaded_docs)}个文档")
    return loaded_docs 