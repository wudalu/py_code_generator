"""
MCP 客户端使用示例

本示例展示如何在实际项目中使用 MCP 客户端进行代码嵌入和搜索。
"""
import os
import asyncio
import logging
from typing import List, Dict, Any

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - [EXAMPLE] - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# 导入 MCP 客户端
from mcp_client_stdio_new import CodeEmbeddingClient

async def process_project_files(client: CodeEmbeddingClient, project_dir: str, collection_name: str) -> Dict[str, Any]:
    """处理项目文件
    
    Args:
        client: MCP 客户端
        project_dir: 项目目录
        collection_name: 集合名称
        
    Returns:
        处理结果
    """
    logger.info(f"处理项目: {project_dir}")
    logger.info(f"集合名称: {collection_name}")
    
    # 收集 Python 文件
    python_files = []
    for root, _, files in os.walk(project_dir):
        for file in files:
            if file.endswith(".py"):
                file_path = os.path.join(root, file)
                python_files.append(file_path)
    
    logger.info(f"找到 {len(python_files)} 个 Python 文件")
    
    # 处理文件
    if python_files:
        result = await client.process_files(python_files, collection_name)
        logger.info(f"处理结果: {result}")
        return result
    else:
        logger.warning("未找到 Python 文件")
        return {"status": "warning", "message": "未找到 Python 文件"}

async def search_code_examples(client: CodeEmbeddingClient, queries: List[str], collection_name: str) -> Dict[str, List[Dict[str, Any]]]:
    """搜索代码示例
    
    Args:
        client: MCP 客户端
        queries: 搜索查询列表
        collection_name: 集合名称
        
    Returns:
        搜索结果字典，键为查询，值为结果列表
    """
    logger.info(f"搜索代码示例，集合: {collection_name}")
    
    results = {}
    for query in queries:
        logger.info(f"搜索查询: {query}")
        search_results = await client.search_code(query, k=3, collection_name=collection_name)
        results[query] = search_results
        logger.info(f"找到 {len(search_results) if isinstance(search_results, list) else 0} 个结果")
    
    return results

async def main():
    """主函数"""
    logger.info("启动 MCP 客户端示例")
    
    # 创建客户端
    client = CodeEmbeddingClient()
    
    try:
        # 连接到服务器
        logger.info("连接到 MCP 服务器...")
        server_info = await client.connect_to_server("mcp_server_stdio_new.py")
        logger.info(f"连接成功！服务器信息: {server_info}")
        
        # 列出所有集合
        logger.info("列出所有集合...")
        collections = await client.list_collections()
        logger.info(f"集合列表: {collections}")
        
        # 处理当前目录
        current_dir = os.path.dirname(os.path.abspath(__file__))
        collection_name = "example_collection"
        
        # 处理项目文件
        await process_project_files(client, current_dir, collection_name)
        
        # 获取集合信息
        logger.info(f"获取集合信息: {collection_name}")
        info = await client.get_collection_info(collection_name)
        logger.info(f"集合信息: {info}")
        
        # 搜索代码示例
        queries = [
            "code embedding",
            "process files",
            "search code",
            "vector storage"
        ]
        search_results = await search_code_examples(client, queries, collection_name)
        
        # 显示搜索结果
        logger.info("搜索结果摘要:")
        for query, results in search_results.items():
            logger.info(f"查询: {query}")
            if isinstance(results, list) and results:
                for i, result in enumerate(results):
                    if isinstance(result, dict) and 'content' in result:
                        content = result['content']
                        # 只显示前 100 个字符
                        logger.info(f"  结果 {i+1}: {content[:100]}...")
                    else:
                        logger.info(f"  结果 {i+1}: {result}")
            else:
                logger.info("  未找到结果")
        
    except Exception as e:
        logger.error(f"发生错误: {e}")
        import traceback
        logger.error(traceback.format_exc())
    finally:
        # 断开连接
        logger.info("断开连接...")
        await client.disconnect()
        logger.info("示例完成")

if __name__ == "__main__":
    asyncio.run(main())
