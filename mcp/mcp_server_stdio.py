"""
MCP (Model Context Protocol) Server 实现 - FastMCP v2 版本
提供代码向量化和搜索功能
使用 FastMCP v2 库，遵循 MCP 最佳实践

参考: https://gofastmcp.com
"""
import os
import logging
from typing import List, Dict, Any, Optional

from fastmcp import FastMCP
from embedding import CodeEmbedder

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - [FASTMCP-SERVER] - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# 创建 MCP 服务器
mcp = FastMCP(
    "CodeEmbedder",
    dependencies=["langchain", "langchain_community", "chromadb", "sentence-transformers"]
)

# 初始化代码嵌入器
logger.info("初始化代码嵌入器...")
code_embedder = CodeEmbedder()
logger.info("代码嵌入器初始化完成")

# 工具: 处理代码文件
@mcp.tool()
async def process_files(file_paths: List[str], collection_name: str = "code_collection") -> Dict[str, Any]:
    """
    处理代码文件：加载、分割、嵌入、存储

    Args:
        file_paths: 代码文件路径列表
        collection_name: 集合名称，默认为code_collection

    Returns:
        处理结果
    """
    logger.info(f"处理代码文件: {file_paths}")
    logger.info(f"集合名称: {collection_name}")

    try:
        # 处理文件
        result = code_embedder.process_code_files(file_paths, collection_name)
        logger.info(f"处理结果: {result}")
        return result
    except Exception as e:
        logger.error(f"处理文件时出错: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return {"status": "error", "message": str(e)}

# 工具: 搜索代码
@mcp.tool()
async def search_code(query: str, k: int = 5, collection_name: Optional[str] = None) -> List[Dict[str, Any]]:
    """
    搜索相关代码

    Args:
        query: 查询字符串
        k: 返回结果数量，默认为5
        collection_name: 集合名称，可选

    Returns:
        搜索结果列表
    """
    logger.info(f"搜索代码: {query}")
    logger.info(f"k: {k}")
    logger.info(f"集合名称: {collection_name}")

    try:
        # 搜索代码
        results = code_embedder.search(query, k, collection_name)
        logger.info(f"找到 {len(results)} 个结果")
        return results
    except Exception as e:
        logger.error(f"搜索代码时出错: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return []

# 工具: 加载集合
@mcp.tool()
async def load_collection(collection_name: Optional[str] = None) -> Dict[str, Any]:
    """
    加载已存在的向量存储集合

    Args:
        collection_name: 集合名称，可选

    Returns:
        加载结果
    """
    logger.info(f"加载集合: {collection_name}")

    try:
        # 加载集合
        result = code_embedder.load_collection(collection_name)
        logger.info(f"加载结果: {result}")
        return result
    except Exception as e:
        logger.error(f"加载集合时出错: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return {"status": "error", "message": str(e)}

# 资源: 集合列表
@mcp.resource("code://collections")
async def get_collections() -> Dict[str, List[str]]:
    """
    获取所有可用的集合

    Returns:
        集合列表
    """
    logger.info("获取集合列表")

    try:
        # 获取集合列表
        collections = code_embedder.get_collections()
        logger.info(f"找到 {len(collections)} 个集合")
        return {"collections": collections}
    except Exception as e:
        logger.error(f"获取集合列表时出错: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return {"collections": []}

# 资源: 集合信息
@mcp.resource("code://collection/{collection_name}/info")
async def get_collection_info(collection_name: str) -> Dict[str, Any]:
    """
    获取特定集合的信息

    Args:
        collection_name: 集合名称

    Returns:
        集合信息
    """
    logger.info(f"获取集合信息: {collection_name}")

    try:
        # 获取集合信息
        info = code_embedder.get_collection_info(collection_name)
        logger.info(f"集合信息: {info}")
        return info
    except Exception as e:
        logger.error(f"获取集合信息时出错: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return {"name": collection_name, "error": str(e)}

# 提示: 代码搜索
@mcp.prompt()
async def code_search(query: str) -> str:
    """
    生成代码搜索提示

    Args:
        query: 搜索查询

    Returns:
        搜索提示
    """
    logger.info(f"生成代码搜索提示: {query}")

    prompt = f"""
我正在搜索与以下查询相关的代码:

{query}

请使用search_code工具执行搜索，然后解释找到的代码。
"""
    return prompt

if __name__ == "__main__":
    # 运行MCP Server
    logger.info("启动 MCP Server...")
    try:
        # 检查环境变量以决定传输协议
        transport = os.environ.get("FASTMCP_TRANSPORT", "stdio")
        logger.info(f"使用 {transport} 传输协议")

        # 启动服务器
        if transport == "sse":
            # 对于 SSE 传输协议，指定端口和主机
            logger.info("使用 SSE 传输协议，端口: 8000")
            mcp.run(transport=transport, port=8000, host="0.0.0.0")
        else:
            # 对于其他传输协议，使用默认设置
            mcp.run(transport=transport)
    except KeyboardInterrupt:
        logger.info("接收到中断信号，服务器即将关闭")
    except Exception as e:
        logger.error(f"服务器运行时出错: {e}")
        import traceback
        logger.error(traceback.format_exc())
    finally:
        logger.info("服务器已关闭")
