"""
MCP (Model Context Protocol) Client 实现 - FastMCP v2 版本
连接到 CodeEmbedder MCP Server，使用代码向量化和搜索功能
使用 FastMCP v2 库，遵循 MCP 最佳实践

参考: https://gofastmcp.com
"""
import logging
import asyncio
from typing import Optional, List, Dict, Any

from fastmcp import Client
from fastmcp.client.transports import SSETransport, PythonStdioTransport

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - [FASTMCP-CLIENT] - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class CodeEmbeddingClient:
    """代码嵌入 FastMCP 客户端"""

    def __init__(self, server_url_or_path: str, transport_type: str = "stdio"):
        """初始化客户端

        Args:
            server_url_or_path: MCP 服务器 URL 或脚本路径
            transport_type: 传输类型，可选 'stdio' 或 'sse'，默认为 'stdio'
        """
        self.server_url_or_path = server_url_or_path
        self.transport_type = transport_type
        self.client = None

    async def __aenter__(self):
        """异步上下文管理器入口"""
        await self.connect()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """异步上下文管理器出口"""
        await self.disconnect()

    async def connect(self):
        """连接到 MCP 服务器"""
        logger.info(f"连接到 MCP 服务器: {self.server_url_or_path}")

        # 根据传输类型创建适当的传输对象
        if self.transport_type == "sse":
            transport = SSETransport(self.server_url_or_path)
        else:  # 默认使用 stdio
            transport = PythonStdioTransport(self.server_url_or_path)

        # 创建客户端
        self.client = Client(transport)

        # 连接到服务器
        await self.client.__aenter__()
        logger.info("成功连接到 MCP 服务器")

        return self

    async def disconnect(self):
        """断开与服务器的连接"""
        if self.client:
            logger.info("断开与服务器的连接")
            await self.client.__aexit__(None, None, None)
            self.client = None
            logger.info("已断开连接")

    async def process_files(self, file_paths: List[str], collection_name: str = "code_collection") -> Dict[str, Any]:
        """处理代码文件：加载、分割、嵌入、存储

        Args:
            file_paths: 代码文件路径列表
            collection_name: 集合名称，默认为code_collection

        Returns:
            处理结果
        """
        if not self.client:
            raise RuntimeError("未连接到服务器，请先调用 connect()")

        logger.info(f"调用工具: process_files")
        logger.info(f"file_paths: {file_paths}")
        logger.info(f"collection_name: {collection_name}")

        try:
            result = await self.client.call_tool(
                "process_files",
                {
                    "file_paths": file_paths,
                    "collection_name": collection_name
                }
            )
            logger.info(f"工具调用结果: {result}")

            # 解析结果
            if result and len(result) > 0:
                # 将字符串转换为字典
                import json
                return json.loads(result[0].text)
            else:
                return {"status": "error", "message": "工具调用返回空结果"}
        except Exception as e:
            logger.error(f"调用 process_files 工具时出错: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return {"status": "error", "message": f"调用工具时出错: {str(e)}"}

    async def search_code(self, query: str, k: int = 5, collection_name: Optional[str] = None) -> List[Dict[str, Any]]:
        """搜索相关代码

        Args:
            query: 查询字符串
            k: 返回结果数量，默认为5
            collection_name: 集合名称，可选

        Returns:
            搜索结果列表
        """
        if not self.client:
            raise RuntimeError("未连接到服务器，请先调用 connect()")

        logger.info(f"调用工具: search_code")
        logger.info(f"query: {query}")
        logger.info(f"k: {k}")
        logger.info(f"collection_name: {collection_name}")

        arguments = {
            "query": query,
            "k": k
        }

        if collection_name:
            arguments["collection_name"] = collection_name

        try:
            result = await self.client.call_tool("search_code", arguments)
            logger.info(f"工具调用结果: {result}")

            # 解析结果
            if result and len(result) > 0:
                # 将字符串转换为列表
                import json
                return json.loads(result[0].text)
            else:
                return []
        except Exception as e:
            logger.error(f"调用 search_code 工具时出错: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return []

    async def load_collection(self, collection_name: Optional[str] = None) -> Dict[str, Any]:
        """加载已存在的向量存储集合

        Args:
            collection_name: 集合名称，可选

        Returns:
            加载结果
        """
        if not self.client:
            raise RuntimeError("未连接到服务器，请先调用 connect()")

        logger.info(f"调用工具: load_collection")
        logger.info(f"collection_name: {collection_name}")

        arguments = {}
        if collection_name:
            arguments["collection_name"] = collection_name

        try:
            result = await self.client.call_tool("load_collection", arguments)
            logger.info(f"工具调用结果: {result}")

            # 解析结果
            if result and len(result) > 0:
                # 将字符串转换为字典
                import json
                return json.loads(result[0].text)
            else:
                return {"status": "error", "message": "工具调用返回空结果"}
        except Exception as e:
            logger.error(f"调用 load_collection 工具时出错: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return {"status": "error", "message": f"调用工具时出错: {str(e)}"}

    async def list_collections(self) -> Dict[str, List[str]]:
        """列出所有可用的集合

        Returns:
            集合列表
        """
        if not self.client:
            raise RuntimeError("未连接到服务器，请先调用 connect()")

        logger.info("读取资源: code://collections")

        try:
            result = await self.client.read_resource("code://collections")
            logger.info(f"资源读取结果: {result}")

            # 解析结果
            if result and len(result) > 0:
                # 将字符串转换为字典
                import json
                return json.loads(result[0].text)
            else:
                return {"collections": []}
        except Exception as e:
            logger.error(f"读取 collections 资源时出错: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return {"collections": [], "error": str(e)}

    async def get_collection_info(self, collection_name: str) -> Dict[str, Any]:
        """获取特定集合的信息

        Args:
            collection_name: 集合名称

        Returns:
            集合信息
        """
        if not self.client:
            raise RuntimeError("未连接到服务器，请先调用 connect()")

        logger.info(f"读取资源: code://collection/{collection_name}/info")

        try:
            result = await self.client.read_resource(f"code://collection/{collection_name}/info")
            logger.info(f"资源读取结果: {result}")

            # 解析结果
            if result and len(result) > 0:
                # 将字符串转换为字典
                import json
                return json.loads(result[0].text)
            else:
                return {"name": collection_name, "error": "资源读取返回空结果"}
        except Exception as e:
            logger.error(f"读取集合信息资源时出错: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return {"name": collection_name, "error": str(e)}

    async def get_code_search_prompt(self, query: str) -> str:
        """获取代码搜索提示

        Args:
            query: 搜索查询

        Returns:
            搜索提示
        """
        if not self.client:
            raise RuntimeError("未连接到服务器，请先调用 connect()")

        logger.info(f"获取提示: code_search")
        logger.info(f"query: {query}")

        try:
            result = await self.client.get_prompt("code_search", {"query": query})
            logger.info(f"提示获取结果: {result}")

            # 解析结果
            if result and len(result) > 0:
                return result[0].content.text
            else:
                return f"获取提示失败: 返回空结果"
        except Exception as e:
            logger.error(f"获取 code_search 提示时出错: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return f"获取提示时出错: {str(e)}"

async def interactive_client(server_path: str, transport_type: str = "stdio"):
    """交互式客户端

    Args:
        server_path: 服务器脚本路径
        transport_type: 传输类型，可选 'stdio' 或 'sse'，默认为 'stdio'
    """
    async with CodeEmbeddingClient(server_path, transport_type) as client:
        print("客户端初始化完成！")

        # 列出所有集合
        print("获取可用集合...")
        collections = await client.list_collections()

        # 显示初始化结果
        if isinstance(collections, dict) and collections.get("collections"):
            print(f"\n可用集合:")
            for collection in collections["collections"]:
                print(f"  - {collection}")
        else:
            print("未找到任何集合")

        # 交互式循环
        running = True
        while running:
            print("\n可用命令:")
            print("1. 处理代码文件")
            print("2. 搜索代码")
            print("3. 加载集合")
            print("4. 列出所有集合")
            print("5. 获取集合信息")
            print("6. 获取代码搜索提示")
            print("7. 退出")

            choice = input("\n请选择命令 (1-7): ")

            if choice == "1":
                # 处理代码文件
                file_paths_input = input("请输入代码文件路径 (多个路径用逗号分隔): ")
                file_paths = [path.strip() for path in file_paths_input.split(",")]
                collection_name = input("请输入集合名称 (默认: code_collection): ") or "code_collection"

                result = await client.process_files(file_paths, collection_name)
                print(f"处理结果: {result}")

            elif choice == "2":
                # 搜索代码
                query = input("请输入搜索查询: ")
                k_input = input("请输入返回结果数量 (默认: 5): ")
                k = int(k_input) if k_input.isdigit() else 5
                collection_name = input("请输入集合名称 (可选): ")

                if not collection_name:
                    collection_name = None

                results = await client.search_code(query, k, collection_name)

                if results:
                    print(f"\n找到 {len(results)} 个结果:")
                    for i, result in enumerate(results):
                        print(f"\n结果 {i+1}:")
                        print(f"内容: {result['content']}")
                        if result.get('metadata'):
                            print(f"元数据: {result['metadata']}")
                else:
                    print("未找到任何结果")

            elif choice == "3":
                # 加载集合
                collection_name = input("请输入集合名称 (可选): ")

                if not collection_name:
                    collection_name = None

                result = await client.load_collection(collection_name)
                print(f"加载结果: {result}")

            elif choice == "4":
                # 列出所有集合
                collections = await client.list_collections()

                if isinstance(collections, dict) and collections.get("collections"):
                    print(f"\n可用集合:")
                    for collection in collections["collections"]:
                        print(f"  - {collection}")
                else:
                    print("未找到任何集合")

            elif choice == "5":
                # 获取集合信息
                collection_name = input("请输入集合名称: ")

                if collection_name:
                    info = await client.get_collection_info(collection_name)
                    print(f"集合信息: {info}")
                else:
                    print("集合名称不能为空")

            elif choice == "6":
                # 获取代码搜索提示
                query = input("请输入搜索查询: ")

                if query:
                    prompt = await client.get_code_search_prompt(query)
                    print(f"\n代码搜索提示:\n{prompt}")
                else:
                    print("搜索查询不能为空")

            elif choice == "7":
                # 退出
                running = False
                print("已退出")

            else:
                print("无效的选择，请重试")

if __name__ == "__main__":
    import sys
    import argparse

    # 创建命令行参数解析器
    parser = argparse.ArgumentParser(description="FastMCP 客户端")
    parser.add_argument("server_path", nargs="?", default="mcp_server_stdio.py",
                        help="服务器脚本路径或 URL")
    parser.add_argument("--sse", action="store_true", help="使用 SSE 传输协议")

    # 解析命令行参数
    args = parser.parse_args()

    # 设置传输类型和服务器路径
    transport_type = "sse" if args.sse else "stdio"
    server_path = args.server_path

    # 如果使用 SSE 传输协议但没有指定 URL，则使用默认 URL
    if transport_type == "sse" and not server_path.startswith("http"):
        server_path = "http://localhost:8000"

    try:
        asyncio.run(interactive_client(server_path, transport_type))
    except KeyboardInterrupt:
        print("\n接收到中断信号，正在退出...")
    except Exception as e:
        print(f"发生错误: {e}")
        import traceback
        print(traceback.format_exc())
