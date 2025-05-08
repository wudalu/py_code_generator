"""
检查 MCP 版本

此脚本用于检查当前安装的 MCP 版本是否与要求的版本匹配。
"""
import sys
import logging
from importlib.metadata import version, PackageNotFoundError

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - [VERSION-CHECK] - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# 要求的 MCP 版本
REQUIRED_MCP_VERSION = "1.6.0"

def check_mcp_version():
    """检查 MCP 版本"""
    logger.info(f"检查 MCP 版本，要求版本: {REQUIRED_MCP_VERSION}")

    try:
        # 获取当前安装的 MCP 版本
        mcp_version = version("mcp")
        logger.info(f"当前安装的 MCP 版本: {mcp_version}")

        # 检查版本是否匹配
        if mcp_version != REQUIRED_MCP_VERSION:
            logger.warning(f"警告: MCP 版本不匹配! 当前版本: {mcp_version}, 要求版本: {REQUIRED_MCP_VERSION}")
            logger.warning("版本不匹配可能导致客户端和服务端通信问题")
            return False
        else:
            logger.info(f"MCP 版本检查通过: {mcp_version}")
            return True
    except PackageNotFoundError:
        logger.error("MCP 包未安装")
        return False
    except Exception as e:
        logger.error(f"无法检查 MCP 版本: {e}")
        return False

def check_dependencies():
    """检查依赖项"""
    logger.info("检查依赖项...")

    dependencies = [
        ("mcp", REQUIRED_MCP_VERSION),
        ("httpx", None),
        ("langchain", None),
        ("langchain_community", None),
        ("chromadb", None),
        ("sentence-transformers", None)
    ]

    all_ok = True
    for package, required_version in dependencies:
        try:
            installed_version = version(package)
            if required_version and installed_version != required_version:
                logger.warning(f"{package}: 版本不匹配! 当前版本: {installed_version}, 要求版本: {required_version}")
                all_ok = False
            else:
                logger.info(f"{package}: 已安装 (版本: {installed_version})")
        except PackageNotFoundError:
            logger.error(f"{package}: 未安装")
            all_ok = False
        except Exception as e:
            logger.error(f"{package}: 无法检查版本 - {e}")
            all_ok = False

    return all_ok

if __name__ == "__main__":
    logger.info("开始检查 MCP 版本和依赖项...")

    # 检查 MCP 版本
    mcp_ok = check_mcp_version()

    # 检查依赖项
    deps_ok = check_dependencies()

    # 输出结果
    if mcp_ok and deps_ok:
        logger.info("所有检查通过！")
        sys.exit(0)
    else:
        if not mcp_ok:
            logger.error("MCP 版本检查失败！")
        if not deps_ok:
            logger.error("依赖项检查失败！")
        logger.error("请安装正确版本的依赖项，然后重试。")
        logger.error(f"可以使用以下命令安装依赖项: pip install -r requirements.txt")
        sys.exit(1)
