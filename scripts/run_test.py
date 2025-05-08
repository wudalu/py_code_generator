"""
测试 FastMCP v2 实现的服务器和客户端
"""
import os
import sys
import logging
import asyncio
import subprocess
import time
from pathlib import Path

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - [FASTMCP-TEST] - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# 测试文件目录
TEST_DIR = "../test_code_files"

def create_test_files():
    """创建测试文件"""
    logger.info("创建测试文件...")

    # 创建测试目录
    os.makedirs(TEST_DIR, exist_ok=True)

    # 创建测试文件
    test_files = [
        {
            "name": "example1.py",
            "content": """
def fibonacci(n):
    \"\"\"计算斐波那契数列的第n个数

    Args:
        n: 位置索引，从0开始

    Returns:
        斐波那契数
    \"\"\"
    if n <= 0:
        return 0
    elif n == 1:
        return 1
    else:
        return fibonacci(n-1) + fibonacci(n-2)

def factorial(n):
    \"\"\"计算阶乘

    Args:
        n: 正整数

    Returns:
        n的阶乘
    \"\"\"
    if n <= 1:
        return 1
    else:
        return n * factorial(n-1)
"""
        },
        {
            "name": "example2.py",
            "content": """
def is_prime(n):
    \"\"\"判断一个数是否为素数

    Args:
        n: 整数

    Returns:
        是否为素数
    \"\"\"
    if n <= 1:
        return False
    if n <= 3:
        return True
    if n % 2 == 0 or n % 3 == 0:
        return False
    i = 5
    while i * i <= n:
        if n % i == 0 or n % (i + 2) == 0:
            return False
        i += 6
    return True

def gcd(a, b):
    \"\"\"计算最大公约数

    Args:
        a: 整数
        b: 整数

    Returns:
        a和b的最大公约数
    \"\"\"
    while b:
        a, b = b, a % b
    return a
"""
        }
    ]

    # 写入测试文件
    file_paths = []
    for file_info in test_files:
        file_path = os.path.join(TEST_DIR, file_info["name"])
        with open(file_path, "w") as f:
            f.write(file_info["content"])
        file_paths.append(file_path)
        logger.info(f"创建测试文件: {file_path}")

    return file_paths

async def test_stdio_mode():
    """测试 stdio 模式"""
    logger.info("测试 stdio 模式...")

    # 创建测试文件
    test_files = create_test_files()

    # 启动服务器进程
    server_process = subprocess.Popen(
        ["python", "../mcp_server_stdio.py"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )

    # 等待服务器初始化
    logger.info("等待服务器初始化...")
    time.sleep(3)

    try:
        # 运行客户端
        client_process = subprocess.Popen(
            ["python", "../mcp_client_stdio.py", "../mcp_server_stdio.py"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )

        # 发送命令
        logger.info("发送处理文件命令...")
        client_process.stdin.write("1\n")
        client_process.stdin.write(f"{','.join(test_files)}\n")
        client_process.stdin.write("test_collection\n")
        client_process.stdin.flush()

        # 等待处理完成
        logger.info("等待处理完成...")
        time.sleep(5)

        # 发送搜索命令
        logger.info("发送搜索命令...")
        client_process.stdin.write("2\n")
        client_process.stdin.write("fibonacci\n")
        client_process.stdin.write("2\n")
        client_process.stdin.write("test_collection\n")
        client_process.stdin.flush()

        # 等待搜索完成
        logger.info("等待搜索完成...")
        time.sleep(5)

        # 退出客户端
        logger.info("退出客户端...")
        client_process.stdin.write("7\n")
        client_process.stdin.flush()

        # 等待客户端退出
        client_process.wait(timeout=5)

        # 检查客户端输出
        client_output = client_process.stdout.read()
        logger.info(f"客户端输出: {client_output}")

        # 检查是否成功
        if "找到" in client_output and "fibonacci" in client_output:
            logger.info("stdio 模式测试成功")
            return True
        else:
            logger.error("stdio 模式测试失败")
            return False
    except Exception as e:
        logger.error(f"stdio 模式测试出错: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False
    finally:
        # 停止服务器
        logger.info("停止服务器...")
        server_process.terminate()
        try:
            server_process.wait(timeout=5)
            logger.info("服务器已停止")
        except subprocess.TimeoutExpired:
            logger.warning("服务器未能在超时时间内停止，强制终止")
            server_process.kill()
            server_process.wait()
            logger.info("服务器已强制终止")

async def test_sse_mode():
    """测试 SSE 模式"""
    logger.info("测试 SSE 模式...")

    # 创建测试文件
    test_files = create_test_files()

    # 启动服务器进程
    server_process = subprocess.Popen(
        ["python", "../mcp_server_sse.py"],
        env={**os.environ, "FASTMCP_TRANSPORT": "sse"},
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )

    # 等待服务器初始化
    logger.info("等待服务器初始化...")
    time.sleep(5)

    try:
        # 运行客户端
        client_process = subprocess.Popen(
            ["python", "../mcp_client_sse.py", "--sse", "http://localhost:8000"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )

        # 发送命令
        logger.info("发送处理文件命令...")
        client_process.stdin.write("1\n")
        client_process.stdin.write(f"{','.join(test_files)}\n")
        client_process.stdin.write("test_collection_sse\n")
        client_process.stdin.flush()

        # 等待处理完成
        logger.info("等待处理完成...")
        time.sleep(5)

        # 发送搜索命令
        logger.info("发送搜索命令...")
        client_process.stdin.write("2\n")
        client_process.stdin.write("fibonacci\n")
        client_process.stdin.write("2\n")
        client_process.stdin.write("test_collection_sse\n")
        client_process.stdin.flush()

        # 等待搜索完成
        logger.info("等待搜索完成...")
        time.sleep(5)

        # 退出客户端
        logger.info("退出客户端...")
        client_process.stdin.write("7\n")
        client_process.stdin.flush()

        # 等待客户端退出
        client_process.wait(timeout=5)

        # 检查客户端输出
        client_output = client_process.stdout.read()
        logger.info(f"客户端输出: {client_output}")

        # 检查是否成功
        if "找到" in client_output and "fibonacci" in client_output:
            logger.info("SSE 模式测试成功")
            return True
        else:
            logger.error("SSE 模式测试失败")
            return False
    except Exception as e:
        logger.error(f"SSE 模式测试出错: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False
    finally:
        # 停止服务器
        logger.info("停止服务器...")
        server_process.terminate()
        try:
            server_process.wait(timeout=5)
            logger.info("服务器已停止")
        except subprocess.TimeoutExpired:
            logger.warning("服务器未能在超时时间内停止，强制终止")
            server_process.kill()
            server_process.wait()
            logger.info("服务器已强制终止")

async def main():
    """主函数"""
    logger.info("开始测试 FastMCP v2 实现的服务器和客户端...")

    # 测试 stdio 模式
    stdio_success = await test_stdio_mode()

    # 测试 SSE 模式
    sse_success = await test_sse_mode()

    # 输出测试结果
    logger.info("测试结果:")
    logger.info(f"  stdio 模式: {'成功' if stdio_success else '失败'}")
    logger.info(f"  SSE 模式: {'成功' if sse_success else '失败'}")

    if stdio_success and sse_success:
        logger.info("所有测试都成功通过！")
    else:
        logger.error("部分测试失败！")

if __name__ == "__main__":
    # 切换到脚本所在目录
    os.chdir(os.path.dirname(os.path.abspath(__file__)))

    # 运行测试
    asyncio.run(main())
