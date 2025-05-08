"""
启动 FastMCP SSE 服务器
"""
import os
import sys
import logging
import subprocess
import time
import signal
import atexit

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - [SSE-LAUNCHER] - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# 服务器配置
SERVER_SCRIPT = "../mcp_server_sse.py"
SERVER_URL = "http://127.0.0.1:8000"

def start_server(server_script=SERVER_SCRIPT):
    """启动 SSE 服务器

    Args:
        server_script: 服务器脚本路径，默认为 SERVER_SCRIPT
    """
    logger.info(f"启动 SSE 服务器: {server_script}")

    # 设置环境变量以使用 SSE 传输协议
    env = os.environ.copy()
    env["FASTMCP_TRANSPORT"] = "sse"

    # 启动服务器进程
    server_process = subprocess.Popen(
        ["python", server_script],
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )

    # 注册退出处理函数
    def cleanup():
        logger.info("清理资源...")
        if server_process.poll() is None:
            logger.info("终止服务器进程...")
            server_process.terminate()
            try:
                server_process.wait(timeout=5)
                logger.info("服务器进程已终止")
            except subprocess.TimeoutExpired:
                logger.warning("服务器进程未能在超时时间内终止，强制终止")
                server_process.kill()
                server_process.wait()
                logger.info("服务器进程已强制终止")

    atexit.register(cleanup)

    # 处理信号
    def signal_handler(sig, frame):
        logger.info(f"接收到信号: {sig}")
        cleanup()
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    # 等待服务器初始化
    logger.info("等待服务器初始化...")
    time.sleep(3)

    # 输出服务器信息
    logger.info(f"服务器 URL: {SERVER_URL}")
    logger.info("服务器已启动，按 Ctrl+C 终止")

    # 持续输出服务器日志
    while True:
        output = server_process.stdout.readline()
        if output:
            print(output.strip())

        error = server_process.stderr.readline()
        if error:
            print(f"ERROR: {error.strip()}", file=sys.stderr)

        # 检查服务器进程是否仍在运行
        if server_process.poll() is not None:
            logger.error(f"服务器进程已终止，退出码: {server_process.returncode}")
            break

        # 避免 CPU 占用过高
        time.sleep(0.1)

if __name__ == "__main__":
    # 获取脚本目录和项目根目录
    script_dir = os.path.dirname(os.path.abspath(__file__))
    root_dir = os.path.dirname(script_dir)  # 项目根目录

    # 修改服务器脚本路径为绝对路径
    server_script = os.path.join(root_dir, "mcp_server_sse.py")

    # 切换到脚本所在目录
    os.chdir(script_dir)

    start_server(server_script)
