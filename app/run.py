"""LogWise 一键启动器
======================================================

作用：不用敲任何命令行，直接启动 LogWise 服务并自动打开浏览器。

启动方式（任选一种）：
  1. 【推荐】双击本文件
  2. 在 PyCharm 里右键本文件 → 运行 'run'
  3. 命令行： .venv\\Scripts\\python.exe app\\run.py

停止服务：把弹出的黑色窗口关掉，或按 Ctrl+C。

可选参数：
  python app\\run.py --reload      改代码后自动重启（开发时用）
  python app\\run.py --port 9000   手动指定端口
======================================================
"""
from __future__ import annotations

import argparse
import os
import socket
import sys
import threading
import time
import webbrowser
from pathlib import Path

# 注意：本文件位于 app/ 子目录内，所以项目根目录要往上退一层。
# （原来的 run.py 在根目录时这里是 .parent，移进来后必须改成 .parent.parent）
ROOT = Path(__file__).resolve().parent.parent

# 脚本被放在子目录里跑时，Python 只会把「脚本所在目录」加进 sys.path，
# 于是 `import app.main` 会失败。这里手动把项目根目录补进去。
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

HOST = "127.0.0.1"
DEFAULT_PORT = 8000
MIN_PYTHON = (3, 10)
LINE = "=" * 58
ENTRY = "app/run.py"


def _setup_console() -> None:
    """让中文在 Windows 终端里正常显示（实现统一在 app/launcher.py）。"""
    from app.launcher import setup_console

    setup_console()


def _title(text: str) -> None:
    print(LINE)
    print(f"  {text}")
    print(LINE)


def _fatal(msg: str, hint: str = "") -> None:
    """打印友好的错误并退出。"""
    print()
    print(LINE)
    print("  [启动失败]")
    print(LINE)
    print(f"  {msg}")
    if hint:
        print()
        for line in hint.strip().splitlines():
            print(f"  {line}")
    print(LINE)
    print()
    try:
        input("  按回车键关闭窗口...")
    except (EOFError, KeyboardInterrupt):
        pass
    sys.exit(1)


def check_python() -> None:
    if sys.version_info < MIN_PYTHON:
        _fatal(
            f"当前 Python 版本是 {sys.version.split()[0]}，太旧了。",
            f"""
项目需要 Python {MIN_PYTHON[0]}.{MIN_PYTHON[1]} 或更高版本。

你现在用的解释器是：
  {sys.executable}

解决办法：确认项目根目录下的 .venv 是用新版 Python 建的。
（正常情况下直接用项目里的 .venv 运行就没有这个问题）
""",
        )


def check_env_file() -> str | None:
    """检查 .env 与 API Key，返回 Key（无效时返回 None）。"""
    env_path = ROOT / ".env"
    if not env_path.exists():
        print("  [提醒] 没有找到 .env 文件，AI 诊断功能将不可用。")
        print("        修复：把 .env.example 复制一份改名为 .env，填入你的 Key。")
        print()
        return None

    key = ""
    base_url = ""
    for raw in env_path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        name, _, value = line.partition("=")
        name, value = name.strip(), value.strip()
        if name == "DEEPSEEK_API_KEY":
            key = value
        elif name == "DEEPSEEK_BASE_URL":
            base_url = value

    # 占位符判定统一走 app.apikey —— 这里原先自己匹配 "sk-在这里"，
    # 且额外用 len(key) < 20 猜长度，会误伤合法的短 Key。
    from app.apikey import is_placeholder, mask

    if is_placeholder(key):
        print("  [提醒] .env 里的 DEEPSEEK_API_KEY 还没填，AI 诊断功能将不可用。")
        print(f"        文件位置：{env_path}")
        print("        （不过「解析日志」功能不需要 Key，仍然可以正常演示）")
        print()
        return None

    print(f"  [OK] 已读到 API Key（{mask(key)}）")
    print(f"  [OK] 接口地址：{base_url or 'https://api.deepseek.com'}")
    if len(key) < 20:
        # 只提醒、不当成「没填」—— 长度不是可靠信号，误判成未配置更糟
        print(f"  [注意] 这个 Key 只有 {len(key)} 个字符，比常见长度短。")
        print("         如果调用时报 401，请核对是否复制完整。")
    return key


def check_dependencies() -> None:
    missing = []
    for mod, pkg in (("fastapi", "fastapi"), ("uvicorn", "uvicorn"), ("openai", "openai"), ("dotenv", "python-dotenv")):
        try:
            __import__(mod)
        except ImportError:
            missing.append(pkg)

    if missing:
        _fatal(
            "缺少这些依赖库：" + "、".join(missing),
            f"""
说明：虚拟环境里还没装好依赖，或者装得不完整。

解决办法：在项目根目录下依次执行这两条命令
  py -3.12 -m venv .venv
  .venv\\Scripts\\python.exe -m pip install -i https://pypi.tuna.tsinghua.edu.cn/simple -r requirements.txt

或者把这句话发给 AI 助手，让它帮你装好。

项目根目录：{ROOT}
""",
        )


def is_port_busy(port: int) -> bool:
    """检测端口上是否已经有服务在监听。"""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(0.4)
        return sock.connect_ex((HOST, port)) == 0


def pick_port(start: int) -> int:
    """从 start 开始找一个没被占用的端口。"""
    for port in range(start, start + 20):
        if not is_port_busy(port):
            return port
    _fatal(
        f"端口 {start} ~ {start + 19} 全部被占用了。",
        "请关掉一些正在运行的程序，或手动指定端口：\n"
        f"  .venv\\Scripts\\python.exe {ENTRY} --port 9000",
    )
    raise SystemExit(1)


def open_browser_when_ready(url: str) -> None:
    """后台等待服务起来，然后自动打开浏览器。"""
    for _ in range(120):  # 最多等 60 秒
        if is_port_busy(int(url.rsplit(":", 1)[1])):
            time.sleep(0.3)  # 再给一点点时间让首页加载完
            try:
                webbrowser.open(url)
            except Exception:
                pass
            return
        time.sleep(0.5)


def _reenter_venv() -> None:
    """确保用项目自带的 .venv 解释器运行。

    为什么需要这个：Windows 上直接运行 .py 文件时，可能用的是系统全局 Python，
    而依赖（fastapi / uvicorn）只装在项目的 .venv 里。这里自动切过去，
    这样「直接运行」和「在 .venv 里运行」效果完全一样。

    实现在 ``app/launcher.py`` —— ``build_index.py`` 与 ``eval_retrieval.py``
    也要用同一段逻辑，三处各写一份的话，以后改了一处另外两处会悄悄不一致。
    """
    from app.launcher import reenter_venv

    reenter_venv()


def main() -> None:
    _setup_console()
    _reenter_venv()

    parser = argparse.ArgumentParser(add_help=True)
    parser.add_argument("--reload", action="store_true", help="改代码后自动重启（开发用）")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT, help=f"监听端口，默认 {DEFAULT_PORT}")
    parser.add_argument("--no-browser", action="store_true", help="启动后不自动打开浏览器")
    args = parser.parse_args()

    # 切到项目根目录：保证 static/ 与 .env 的相对路径正确
    os.chdir(ROOT)

    # 结构化日志（一行一个 JSON）。放在这里而不是模块导入时，是为了让测试
    # 环境保持安静 —— pytest 里不配置 handler，INFO 级别的请求日志就不会刷屏。
    from app.logging_setup import configure

    configure(os.getenv("LOGWISE_LOG_LEVEL", "INFO"))

    _title("LogWise · LLM 智能日志诊断助手")
    print(f"  Python  : {sys.version.split()[0]}  ({sys.executable})")
    print(f"  项目目录: {ROOT}")
    print()

    check_python()
    check_dependencies()
    check_env_file()

    # 打印接口防护状态（延迟导入：万一依赖缺失，上面的检查已经给出友好提示了）
    try:
        from app.guard import startup_notice

        print()
        for line in startup_notice().splitlines():
            print(f"  {line}")
    except Exception:
        pass

    port = pick_port(args.port)
    if port != args.port:
        print(f"  [注意] 端口 {args.port} 已被占用，自动改用 {port}")
        print("         （如果你刚才用 Docker 启动过，那就是它在占用）")
    url = f"http://{HOST}:{port}"

    print()
    print(LINE)
    print("  服务已启动" + ("" if args.no_browser else "，浏览器马上自动打开"))
    print()
    print(f"    演示页面 : {url}")
    print(f"    接口文档 : {url}/docs")
    print()
    print("  关闭本窗口（或按 Ctrl+C）即可停止服务")
    print(LINE)
    print()

    if not args.no_browser:
        threading.Thread(target=open_browser_when_ready, args=(url,), daemon=True).start()

    import uvicorn

    try:
        uvicorn.run(
            "app.main:app",
            host=HOST,
            port=port,
            reload=args.reload,
            log_level="info",
        )
    except KeyboardInterrupt:
        pass
    finally:
        print()
        print("  服务已停止。")


if __name__ == "__main__":
    main()
