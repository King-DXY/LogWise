"""演示脚本：解析 sample.log 与 sample_hdfs.log，打印结构化结果。

运行方式（任选一种）：
    1. 直接双击本文件
    2. 在终端里： .venv\\Scripts\\python.exe demo_parse.py
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

if __name__ == "__main__":
    # ⚠️ 必须在导入任何项目模块**之前**切到 .venv ——
    # 双击运行时用的是系统全局 Python，依赖只装在 .venv 里。
    from app.launcher import reenter_venv

    reenter_venv()

from app.parser import parse_text  # noqa: E402


def main() -> None:
    # 路径锚定到项目根目录，而不是依赖「当前工作目录」——
    # 否则从别处调用（比如 IDE 把工作目录设成别的）就找不到文件。
    for filename in ("sample.log", "sample_hdfs.log"):
        path = ROOT / "data" / filename
        print("=" * 50)
        print(f"data/{filename}")
        print("=" * 50)
        text = path.read_text(encoding="utf-8")
        logs = parse_text(text)
        print("解析出日志条数:", len(logs))
        for i, log in enumerate(logs):
            print(f"--- 第 {i + 1} 条 ---")
            print("时间:", log.timestamp)
            print("级别:", log.level)
            print("模块:", log.module)
            print("消息:", log.message[:80])
            if log.stack_trace:
                print("堆栈:", log.stack_trace.replace("\n", " | ")[:120])
        print()


if __name__ == "__main__":
    from app.launcher import pause_if_console

    try:
        main()
    finally:
        # 双击运行时窗口会一闪而过，留个停顿
        pause_if_console()
