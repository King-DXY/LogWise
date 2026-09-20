"""演示：解析 sample.log 里的 ERROR 日志，调 DeepSeek 生成结构化诊断。

运行方式（任选一种）：
    1. 直接双击本文件
    2. 在终端里： .venv\\Scripts\\python.exe demo_diagnose.py

注意：会真实调用模型，产生少量费用（约 0.001 元）。
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

if __name__ == "__main__":
    # ⚠️ 必须在导入任何项目模块**之前**切到 .venv ——
    # 双击运行时用的是系统全局 Python，而 openai / python-dotenv
    # 这些依赖只装在 .venv 里，放到末尾调用就太晚了。
    from app.launcher import reenter_venv

    reenter_venv()

from app.diagnose import diagnose  # noqa: E402
from app.parser import parse_text  # noqa: E402


def main() -> None:
    text = (ROOT / "data" / "sample.log").read_text(encoding="utf-8")
    logs = parse_text(text)
    # 取第一条 ERROR（没有就退回第一条）—— 不写死下标，换日志文件也不会错位
    error_log = next((l for l in logs if l.level in ("ERROR", "CRITICAL", "FATAL")), logs[0])

    print("待诊断日志:")
    print(f"  [{error_log.level}] {error_log.message}")
    if error_log.stack_trace:
        print(f"  堆栈: {error_log.stack_trace.replace(chr(10), ' | ')[:120]}")
    print("\n" + "=" * 50)
    print("DeepSeek 诊断结果:")
    result = diagnose(error_log)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    from app.launcher import pause_if_console

    try:
        main()
    finally:
        pause_if_console()
