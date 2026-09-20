"""让脚本在「双击运行」时也能正确工作。

Windows 上双击 .py 文件用的是系统全局 Python，而项目依赖只装在 .venv 里 ——
直接双击会因为找不到依赖而失败。这里解决两件事：

1. ``reenter_venv()``   自动切到项目的 .venv 解释器重新执行本脚本
2. ``pause_if_console()`` 结束前停一下，否则窗口一闪而过，看不到输出

`run.py`、`build_index.py`、`eval_retrieval.py` 共用这两个函数，
避免同一段逻辑在多处各写一份、以后只改了一处。
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def reenter_venv() -> None:
    """确保用项目自带的 .venv 解释器运行。

    已经是目标解释器时不做事，因此不会无限递归。
    """
    venv_dir = ROOT / ".venv"
    venv_python = venv_dir / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
    if not venv_python.exists():
        return  # 没有 .venv，就用当前解释器；后续的依赖检查会给出提示

    try:
        if Path(sys.prefix).resolve() == venv_dir.resolve():
            return  # 已经在这个虚拟环境里了
        if Path(sys.executable).resolve() == venv_python.resolve():
            return
    except Exception:
        pass

    # flush=True：父进程的输出必须立刻刷出来。
    # 否则子进程（.venv 里那个）先跑起来并打印到同一个终端，
    # 用户会看到「按回车键关闭窗口」出现在「[切换] …」之前，顺序反了很困惑。
    print(f"  [切换] 正在切换到项目虚拟环境：{venv_python}", flush=True)
    completed = subprocess.run(
        [str(venv_python), str(_entry_script()), *sys.argv[1:]],
        cwd=str(ROOT),
    )
    sys.exit(completed.returncode)


def _entry_script() -> Path:
    """当前正在运行的那个脚本文件（用于切换解释器后重新执行它）。"""
    main = sys.modules.get("__main__")
    path = getattr(main, "__file__", None)
    return Path(path).resolve() if path else Path(sys.argv[0]).resolve()


def pause_if_console(message: str = "按回车键关闭窗口…") -> None:
    """双击运行时暂停一下，避免结果一闪而过。

    只在真正的交互式终端里暂停：用管道或重定向跑（比如自动化脚本）不会卡住。
    需要跳过时设环境变量 ``LOGWISE_NO_PAUSE=1``。
    """
    if os.getenv("LOGWISE_NO_PAUSE"):
        return
    try:
        if not sys.stdin or not sys.stdin.isatty():
            return
    except Exception:
        return
    try:
        input(f"\n{message}")
    except (EOFError, KeyboardInterrupt):
        pass
