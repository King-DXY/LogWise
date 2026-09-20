"""阶段 5 评测：用 Loghub 真实日志（HDFS_2k.log, 2000 行）验证解析器，产出量化指标。

运行方式（任选一种）：
    1. 直接双击本文件
    2. 在终端里： .venv\\Scripts\\python.exe eval_hdfs.py

不调用大模型，**零成本**。
"""
from __future__ import annotations

import random
import re
import sys
import time
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

if __name__ == "__main__":
    # ⚠️ 必须在导入任何项目模块**之前**切到 .venv（双击时用的是系统 Python）
    from app.launcher import reenter_venv

    reenter_venv()

from app.parser import parse_text  # noqa: E402

_TS_HDFS = re.compile(r"^\d{6} \d{6}$")
_LEVELS = {"INFO", "WARN", "ERROR", "DEBUG", "FATAL", "WARNING", "CRITICAL"}
_MODULE_PATTERN = re.compile(r"^dfs\.\w+(\$\w+)?$")


def main() -> None:
    # 锚定项目根目录，不依赖当前工作目录
    raw = (ROOT / "data" / "HDFS_2k.log").read_text(encoding="utf-8")
    total = len([l for l in raw.splitlines() if l.strip()])

    t0 = time.perf_counter()
    logs = parse_text(raw)
    elapsed = time.perf_counter() - t0

    # 1. 覆盖率：完整解析出核心字段的行
    core_ok = [log for log in logs if log.timestamp and log.level and log.message]
    module_ok = [log for log in core_ok if log.module]

    # 2. 字段合法性校验（对照 HDFS 日志格式规则）
    bad_ts = [log for log in core_ok if not _TS_HDFS.match(log.timestamp)]
    bad_level = [log for log in core_ok if log.level not in _LEVELS]
    bad_module = [log for log in core_ok if log.module and not _MODULE_PATTERN.match(log.module)]

    # 3. 级别与模块分布
    level_dist = Counter(log.level for log in core_ok)
    module_top = Counter(log.module for log in module_ok).most_common(5)

    # 4. 抽样人工核对（固定随机种子，结果可复现）
    random.seed(42)
    samples = random.sample(module_ok, min(10, len(module_ok)))

    print(f"总行数: {total}")
    print(f"解析出日志条数: {len(logs)}")
    print(f"核心字段完整: {len(core_ok)} ({len(core_ok)/total*100:.2f}%)")
    print(f"模块字段完整: {len(module_ok)} ({len(module_ok)/total*100:.2f}%)")
    print(f"时间戳格式异常: {len(bad_ts)}")
    print(f"级别非法: {len(bad_level)}")
    print(f"模块模式异常: {len(bad_module)}")
    print(f"解析耗时: {elapsed*1000:.1f} ms")
    print(f"吞吐: {total/elapsed:.0f} 行/秒")
    print(f"\n级别分布: {dict(level_dist)}")
    print(f"模块 Top5: {dict(module_top)}")
    print("\n抽样核对（10 条）:")
    for log in samples:
        print(f"  [{log.level}] {log.timestamp} | {log.module} | {log.message[:60]}")


if __name__ == "__main__":
    from app.launcher import pause_if_console

    try:
        main()
    finally:
        pause_if_console()
