"""LogWise 日志解析模块：把原始日志文本解析成结构化字段。

支持两种日志格式：
1. 通用格式：2025-09-14 10:23:45,123 INFO [module] message
2. HDFS 格式（Loghub 数据集）：081109 203615 148 INFO dfs.DataNode: message
"""
import re
from dataclasses import dataclass


@dataclass
class ParsedLog:
    """一条日志的结构化表示。"""
    timestamp: str = ""
    level: str = ""
    module: str = ""
    message: str = ""
    stack_trace: str = ""


# ---- 时间戳规则 ----
# 通用：2025-09-14 10:23:45,123 / 2025-09-14T10:23:45.123
_TIMESTAMP_GENERIC = r"\d{4}-\d{2}-\d{2}[ T]\d{2}:\d{2}:\d{2}(?:[,.]\d{3})?"
# HDFS：081109 203615（前 6 位日期 + 后 6 位时间，无分隔符）
_TIMESTAMP_HDFS = r"\d{6} \d{6}"

# 级别：长的放前面，避免 WARNING 被 WARN 抢先
_LEVEL = r"WARNING|CRITICAL|ERROR|INFO|DEBUG|WARN|FATAL"

# 新日志行模式：时间戳（二选一）+ 可选序号 + 级别 + 模块（[x] 或 类名:） + 消息
_LINE_START = re.compile(
    rf"^(?P<timestamp>{_TIMESTAMP_GENERIC}|{_TIMESTAMP_HDFS})\s+"
    rf"(?:\d+\s+)?"
    rf"(?P<level>{_LEVEL})\s+"
    rf"(?:\[(?P<module>[^\]]+)\]|(?P<module_hdfs>[^\s:]+):)?\s*"
    rf"(?P<message>.*)$",
    re.IGNORECASE,
)


def _split_blocks(text: str) -> list[str]:
    """按'行首特征'切分日志块：一行若匹配新日志行模式，就是新日志的开始。

    不再依赖空行分隔——真实日志（如 HDFS）是连续无空行的。
    """
    blocks: list[str] = []
    current: list[str] = []
    for line in text.splitlines():
        if not line.strip():
            continue  # 跳过空行
        if _LINE_START.match(line):
            if current:
                blocks.append("\n".join(current))
            current = [line]
        else:
            current.append(line)  # 堆栈/续行，归入当前日志块
    if current:
        blocks.append("\n".join(current))
    return blocks


def parse_log(block: str) -> ParsedLog:
    """解析单个日志块：首行（时间/级别/模块/消息）+ 后续堆栈行。"""
    lines = block.strip().splitlines()
    if not lines:
        return ParsedLog()

    m = _LINE_START.match(lines[0])
    if not m:
        # 首行不符合已知格式：整块视为消息（不崩、不误判）
        return ParsedLog(message="\n".join(lines))

    module = m.group("module") or m.group("module_hdfs") or ""
    parsed = ParsedLog(
        timestamp=m.group("timestamp"),
        level=m.group("level").upper(),
        module=module,
        message=m.group("message").strip(),
    )
    if len(lines) > 1:
        parsed.stack_trace = "\n".join(lines[1:])
    return parsed


def parse_text(text: str) -> list[ParsedLog]:
    """解析整段日志：按行首特征切块，逐块解析。"""
    return [parse_log(b) for b in _split_blocks(text)]
