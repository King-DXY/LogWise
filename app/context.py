"""日志上下文聚合：决定把哪些日志送给模型。

背景
----
原实现是「从解析结果里挑第一条 ERROR，其余全部丢弃」。实测（见 README
「已量化的短板」）在多日志故障场景下命中率只有 **16.7%**，而把全部日志送进去
能达到 **100%** —— 差距完全来自上下文缺失，不是模型能力问题。

策略
----
不做复杂的语义关联，用分层优先级逐步纳入，**每一层都不依赖日志格式**：

  ① 锚点            必含。第一条 CRITICAL/ERROR；没有则第一条 WARN；再没有则第一条
  ② 窗口内的异常日志  锚点前后 K 条以内、且级别 ≥ WARN（最可能是真因）
  ③ 窗口内的普通日志  锚点前后 K 条以内（提供上下文，比如发布通知这类 INFO）
  ④ 窗口外的异常日志  范围里其他 WARN/ERROR（离得远但可能是真因）

超过上限 M 条时，按上面的优先级从后往前淘汰。最终**按原始顺序输出** ——
时间线本身就是诊断线索：谁先谁后，直接决定谁是根因、谁是后果。

为什么不做时间裁剪
------------------
曾考虑「跨度超过 N 分钟就裁到锚点附近」，但评测用例 multi-03 里：
证书续期失败的 WARN 在 06:00，TLS 握手失败的 ERROR 在 09:14，跨度 3 小时，
却确实是同一次故障。**用户既然把这段范围粘贴进来，说明他已经做过预筛选**，
我们的任务是在这个范围里挑出相关的，而不是再按时间裁一遍。

可一键回退
----------
``LOGWISE_CONTEXT_ENABLED=false`` 即退回到「只取锚点」的旧行为，
方便随时做 A/B 对比或出问题时回退。
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field

from app.parser import ParsedLog

# 默认参数。这两个值是**参数扫描**选出来的，不是拍脑袋定的：
# 评测集 38 条用例中，有 10 条专门用来区分参数（多日志 6 条 + 长日志噪音 4 条），
# 四组参数的「上下文类命中率」呈单调梯度：
#     关闭聚合 10%  →  ±2/8  60%  →  ±4/12  80%  →  ±8/20  100%
# 而四组的输入 token 相差不到 5%（单次成本差异在缓存波动范围内），
# 常规档 24/24 也全程未被拖累 —— 所以直接取最大的那一组。
DEFAULT_WINDOW = 8
DEFAULT_MAX_LOGS = 20

# 视为「异常」的级别：这些级别的日志更可能是故障线索
ANOMALY_LEVELS = {"WARN", "WARNING", "ERROR", "CRITICAL", "FATAL"}
# 锚点优先考虑的级别（沿用原实现的口径，避免行为漂移）
ANCHOR_LEVELS = ("ERROR", "CRITICAL", "FATAL")


def _flag(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() not in ("0", "false", "no", "off", "")


def _int_env(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, default))
    except (TypeError, ValueError):
        return default


def context_enabled() -> bool:
    return _flag("LOGWISE_CONTEXT_ENABLED", True)


def context_window() -> int:
    """位置窗口：锚点前后各取几条。"""
    return max(0, _int_env("LOGWISE_CONTEXT_WINDOW", DEFAULT_WINDOW))


def context_max_logs() -> int:
    """上下文包最多包含几条日志。"""
    return max(1, _int_env("LOGWISE_CONTEXT_MAX_LOGS", DEFAULT_MAX_LOGS))


def _is_anomaly(level: str) -> bool:
    return (level or "").upper() in ANOMALY_LEVELS


@dataclass
class LogContext:
    """一次聚合的结果。"""

    logs: list[ParsedLog] = field(default_factory=list)
    anchor: ParsedLog | None = None
    anchor_index: int = -1
    total_parsed: int = 0
    truncated: bool = False
    enabled: bool = True

    @property
    def size(self) -> int:
        return len(self.logs)

    def cache_signature(self) -> str:
        """放进缓存键的聚合参数签名。

        为什么需要它：改了 K 或 M 之后，同样的输入会产生不同的上下文包，
        进而产生不同的诊断结论。不把参数纳入缓存键的话，调参会以为「没效果」——
        实际是命中了旧缓存。这是缓存类功能很容易踩的一个坑。
        """
        if not self.enabled:
            return "ctx=off"
        return f"ctx=on:{context_window()}:{context_max_logs()}"


def select_anchor_index(logs: list[ParsedLog]) -> int:
    """选锚点。

    口径与原实现保持一致（第一条 ERROR/CRITICAL/FATAL），
    只是多补了两级兜底：没有 ERROR 时退到第一条 WARN，再退到第一条。
    """
    for i, log in enumerate(logs):
        if (log.level or "").upper() in ANCHOR_LEVELS:
            return i
    for i, log in enumerate(logs):
        if (log.level or "").upper() in ("WARN", "WARNING"):
            return i
    return 0


def _priority(index: int, anchor_index: int, window: int, level: str) -> tuple[int, int]:
    """返回 (优先级层级, 距锚点距离)，数值越小越优先保留。"""
    distance = abs(index - anchor_index)
    if index == anchor_index:
        return (0, 0)
    in_window = distance <= window
    anomaly = _is_anomaly(level)
    if in_window and anomaly:
        return (1, distance)
    if in_window:
        return (2, distance)
    if anomaly:
        return (3, distance)
    return (4, distance)


def build_context(logs: list[ParsedLog]) -> LogContext:
    """从解析结果里选出要送进模型的日志集合。"""
    if not logs:
        return LogContext()

    anchor_index = select_anchor_index(logs)
    total = len(logs)

    if not context_enabled():
        # 旧行为：只取锚点一条
        return LogContext(
            logs=[logs[anchor_index]],
            anchor=logs[anchor_index],
            anchor_index=anchor_index,
            total_parsed=total,
            truncated=False,
            enabled=False,
        )

    window = context_window()
    max_logs = context_max_logs()

    # 收集候选：锚点 + 窗口内全部 + 范围里的所有异常日志
    candidates: set[int] = {anchor_index}
    lo = max(0, anchor_index - window)
    hi = min(total - 1, anchor_index + window)
    candidates.update(range(lo, hi + 1))
    for i, log in enumerate(logs):
        if _is_anomaly(log.level):
            candidates.add(i)

    truncated = len(candidates) > max_logs
    if truncated:
        ranked = sorted(
            candidates,
            key=lambda i: _priority(i, anchor_index, window, logs[i].level),
        )
        keep = set(ranked[:max_logs])
        keep.add(anchor_index)  # 锚点无论如何都要留
        candidates = keep

    selected = [logs[i] for i in sorted(candidates)]
    return LogContext(
        logs=selected,
        anchor=logs[anchor_index],
        anchor_index=anchor_index,
        total_parsed=total,
        truncated=truncated,
        enabled=True,
    )


def describe(context: LogContext) -> dict:
    """给 /stats 或调试用的一句话摘要。"""
    return {
        "enabled": context.enabled,
        "total_parsed": context.total_parsed,
        "selected": context.size,
        "truncated": context.truncated,
        "window": context_window() if context.enabled else 0,
        "max_logs": context_max_logs() if context.enabled else 1,
    }
