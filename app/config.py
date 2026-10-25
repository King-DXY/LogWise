"""统一的配置读取入口 + 生效配置回显。

为什么要有这个模块
------------------
之前每个模块各写一份 ``os.getenv`` 加 ``try/except ValueError`` 的解析逻辑：

    app/guard.py      _flag()
    app/context.py    _flag() + _int_env()
    app/cache.py      _flag()
    app/metrics.py    _read_price()
    app/diagnose.py   _float_env()

问题不在重复本身，而在**默认值、取值范围、非法输入的处理各不相同**：
同一个 ``LOGWISE_RATE_LIMIT=abc``，一处当成「没配过、用默认」，
另一处会直接抛异常。加一个配置项还得同步改三处（代码 / ``.env.example`` / 文档），
漏一处就静默失效 —— 这正是 PITFALLS #23 记的那类问题。

现在的规矩
----------
1. **所有读取都走这里**，解析规则只有一份：
   空串 = 未设置；非法值 = 回落默认值（绝不抛异常，启动不该因为一个手滑挂掉）。
2. **取值范围集中声明**（min / max），越界自动夹紧，而不是让调用方各自 ``max()``。
3. ``snapshot()`` 一次给出**当前真正生效**的配置。``GET /config`` 与 ``GET /stats``
   直接回显它 —— 排查「我明明改了 .env 怎么没效果」这是最快的入口。

⚠️ 全部写成**函数**而不是模块级常量：测试大量用 monkeypatch 改环境变量，
   缓存成常量会让那些测试悄悄失效（本项目已经踩过一次「配置项不生效」）。
"""
from __future__ import annotations

import os
from contextvars import ContextVar

# --------------------------------------------------------------------------- #
# 基础读取
# --------------------------------------------------------------------------- #
_TRUE = ("1", "true", "yes", "on")


def env_str(name: str, default: str = "") -> str:
    """读字符串配置。未设置时返回默认值；设置了就原样返回（去首尾空白）。"""
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip()


def env_int(
    name: str,
    default: int,
    *,
    minimum: int | None = None,
    maximum: int | None = None,
) -> int:
    """读整数配置，非法值回落默认，并按需夹到 [minimum, maximum]。"""
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        value = default
    else:
        try:
            value = int(raw.strip())
        except (TypeError, ValueError):
            value = default
    if minimum is not None:
        value = max(minimum, value)
    if maximum is not None:
        value = min(maximum, value)
    return value


def env_float(
    name: str,
    default: float,
    *,
    minimum: float | None = None,
    maximum: float | None = None,
) -> float:
    """读浮点配置，非法值回落默认，并按需夹到 [minimum, maximum]。"""
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        value = default
    else:
        try:
            value = float(raw.strip())
        except (TypeError, ValueError):
            value = default
    if minimum is not None:
        value = max(minimum, value)
    if maximum is not None:
        value = min(maximum, value)
    return value


def env_flag(name: str, default: bool) -> bool:
    """读布尔配置。

    没有值 → 默认值；``0/false/no/off`` 与空串 → False；其余非空值 → True。
    与之前各省一份的实现保持同一套判定，避免行为漂移。
    """
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() not in ("0", "false", "no", "off", "")


# --------------------------------------------------------------------------- #
# 输入上限
# --------------------------------------------------------------------------- #
# 单次请求允许粘贴的最大字符数，默认 200 万字符（约 3.5 万行日志、2MB）。
#
# 为什么定得这么大而不是 20 万：
# 上下文聚合只把最多 20 条日志送进模型，所以**模型侧成本不随粘贴长度增长**；
# 真正会被拖慢的是解析（纯 Python 正则）和内存。而「贴两万行日志看聚合效果」
# 恰恰是这个项目最有价值的演示场景（/context 的展示上限就是按 2 万条设计的）。
# 所以限制的作用是挡住「把整个 100MB 日志文件拖进来」，不是挡住正常使用。
DEFAULT_MAX_LOG_CHARS = 2_000_000


def max_log_chars() -> int:
    """单次请求的日志长度上限（字符）。0 表示关闭限制。"""
    return env_int("LOGWISE_MAX_LOG_CHARS", DEFAULT_MAX_LOG_CHARS, minimum=0)


# --------------------------------------------------------------------------- #
# 成本预算
# --------------------------------------------------------------------------- #
# 每天允许花多少钱（元），0 表示不限制。
# 限流只管「速率」，管不住「一天总共花了多少」：默认 20 次/分钟，最坏能跑
# 两万多次/天，按 0.002 元一次就是几十块 —— 而默认 .env 里鉴权是关的。
# 这道闸门是钱包的最后一道防线。
DEFAULT_DAILY_BUDGET = 5.0


def daily_budget_yuan() -> float:
    return env_float("LOGWISE_DAILY_BUDGET_YUAN", DEFAULT_DAILY_BUDGET, minimum=0.0)


# --------------------------------------------------------------------------- #
# 请求 ID（贯穿一条请求的关联标识）
# --------------------------------------------------------------------------- #
REQUEST_ID_HEADER = "X-Request-ID"

_request_id_var: ContextVar[str] = ContextVar("logwise_request_id", default="")


def set_request_id(value: str) -> None:
    """记录当前请求的关联 ID。

    ⚠️ 这里**只设置、不重置**：SSE 的响应体是在中间件返回之后才迭代的，
    提前重置会让流式路径里的度量记录拿不到 ID。每个请求跑在各自的
    task / 上下文中，不会串到下一个请求。
    """
    _request_id_var.set(value)


def current_request_id() -> str:
    return _request_id_var.get()


# --------------------------------------------------------------------------- #
# 生效配置快照（给 /config 与 /stats 用）
# --------------------------------------------------------------------------- #
def snapshot() -> dict:
    """返回当前**真正生效**的配置。

    延迟导入各模块：本模块被它们反向依赖，顶部 import 会成环。
    这里读到的都是各模块的 accessor 结果，而不是环境变量的原始值 ——
    差别在于 accessor 已经做过夹紧与默认值处理。
    """
    from app import cache as cache_mod
    from app import context as context_mod
    from app import guard, metrics, rag
    from app.diagnose import (
        connect_timeout,
        current_model,
        max_retries,
        read_timeout,
    )
    from app.prompts import PROMPT_VERSION

    return {
        "model": {
            "name": current_model(),
            "prompt_version": PROMPT_VERSION,
            "base_url": env_str("DEEPSEEK_BASE_URL", "https://api.deepseek.com"),
        },
        "llm": {
            "read_timeout_s": read_timeout(),
            "connect_timeout_s": connect_timeout(),
            "max_retries": max_retries(),
        },
        "context": {
            "enabled": context_mod.context_enabled(),
            "window": context_mod.context_window(),
            "max_logs": context_mod.context_max_logs(),
        },
        "rag": {
            "enabled": rag.rag_enabled(),
            "mode": rag.rag_mode(),
            "top_k": rag.rag_top_k(),
        },
        "cache": {
            "enabled": cache_mod.cache_enabled(),
            "ttl_seconds": cache_mod.cache_ttl(),
            "max_entries": cache_mod.cache_max_size(),
        },
        "guard": {
            "auth_enabled": guard.auth_enabled(),
            "trust_localhost": guard.trust_localhost(),
            "rate_limit_per_window": guard.rate_capacity(),
            "rate_window_seconds": guard.rate_window(),
        },
        "limits": {
            "max_log_chars": max_log_chars(),
            "max_context_display": env_int("LOGWISE_MAX_CONTEXT_DISPLAY", 500, minimum=1),
            "daily_budget_yuan": daily_budget_yuan(),
        },
        "pricing": {
            "unit": "元 / 百万 token",
            "input_uncached": metrics.price_input(),
            "input_cached": metrics.price_input_cached(),
            "output": metrics.price_output(),
        },
    }
