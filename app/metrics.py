"""运行时度量：记录每次请求的耗时、token 用量与估算成本。

设计取舍
--------
- **纯内存 + 零第三方依赖**：不引入 prometheus_client。对当前这个规模的服务，
  一个 /stats 接口已经够用；真要上生产再换成 Prometheus 也不迟。
- **环形缓冲存明细**：只保留最近 N 条，避免内存无限增长；聚合指标全量累计。
- **价格不硬编码**：DeepSeek 调价频繁，单价从环境变量读取，并在 /stats 里回显
  当前使用的单价，方便一眼看出配置是否过时。
"""
from __future__ import annotations

import math
import os
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any

# 默认单价（元 / 百万 token）。
# ⚠️ DeepSeek 调价频繁且分高峰/空闲时段，请到官方定价页核对后，
#    在 .env 里用 DEEPSEEK_PRICE_* 覆盖。
#    官方定价：https://api-docs.deepseek.com/quick_start/pricing
#
# 注意这里有三档价格：DeepSeek 会自动缓存「重复出现的 Prompt 前缀」
# （我们每次请求的 System Prompt + Few-shot 都是固定的，正好命中），
# 缓存命中的输入 token 单价远低于未命中的，必须分开计算，否则成本会显著高估。
_DEFAULT_PRICE_INPUT = 1.0          # 输入 · 缓存未命中
_DEFAULT_PRICE_INPUT_CACHED = 0.2   # 输入 · 缓存命中（约为未命中的 1/5）
_DEFAULT_PRICE_OUTPUT = 2.0         # 输出

# 明细最多保留多少条（只影响 /stats 里能看到的历史，不影响累计统计）
MAX_RECORDS = 200

# /stats 里展示几条最近记录
RECENT_LIMIT = 20


def _read_price(name: str, default: float) -> float:
    """延迟读取单价：确保 .env 已经加载完成，且允许运行时改配置后重启生效。"""
    try:
        value = float(os.getenv(name, "") or default)
    except ValueError:
        return default
    return value if value >= 0 else default


def _percentile(sorted_values: list[float], q: float) -> float:
    """最近秩（nearest-rank）分位数，``sorted_values`` 必须已升序排列。

    ⚠️ 不要写成 ``values[int(len(values) * q)]`` —— 那样少减了一次，
    取到的是第 q*n 个元素（从 0 计数），实测 **n ≤ 20 时结果恒等于最大值**，
    而 RECENT_LIMIT 正好是 20，等于 p95 这个指标永远在报 max。
    最近秩的正确定义是取第 ceil(q*n) 个（从 1 开始计数）。
    """
    if not sorted_values:
        return 0.0
    n = len(sorted_values)
    rank = max(1, math.ceil(q * n))
    return sorted_values[min(rank - 1, n - 1)]


def price_input() -> float:
    """输入 token 单价（缓存未命中），单位：元 / 百万 token。"""
    return _read_price("DEEPSEEK_PRICE_INPUT", _DEFAULT_PRICE_INPUT)


def price_input_cached() -> float:
    """输入 token 单价（命中上下文缓存），单位：元 / 百万 token。"""
    return _read_price("DEEPSEEK_PRICE_INPUT_CACHED", _DEFAULT_PRICE_INPUT_CACHED)


def price_output() -> float:
    """输出 token 单价，单位：元 / 百万 token。"""
    return _read_price("DEEPSEEK_PRICE_OUTPUT", _DEFAULT_PRICE_OUTPUT)


def estimate_tokens(text: str) -> int:
    """粗略估算 token 数，用于流式响应拿不到 usage 时的兜底。

    DeepSeek 的分词器对中文效率很高（一个汉字通常就是一个 token），
    英文大约 4 个字符一个 token。这里按字符类别估算，误差可接受，
    且不需要引入 tokenizer 依赖。

    估算出来的值会在记录里标记 tokens_estimated=True，避免和真实值混淆。
    """
    if not text:
        return 0
    cjk = 0
    for ch in text:
        code = ord(ch)
        # 中日韩统一表意文字 / 中文标点 / 全角字符
        if 0x4E00 <= code <= 0x9FFF or 0x3000 <= code <= 0x303F or 0xFF00 <= code <= 0xFFEF:
            cjk += 1
    other = len(text) - cjk
    return cjk + (other + 3) // 4


def estimate_tokens_from_chars(chars: int, cjk_ratio: float = 0.55) -> int:
    """只知道字符数时的粗略估算。

    仅在拿不到真实 usage 时用作兜底（实测 DeepSeek 会正常返回 usage，
    所以这条分支一般不会走到）。cjk_ratio 取 0.55 是因为本项目的 Prompt
    和诊断结果以中文为主。
    """
    if chars <= 0:
        return 0
    cjk = int(chars * cjk_ratio)
    other = chars - cjk
    return cjk + (other + 3) // 4


@dataclass
class CallRecord:
    """一次接口调用的度量记录。"""

    endpoint: str
    duration_ms: float
    prompt_tokens: int = 0
    cached_prompt_tokens: int = 0   # 其中命中上下文缓存的部分（单价更低）
    completion_tokens: int = 0
    cache_hit: bool = False         # 命中本服务的诊断缓存（完全不调模型）
    tokens_estimated: bool = False
    # 是否**真的**调用了大模型。
    # ⚠️ 不要用 endpoint 前缀来判断 —— 缓存命中和"没解析到日志"都走 /diagnose，
    #    但它们一次模型都没调。之前按前缀判断，导致 llm_calls 被算成
    #    "diagnose 请求数"，指标名与含义不符。现在由调用方显式传入。
    llm_call: bool = False
    error: str = ""
    # ---- Agent 模式专属（非 Agent 请求全部保持默认值，不参与统计）----
    # 加这些字段的原因：Agent 的行为特征（查不查工具、跑几轮）本身就是
    # 要观测的指标，混在通用指标里看不出来。
    agent: bool = False
    agent_tool_calls: int = 0    # 本次调用工具的次数（0 = 模型直接回答）
    agent_rounds: int = 0        # 决策轮数（含直接回答的 1 轮）
    agent_forced: bool = False   # 是否因轮次上限被强制收尾
    ts: float = field(default_factory=time.time)

    @property
    def uncached_prompt_tokens(self) -> int:
        return max(0, self.prompt_tokens - self.cached_prompt_tokens)

    def cost(self) -> float:
        """本次调用的估算成本（元）。

        三档分开算：缓存命中的输入最便宜，未命中的输入次之，输出最贵。
        """
        if self.cache_hit:
            return 0.0
        return (
            self.uncached_prompt_tokens / 1_000_000 * price_input()
            + self.cached_prompt_tokens / 1_000_000 * price_input_cached()
            + self.completion_tokens / 1_000_000 * price_output()
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "time": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(self.ts)),
            "endpoint": self.endpoint,
            "duration_ms": round(self.duration_ms, 2),
            "prompt_tokens": self.prompt_tokens,
            "cached_prompt_tokens": self.cached_prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "cache_hit": self.cache_hit,
            "tokens_estimated": self.tokens_estimated,
            "cost": round(self.cost(), 6),
            "error": self.error,
        }


class MetricsStore:
    """线程安全的度量存储。FastAPI 的同步端点跑在线程池里，必须加锁。"""

    def __init__(self, max_records: int = MAX_RECORDS) -> None:
        self._lock = threading.Lock()
        self._records: deque[CallRecord] = deque(maxlen=max_records)
        self._started_at = time.time()
        self._total = 0
        self._errors = 0
        self._cache_hits = 0
        self._llm_calls = 0
        self._billable_calls = 0
        self._prompt_tokens = 0
        self._cached_prompt_tokens = 0
        self._completion_tokens = 0
        self._cost = 0.0
        self._estimated_calls = 0
        # Agent 聚合：只统计 agent=True 的记录
        self._agent_calls = 0
        self._agent_tool_calls = 0
        self._agent_rounds = 0
        self._agent_forced = 0
        self._agent_direct_answers = 0  # 一次工具都没调的 Agent 请求

    def record(self, rec: CallRecord) -> None:
        with self._lock:
            self._total += 1
            if rec.error:
                self._errors += 1
            if rec.cache_hit:
                self._cache_hits += 1
            if rec.llm_call:
                self._llm_calls += 1
            # 计费调用的唯一判据是"产生了费用"，而不是"调用是否发生"。
            # 模型调用失败（如返回非 JSON）不会产生 token，也就不该被算作计费调用 ——
            # 否则 avg_per_billable_call 会被低估：分子只有真实花费，分母却含失败次数。
            if rec.cost() > 0:
                self._billable_calls += 1
            if rec.tokens_estimated:
                self._estimated_calls += 1
            if rec.agent:
                self._agent_calls += 1
                self._agent_tool_calls += rec.agent_tool_calls
                self._agent_rounds += rec.agent_rounds
                if rec.agent_forced:
                    self._agent_forced += 1
                if rec.agent_tool_calls == 0 and not rec.error:
                    self._agent_direct_answers += 1
            self._prompt_tokens += rec.prompt_tokens
            self._cached_prompt_tokens += rec.cached_prompt_tokens
            self._completion_tokens += rec.completion_tokens
            self._cost += rec.cost()
            self._records.append(rec)

    def summary(self, recent_limit: int = RECENT_LIMIT) -> dict[str, Any]:
        with self._lock:
            durations = sorted(r.duration_ms for r in self._records)
            recent = [r.to_dict() for r in list(self._records)[-recent_limit:]]
            total = self._total
            llm_calls = self._llm_calls
            cache_hits = self._cache_hits
            billable = self._billable_calls
            prompt_tokens = self._prompt_tokens
            cached_prompt_tokens = self._cached_prompt_tokens
            completion_tokens = self._completion_tokens
            cost = self._cost
            errors = self._errors
            estimated = self._estimated_calls
            started = self._started_at

        # 缓存命中率的分母是「需要诊断的请求」= 真的调了模型的 + 被缓存挡下来的。
        # 不能用 llm_calls 当分母 —— 修复 llm_calls 口径（不再计入缓存命中）之后，
        # 那样算出来会大于 1。
        diagnose_requests = llm_calls + cache_hits
        return {
            "uptime_seconds": round(time.time() - started, 1),
            "price": {
                "unit": "元 / 百万 token",
                "input_uncached": price_input(),
                "input_cached": price_input_cached(),
                "output": price_output(),
                "note": "单价来自 .env，请与官方定价页核对",
            },
            "requests": {
                "total": total,
                "errors": errors,
                "llm_calls": llm_calls,
                "cache_hits": cache_hits,
                "cache_hit_rate": (
                    round(cache_hits / diagnose_requests, 4) if diagnose_requests else 0.0
                ),
                "billable_calls": billable,
            },
            "latency_ms": {
                "avg": round(sum(durations) / len(durations), 2) if durations else 0.0,
                "p95": round(_percentile(durations, 0.95), 2),
                "max": round(durations[-1], 2) if durations else 0.0,
                "sample_size": len(durations),
                "note": "统计范围：最近保留的记录",
            },
            "tokens": {
                "prompt": prompt_tokens,
                "cached_prompt": cached_prompt_tokens,
                "uncached_prompt": max(0, prompt_tokens - cached_prompt_tokens),
                "completion": completion_tokens,
                "total": prompt_tokens + completion_tokens,
                "cache_rate": round(cached_prompt_tokens / prompt_tokens, 4) if prompt_tokens else 0.0,
                "estimated_calls": estimated,
                "note": "cached_prompt 是命中 DeepSeek 上下文缓存的输入 token，单价更低",
            },
            "cost": {
                "total": round(cost, 6),
                "avg_per_billable_call": round(cost / billable, 6) if billable else 0.0,
                "currency": "CNY",
            },
            "agent": self._agent_summary(),
            "recent": recent,
        }

    def _agent_summary(self) -> dict[str, Any]:
        """Agent 模式的行为特征 —— 查不查工具、跑几轮，本身就是核心观测对象。

        ⚠️ 调用方必须已持有 self._lock（在 summary 里被调用）。
        """
        calls = self._agent_calls
        if not calls:
            return {"requests": 0}
        return {
            "requests": calls,
            "tool_call_rate": round((calls - self._agent_direct_answers) / calls, 4),
            "direct_answer_rate": round(self._agent_direct_answers / calls, 4),
            "avg_tool_calls": round(self._agent_tool_calls / calls, 2),
            "avg_rounds": round(self._agent_rounds / calls, 2),
            "forced_answer_count": self._agent_forced,
            "note": "direct_answer_rate 高说明模型判断故障是通识、无需检索 —— 与 RAG 实验结论互为印证",
        }

    def reset(self) -> None:
        """清空统计（方便做对照实验：清零后跑一轮，看这一轮的数字）。"""
        with self._lock:
            self._records.clear()
            self._started_at = time.time()
            self._total = 0
            self._errors = 0
            self._cache_hits = 0
            self._llm_calls = 0
            self._billable_calls = 0
            self._prompt_tokens = 0
            self._cached_prompt_tokens = 0
            self._completion_tokens = 0
            self._cost = 0.0
            self._estimated_calls = 0
            self._agent_calls = 0
            self._agent_tool_calls = 0
            self._agent_rounds = 0
            self._agent_forced = 0
            self._agent_direct_answers = 0


# 全局单例：整个进程共用一个
store = MetricsStore()
