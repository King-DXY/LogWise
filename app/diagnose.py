"""LLM 诊断引擎：把结构化日志交给 DeepSeek，返回结构化诊断结果。

对外接口
--------
- ``diagnose(log)``         → dict           （旧接口，保持兼容）
- ``diagnose_ex(log)``      → (dict, Usage)  （带 token 用量）
- ``stream_diagnosis(...)`` → 异步产出文本增量，并把用量写进传入的 Usage

为什么保留 ``diagnose``：``demo_diagnose.py`` 依赖它返回纯 dict，改签名会连带改坏演示脚本。
"""
from __future__ import annotations

import json
import os
import threading
from collections.abc import AsyncIterator
from dataclasses import dataclass
from pathlib import Path

import httpx
from dotenv import load_dotenv
from openai import AsyncOpenAI, OpenAI

from app.apikey import read_api_key
from app.parser import ParsedLog
# PROMPT_VERSION 在本模块内未直接使用，但 main / 评测脚本一直从这里取，
# 保持这个再导出以免破坏既有调用方。
from app.prompts import PROMPT_VERSION, SYSTEM_PROMPT  # noqa: F401

# 显式指定 .env 路径，不依赖当前工作目录。override 默认 False，
# 因此外部注入的环境变量（docker-compose env_file / K8s Secret）优先级更高。
_ENV_PATH = Path(__file__).resolve().parent.parent / ".env"
if _ENV_PATH.exists():
    load_dotenv(_ENV_PATH)
else:
    load_dotenv()

_client: OpenAI | None = None
_async_client: AsyncOpenAI | None = None
_client_lock = threading.Lock()

# --------------------------------------------------------------------------- #
# 超时与重试
#
# ⚠️ 必须显式设置。OpenAI SDK 的默认值是 connect=5 / read=600 / write=600 / pool=600，
#    并且 max_retries=2 —— 最坏情况 600 秒 ×（1 次原始 + 2 次重试）≈ 30 分钟挂住。
#    而实测一次诊断平均只要 2 秒，这个上限毫无意义，只会让故障时无法快速失败
#    （对 SSE 接口来说，就是浏览器一直转圈）。
# --------------------------------------------------------------------------- #
DEFAULT_READ_TIMEOUT = 60.0
DEFAULT_CONNECT_TIMEOUT = 5.0
DEFAULT_MAX_RETRIES = 1


def _float_env(name: str, default: float) -> float:
    try:
        value = float(os.getenv(name, ""))
    except (TypeError, ValueError):
        return default
    return value if value > 0 else default


def read_timeout() -> float:
    """单次请求的读超时（秒）。"""
    return _float_env("LOGWISE_LLM_TIMEOUT", DEFAULT_READ_TIMEOUT)


def connect_timeout() -> float:
    return _float_env("LOGWISE_LLM_CONNECT_TIMEOUT", DEFAULT_CONNECT_TIMEOUT)


def max_retries() -> int:
    """SDK 层的网络重试次数。

    注意这**不是**「输出格式重试」—— 那是本模块自己的逻辑（见 diagnose_multi_ex）。
    """
    try:
        value = int(os.getenv("LOGWISE_LLM_MAX_RETRIES", DEFAULT_MAX_RETRIES))
    except (TypeError, ValueError):
        return DEFAULT_MAX_RETRIES
    return max(0, min(5, value))


def _client_timeout() -> httpx.Timeout:
    return httpx.Timeout(
        connect=connect_timeout(),
        read=read_timeout(),
        write=read_timeout(),
        pool=connect_timeout(),
    )


def client_options() -> dict:
    """构造 OpenAI 客户端时共用的参数，方便统一调整与测试。"""
    return {"timeout": _client_timeout(), "max_retries": max_retries()}


@dataclass
class Usage:
    """一次模型调用的 token 用量。"""

    prompt_tokens: int = 0        # 输入总 token（含命中上下文缓存的部分）
    completion_tokens: int = 0
    cached_tokens: int = 0        # 其中命中 DeepSeek 上下文缓存的部分

    @property
    def total(self) -> int:
        return self.prompt_tokens + self.completion_tokens

    @property
    def uncached_prompt_tokens(self) -> int:
        return max(0, self.prompt_tokens - self.cached_tokens)


def _extract_usage(raw) -> Usage:
    """从 SDK 的 usage 对象里提取用量。

    为什么要单独记 cached_tokens：DeepSeek 会自动缓存「重复出现的 Prompt 前缀」，
    我们每次请求的 System Prompt + Few-shot 都是固定的，正好命中缓存，
    而这部分单价远低于未命中的输入。不区分的话成本会被显著高估。

    两个来源都取一下，兼容不同实现：
    - OpenAI 标准字段 ``prompt_tokens_details.cached_tokens``
    - DeepSeek 专有的顶层字段 ``prompt_cache_hit_tokens``
    """
    if raw is None:
        return Usage()

    prompt = int(getattr(raw, "prompt_tokens", 0) or 0)
    completion = int(getattr(raw, "completion_tokens", 0) or 0)

    cached = 0
    details = getattr(raw, "prompt_tokens_details", None)
    if details is not None:
        cached = getattr(details, "cached_tokens", 0) or 0
    if not cached:
        cached = getattr(raw, "prompt_cache_hit_tokens", 0) or 0
    cached = min(int(cached or 0), prompt)  # 防御：命中数不该超过输入总数

    return Usage(prompt_tokens=prompt, completion_tokens=completion, cached_tokens=cached)


def current_model() -> str:
    """当前模型名。缓存键里会带上它，换模型后旧缓存自动失效。"""
    return os.getenv("DEEPSEEK_MODEL", "deepseek-chat")


def get_client() -> OpenAI:
    """懒加载单例同步客户端：只在第一次调用时创建，之后复用。

    加锁的原因：FastAPI 的同步端点跑在线程池里，两个请求同时首次进来时
    会各建一个客户端 —— 多出来的那个再也不会被关闭，连接池被白白占着。
    """
    global _client
    if _client is None:
        with _client_lock:
            if _client is None:
                api_key = _require_api_key()
                _client = OpenAI(
                    api_key=api_key,
                    base_url=os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com"),
                    **client_options(),
                )
    return _client


def get_async_client() -> AsyncOpenAI:
    """懒加载单例异步客户端：供 SSE 流式接口使用，避免同步 IO 阻塞事件循环。"""
    global _async_client
    if _async_client is None:
        with _client_lock:
            if _async_client is None:
                api_key = _require_api_key()
                _async_client = AsyncOpenAI(
                    api_key=api_key,
                    base_url=os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com"),
                    **client_options(),
                )
    return _async_client


def _require_api_key() -> str:
    """取诊断用的 API Key；没填（或还是占位符）就给出可读的报错。

    判定交给 ``app.apikey`` 统一处理 —— 之前这里自己匹配 ``"sk-在这里"``，
    和 embedding / run 里的写法各不相同，改一处漏一处会静默失效。
    """
    api_key = read_api_key("DEEPSEEK_API_KEY")
    if api_key is None:
        raise RuntimeError("未检测到有效的 DEEPSEEK_API_KEY，请检查 .env 文件")
    return api_key


def format_logs(logs: list[ParsedLog]) -> str:
    """把多条日志格式化成给模型看的文本。

    格式与 docs/EXPERIMENTS.md 第 4.1 节「已量化的短板」里的天花板实验**完全一致**
    （那个实验达到 100% 命中），所以这里刻意不做改动。
    每条前面带 [序号]，级别单独成行 —— 让模型能分辨 WARN 和 ERROR，
    并利用时间线的先后顺序判断谁是根因、谁是后果。
    """
    blocks: list[str] = []
    for i, log in enumerate(logs, 1):
        lines = [
            f"[{i}] 时间: {log.timestamp}",
            f"级别: {log.level}",
            f"模块: {log.module}",
            f"消息: {log.message}",
        ]
        if log.stack_trace:
            lines.append(f"堆栈:\n{log.stack_trace}")
        blocks.append("\n".join(lines))
    return "\n\n".join(blocks)


def build_messages(
    logs: ParsedLog | list[ParsedLog],
    mode: str | None = None,
    top_k: int | None = None,
    fixed_examples: list[dict] | None = None,
) -> list[dict]:
    """组装 messages：system + 示例 + 本次日志。

    参数
    ----
    logs
        一条日志（保持向后兼容），或上下文聚合选出的一组日志。
    mode
        rag 模式（fixed / dynamic / hybrid）。None 时读环境变量，默认 fixed。
    top_k
        检索返回的案例数。
    fixed_examples
        覆盖固定的 Few-shot 示例 —— 评测脚本用 ``--from-git`` 拉历史 Prompt 时会传。

    结构随模式变化::

        fixed   → system + 固定示例 + 日志
        dynamic → system + 参考资料说明 + 检索案例 + 日志
        hybrid  → system + 参考资料说明 + 固定示例 + 检索案例 + 日志
    """
    from app import rag  # 函数内导入：rag → retriever → prompts 与本模块无循环，但保持依赖清晰

    log_list = [logs] if isinstance(logs, ParsedLog) else list(logs)
    text = format_logs(log_list)

    examples, retrieved_ids = rag.build_examples(
        text, mode=mode, k=top_k, fixed_examples=fixed_examples
    )

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    if retrieved_ids:
        # 只在真的注入了检索案例时才加这段说明 —— 否则会与「固定示例仅示意格式」的约束打架
        messages.append({"role": "system", "content": rag.RAG_NOTICE})
    for ex in examples:
        messages.append({"role": "user", "content": ex["log"]})
        messages.append({"role": "assistant", "content": ex["diagnosis"]})
    messages.append({"role": "user", "content": text})
    return messages


def prompt_char_size(logs: ParsedLog | list[ParsedLog]) -> int:
    """Prompt 的总字符数，用于 usage 意外缺失时做兜底估算。"""
    return sum(len(m["content"]) for m in build_messages(logs))


# 诊断结果必须包含的字段。与 app/prompts.py 里约定的输出格式一一对应。
REQUIRED_FIELDS = ("error_type", "root_cause", "impact", "solution_steps", "confidence")
# 这几个字段必须是数组（evidence 是 v3 才加的，缺失可以接受，但不能是别的类型）
LIST_FIELDS = ("solution_steps", "evidence")

# 格式不合格时回给模型的重试指令
CORRECTION_PROMPT = """你上一条回复不符合要求：{problem}

请重新输出，并且**只输出一个 JSON 对象**：
1. 必须包含这些字段：error_type、root_cause、evidence、impact、solution_steps、confidence
2. evidence 和 solution_steps 必须是数组
3. confidence 必须是 0 到 1 之间的数字
4. 不要输出任何解释性文字，也不要用 Markdown 代码块包裹"""


class DiagnosisFormatError(RuntimeError):
    """模型返回的内容不符合约定的 JSON 结构（重试之后仍然如此）。"""


def _json_candidates(text: str) -> list[str]:
    """给出几种可能的 JSON 片段，按「越原始越优先」排列。

    真实遇到过的形态：裸 JSON、被 ```json 包裹、前面带一句解释再跟 JSON。
    """
    stripped = text.strip()
    out = [stripped]

    # 去掉代码块围栏（大小写与语言标注都可能变）
    if stripped.startswith("```"):
        body = stripped.split("\n", 1)[1] if "\n" in stripped else ""
        out.append(body.removesuffix("```").strip())

    # 兜底：截取第一个 { 到最后一个 } 之间的内容
    start, end = stripped.find("{"), stripped.rfind("}")
    if start != -1 and end > start:
        out.append(stripped[start : end + 1])

    return [c for c in out if c]


def _parse_json(content: str) -> dict:
    """把模型返回的文本解析成 dict。

    尽量宽容（裸 JSON / 代码块 / 前后带解释都能救回来），
    但**失败时一定会抛 JSONDecodeError**，绝不返回半成品。
    """
    text = content or ""
    last_error: json.JSONDecodeError | None = None
    for candidate in _json_candidates(text):
        try:
            parsed = json.loads(candidate)
        except json.JSONDecodeError as exc:
            last_error = exc
            continue
        if isinstance(parsed, dict):
            return parsed
        last_error = json.JSONDecodeError("顶层不是 JSON 对象", candidate, 0)
    raise last_error or json.JSONDecodeError("模型返回了空内容", text, 0)


def schema_problems(result: object) -> list[str]:
    """检查诊断结果的结构，返回问题列表（空列表表示合格）。

    为什么要查：``response_format={"type": "json_object"}`` 只保证「是合法 JSON」，
    **不保证字段齐全**。实测模型返回过 ``{"error_type": 1}`` 这种残缺结构，
    接口照常 200 返回，前端拿到缺字段的对象就会崩
    （之前出现过的 ``Cannot read properties of undefined`` 就是这个成因）。
    """
    if not isinstance(result, dict):
        return [f"顶层应该是对象，实际是 {type(result).__name__}"]

    problems: list[str] = []
    missing = [f for f in REQUIRED_FIELDS if f not in result]
    if missing:
        problems.append("缺少字段 " + "、".join(missing))

    for field in LIST_FIELDS:
        if field in result and not isinstance(result[field], list):
            problems.append(f"{field} 应该是数组")

    if "confidence" in result:
        conf = result["confidence"]
        if isinstance(conf, bool) or not isinstance(conf, (int, float)):
            problems.append("confidence 应该是数字")
        elif not 0 <= conf <= 1:
            problems.append(f"confidence={conf} 超出 0~1 范围")

    if "error_type" in result and not isinstance(result["error_type"], str):
        problems.append("error_type 应该是字符串")

    return problems


def _sum_usage(a: Usage, b: Usage) -> Usage:
    """重试时把两次调用的用量累加 —— 钱是真花了，不能不记。"""
    return Usage(
        prompt_tokens=a.prompt_tokens + b.prompt_tokens,
        completion_tokens=a.completion_tokens + b.completion_tokens,
        cached_tokens=a.cached_tokens + b.cached_tokens,
    )


def diagnose_ex(log: ParsedLog) -> tuple[dict, Usage]:
    """调用 DeepSeek（单条日志），返回 (结构化诊断 dict, token 用量)。"""
    return diagnose_multi_ex([log])


def diagnose_multi_ex(logs: list[ParsedLog], retries: int = 1) -> tuple[dict, Usage]:
    """调用 DeepSeek（一组日志上下文），返回 (结构化诊断 dict, token 用量)。

    拿到结果后**会校验结构**；不合格就把问题反馈给模型重试一次，
    仍然不合格才抛 ``DiagnosisFormatError``。

    为什么必须重试：实测偶发过模型返回大白话（如"这段日志信息不足，我无法给出结构化诊断"），
    于是 ``json.loads`` 抛 ``JSONDecodeError``，接口直接 502 —— 而把异常原文
    （``Expecting value: line 1 column 1``）甩给用户毫无意义。
    重试一次通常就能拿到合规结果。
    """
    messages = build_messages(logs)
    attempts = max(1, retries + 1)
    total = Usage()
    last_problem = ""
    last_excerpt = ""

    for attempt in range(attempts):
        resp = get_client().chat.completions.create(
            model=current_model(),
            messages=messages,
            temperature=0.2,  # 诊断任务要稳定可复现，温度调低
            response_format={"type": "json_object"},  # 要求模型输出 JSON 对象
        )
        content = resp.choices[0].message.content or ""
        total = _sum_usage(total, _extract_usage(resp.usage))

        try:
            result = _parse_json(content)
        except json.JSONDecodeError:
            last_problem = "返回的内容不是合法 JSON"
        else:
            problems = schema_problems(result)
            if not problems:
                return result, total
            last_problem = "；".join(problems)

        last_excerpt = content.strip()[:200]
        if attempt + 1 < attempts:
            messages = [
                *messages,
                {"role": "assistant", "content": content[:2000]},
                {"role": "user", "content": CORRECTION_PROMPT.format(problem=last_problem)},
            ]

    raise DiagnosisFormatError(
        f"模型连续 {attempts} 次没有返回符合格式的诊断（{last_problem}）。"
        f"它最后一次说的是：{last_excerpt or '（空）'}"
    )


def diagnose(log: ParsedLog) -> dict:
    """调用 DeepSeek，只返回结构化诊断 dict（保持原有签名不变）。"""
    result, _ = diagnose_ex(log)
    return result


async def stream_diagnosis(log: ParsedLog, usage: Usage) -> AsyncIterator[str]:
    """流式诊断（单条日志，保持向后兼容）。"""
    async for delta in stream_diagnosis_multi([log], usage):
        yield delta


async def stream_diagnosis_multi(logs: list[ParsedLog], usage: Usage) -> AsyncIterator[str]:
    """流式产出诊断文本增量（一组日志上下文），并把真实 token 用量写入 usage。

    ``stream_options={"include_usage": True}`` 让 DeepSeek 在流的最后一个 chunk
    里带出 usage（已实测支持）。整个流式过程**没有任何同步 IO** ——
    否则会阻塞事件循环、退化成"攒到最后一次性输出"，这是本项目踩过的坑。
    """
    stream = await get_async_client().chat.completions.create(
        model=current_model(),
        messages=build_messages(logs),
        temperature=0.2,
        stream=True,
        stream_options={"include_usage": True},
    )

    async for chunk in stream:
        # 最后一个 chunk 的 choices 为空，usage 就挂在那里
        chunk_usage = getattr(chunk, "usage", None)
        if chunk_usage is not None:
            got = _extract_usage(chunk_usage)
            usage.prompt_tokens = got.prompt_tokens
            usage.completion_tokens = got.completion_tokens
            usage.cached_tokens = got.cached_tokens

        if not chunk.choices:
            continue
        delta = chunk.choices[0].delta.content
        if delta:
            yield delta
