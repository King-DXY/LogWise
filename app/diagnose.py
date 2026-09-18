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
from collections.abc import AsyncIterator
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv
from openai import AsyncOpenAI, OpenAI

from app.parser import ParsedLog
from app.prompts import FEW_SHOT_EXAMPLES, PROMPT_VERSION, SYSTEM_PROMPT

# 显式指定 .env 路径，不依赖当前工作目录。override 默认 False，
# 因此外部注入的环境变量（docker-compose env_file / K8s Secret）优先级更高。
_ENV_PATH = Path(__file__).resolve().parent.parent / ".env"
if _ENV_PATH.exists():
    load_dotenv(_ENV_PATH)
else:
    load_dotenv()

_client: OpenAI | None = None
_async_client: AsyncOpenAI | None = None


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
    """懒加载单例同步客户端：只在第一次调用时创建，之后复用。"""
    global _client
    if _client is None:
        api_key = os.getenv("DEEPSEEK_API_KEY")
        if not api_key or api_key.startswith("sk-在这里"):
            raise RuntimeError("未检测到有效的 DEEPSEEK_API_KEY，请检查 .env 文件")
        _client = OpenAI(
            api_key=api_key,
            base_url=os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com"),
        )
    return _client


def get_async_client() -> AsyncOpenAI:
    """懒加载单例异步客户端：供 SSE 流式接口使用，避免同步 IO 阻塞事件循环。"""
    global _async_client
    if _async_client is None:
        api_key = os.getenv("DEEPSEEK_API_KEY")
        if not api_key or api_key.startswith("sk-在这里"):
            raise RuntimeError("未检测到有效的 DEEPSEEK_API_KEY，请检查 .env 文件")
        _async_client = AsyncOpenAI(
            api_key=api_key,
            base_url=os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com"),
        )
    return _async_client


def format_logs(logs: list[ParsedLog]) -> str:
    """把多条日志格式化成给模型看的文本。

    格式与 README「已量化的短板」章节里的天花板实验**完全一致**
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


def build_messages(logs: ParsedLog | list[ParsedLog]) -> list[dict]:
    """组装 messages：system + few-shot 示例 + 本次日志。

    参数既可以是一条日志（保持向后兼容），也可以是由上下文聚合选出的一组日志。
    """
    if isinstance(logs, ParsedLog):
        log_list = [logs]
    else:
        log_list = list(logs)

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    for ex in FEW_SHOT_EXAMPLES:
        messages.append({"role": "user", "content": ex["log"]})
        messages.append({"role": "assistant", "content": ex["diagnosis"]})
    messages.append({"role": "user", "content": format_logs(log_list)})
    return messages


def prompt_char_size(logs: ParsedLog | list[ParsedLog]) -> int:
    """Prompt 的总字符数，用于 usage 意外缺失时做兜底估算。"""
    return sum(len(m["content"]) for m in build_messages(logs))


def _parse_json(content: str) -> dict:
    """把模型返回的文本解析成 dict，容忍被 ```json 代码块包裹的情况。"""
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        cleaned = content.strip().removeprefix("```json").removesuffix("```").strip()
        return json.loads(cleaned)


def diagnose_ex(log: ParsedLog) -> tuple[dict, Usage]:
    """调用 DeepSeek（单条日志），返回 (结构化诊断 dict, token 用量)。"""
    return diagnose_multi_ex([log])


def diagnose_multi_ex(logs: list[ParsedLog]) -> tuple[dict, Usage]:
    """调用 DeepSeek（一组日志上下文），返回 (结构化诊断 dict, token 用量)。"""
    resp = get_client().chat.completions.create(
        model=current_model(),
        messages=build_messages(logs),
        temperature=0.2,  # 诊断任务要稳定可复现，温度调低
        response_format={"type": "json_object"},  # 强制模型输出合法 JSON
    )
    content = resp.choices[0].message.content or ""
    return _parse_json(content), _extract_usage(resp.usage)


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
