"""LLM 诊断引擎：把结构化日志交给 DeepSeek，返回结构化诊断结果。"""
import json
import os

from dotenv import load_dotenv
from openai import AsyncOpenAI, OpenAI

from app.parser import ParsedLog
from app.prompts import FEW_SHOT_EXAMPLES, SYSTEM_PROMPT

load_dotenv()

_client: OpenAI | None = None
_async_client: AsyncOpenAI | None = None


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


def _log_to_text(log: ParsedLog) -> str:
    """把结构化日志拼成给模型看的文本。"""
    parts = [
        f"时间: {log.timestamp}",
        f"级别: {log.level}",
        f"模块: {log.module}",
        f"消息: {log.message}",
    ]
    if log.stack_trace:
        parts.append(f"堆栈:\n{log.stack_trace}")
    return "\n".join(parts)


def build_messages(log: ParsedLog) -> list[dict]:
    """组装 messages：system + few-shot 示例 + 本次日志。"""
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    for ex in FEW_SHOT_EXAMPLES:
        messages.append({"role": "user", "content": ex["log"]})
        messages.append({"role": "assistant", "content": ex["diagnosis"]})
    messages.append({"role": "user", "content": _log_to_text(log)})
    return messages


def diagnose(log: ParsedLog) -> dict:
    """调用 DeepSeek，返回结构化的诊断 dict。"""
    resp = get_client().chat.completions.create(
        model=os.getenv("DEEPSEEK_MODEL", "deepseek-chat"),
        messages=build_messages(log),
        temperature=0.2,  # 诊断任务要稳定可复现，温度调低
        response_format={"type": "json_object"},  # 强制模型输出合法 JSON
    )
    content = resp.choices[0].message.content
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        # 容错：万一返回了带代码块包裹的 JSON，剥掉再解析
        cleaned = content.strip().removeprefix("```json").removesuffix("```").strip()
        return json.loads(cleaned)
