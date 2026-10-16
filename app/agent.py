"""Agent 模式诊断：模型自主决定是否调用工具的 Function Calling 循环。

与非 Agent 诊断（app/diagnose.py）的关系
------------------------------------------
- diagnose.py：**人写死决策** —— 检不检索、检什么，由环境变量和代码决定。
- 本模块：**模型做决策** —— 工具摆在那，查不查、查哪个、查几次，模型自己定。
  RAG 实验的结论（「通用故障模型本来就会，检索是白花钱」）被写进
  AGENT_SYSTEM_PROMPT，由模型自己去执行这个策略。

循环结构（经典 ReAct 的简化版）
-------------------------------
    messages → 模型 → 直接输出诊断 JSON → 结束
                    ↘ 返回 tool_calls → 本地执行 → 结果塞回 messages → 再问模型

防护
----
- **轮次硬上限** ``MAX_TOOL_ROUNDS``：超过后强制模型在无工具模式下给结论。
  防成本失控、防死循环 —— 这是 Agent 工程的标配。
- **用量全程累加**：每一轮调用的 token 都计入返回的 Usage，包括被截断的轮次。
  钱是真花了，不能不记。
- **trace 全留痕**：每轮的决策与工具调用都记录并随接口返回 —— Agent 可观测性。
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field

from app.diagnose import (
    CORRECTION_PROMPT,
    Usage,
    _extract_usage,
    _parse_json,
    _sum_usage,
    current_model,
    format_logs,
    get_client,
    schema_problems,
)
from app.parser import ParsedLog
from app.prompts import AGENT_SYSTEM_PROMPT, FEW_SHOT_EXAMPLES
from app.tools import TOOL_SCHEMAS, dispatch_tool

# Agent Prompt 独立版本号：参与缓存键。改 AGENT_SYSTEM_PROMPT 或工具定义时 +1。
# 与 PROMPT_VERSION 分开 —— 改诊断 Prompt 不该让 Agent 缓存失效，反之亦然。
AGENT_PROMPT_VERSION = "a2"

DEFAULT_MAX_TOOL_ROUNDS = 3

# trace 里每条工具结果摘要的最大长度（完整结果在 messages 里给模型看，
# trace 是给**人**看的，不需要全文）
TRACE_RESULT_EXCERPT = 200


def max_tool_rounds() -> int:
    """工具调用轮次上限（环境变量 LOGWISE_AGENT_MAX_ROUNDS，默认 3）。"""
    try:
        value = int(os.getenv("LOGWISE_AGENT_MAX_ROUNDS", DEFAULT_MAX_TOOL_ROUNDS))
    except (TypeError, ValueError):
        return DEFAULT_MAX_TOOL_ROUNDS
    return max(1, min(10, value))


@dataclass
class AgentTrace:
    """一次 Agent 诊断的决策轨迹。"""

    rounds: list[dict] = field(default_factory=list)
    forced_answer: bool = False  # 是否因达到轮次上限被强制收尾

    @property
    def tool_call_count(self) -> int:
        return sum(len(r.get("tool_calls", [])) for r in self.rounds)

    def as_dict(self) -> dict:
        return {
            "rounds": self.rounds,
            "tool_calls_total": self.tool_call_count,
            "forced_answer": self.forced_answer,
        }


def build_agent_messages(logs: list[ParsedLog]) -> list[dict]:
    """组装 Agent 模式的初始 messages：system + 固定 Few-shot + 本次日志。

    保留固定 Few-shot 的原因：它们维持输出格式与判断思路的稳定性，
    而且**每次都一样**，能继续命中 DeepSeek 的 Prompt 前缀缓存 ——
    这是 RAG 实验花钱买来的教训：动态注入会打断缓存，固定前缀不能动。
    """
    messages = [{"role": "system", "content": AGENT_SYSTEM_PROMPT}]
    for ex in FEW_SHOT_EXAMPLES:
        messages.append({"role": "user", "content": ex["log"]})
        messages.append({"role": "assistant", "content": ex["diagnosis"]})
    messages.append({"role": "user", "content": format_logs(logs)})
    return messages


def run_agent(
    logs: list[ParsedLog],
    max_rounds: int | None = None,
) -> tuple[dict, Usage, AgentTrace]:
    """跑一轮 Agent 诊断，返回 (结构化诊断, 累计用量, 决策轨迹)。

    失败处理与 diagnose_multi_ex 对齐：最终输出格式不合格时反馈重试一次，
    仍不合格抛 ``DiagnosisFormatError``（由它带上最后一次的内容摘录）。
    """
    from app.diagnose import DiagnosisFormatError  # 避免顶层循环导入

    rounds_limit = max_rounds if max_rounds is not None else max_tool_rounds()
    messages = build_agent_messages(logs)
    usage = Usage()
    trace = AgentTrace()

    # ------------------------------------------------------------------ #
    # 工具调用循环
    # ------------------------------------------------------------------ #
    for round_no in range(1, rounds_limit + 1):
        resp = get_client().chat.completions.create(
            model=current_model(),
            messages=messages,
            tools=TOOL_SCHEMAS,
            temperature=0.2,
            response_format={"type": "json_object"},
        )
        usage = _sum_usage(usage, _extract_usage(resp.usage))
        msg = resp.choices[0].message

        tool_calls = getattr(msg, "tool_calls", None) or []
        if not tool_calls:
            # 模型选择直接回答 —— 跳出循环走格式校验
            break

        # 把 assistant 的 tool_calls 原样塞回，再逐条执行工具、追加结果。
        # 这是 OpenAI 协议要求的消息结构，顺序不能乱。
        messages.append(msg)
        round_trace = {"round": round_no, "decision": "tool_call", "tool_calls": []}
        for call in tool_calls:
            result_text = dispatch_tool(call.function.name, call.function.arguments, logs)
            round_trace["tool_calls"].append(
                {
                    "name": call.function.name,
                    "arguments": call.function.arguments,
                    "result_excerpt": result_text[:TRACE_RESULT_EXCERPT],
                }
            )
            messages.append(
                {"role": "tool", "tool_call_id": call.id, "content": result_text}
            )
        trace.rounds.append(round_trace)
    else:
        # ------------------------------------------------------------------ #
        # 轮次用尽：强制收尾。不给工具（模型只能回答），并明确告知原因。
        # ------------------------------------------------------------------ #
        trace.forced_answer = True
        messages.append(
            {
                "role": "user",
                "content": "工具调用轮次已用完。请基于目前已有信息直接输出最终诊断 JSON。",
            }
        )
        resp = get_client().chat.completions.create(
            model=current_model(),
            messages=messages,
            temperature=0.2,
            response_format={"type": "json_object"},
        )
        usage = _sum_usage(usage, _extract_usage(resp.usage))
        msg = resp.choices[0].message

    # ------------------------------------------------------------------ #
    # 格式校验（与 diagnose_multi_ex 同一套规则 + 一次纠错重试）
    # ------------------------------------------------------------------ #
    content = msg.content or ""
    result: dict | None = None
    problems: list[str] = []
    try:
        result = _parse_json(content)
        problems = schema_problems(result)
    except json.JSONDecodeError:
        problems = ["返回的内容不是合法 JSON"]

    if problems:
        retry_messages = [
            *messages,
            {"role": "assistant", "content": content[:2000]},
            {"role": "user", "content": CORRECTION_PROMPT.format(problem="；".join(problems))},
        ]
        resp = get_client().chat.completions.create(
            model=current_model(),
            messages=retry_messages,
            temperature=0.2,
            response_format={"type": "json_object"},
        )
        usage = _sum_usage(usage, _extract_usage(resp.usage))
        content = resp.choices[0].message.content or ""
        try:
            result = _parse_json(content)
            problems = schema_problems(result)
        except json.JSONDecodeError:
            problems = ["返回的内容不是合法 JSON"]

    if problems or result is None:
        raise DiagnosisFormatError(
            f"Agent 模式最终输出不符合格式（{'；'.join(problems)}）。"
            f"它最后一次说的是：{content.strip()[:200] or '（空）'}"
        )

    trace.rounds.append({"round": len(trace.rounds) + 1, "decision": "answer"})
    return result, usage, trace
