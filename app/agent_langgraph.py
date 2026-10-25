"""Agent 循环的 LangGraph 版本 —— 与 app/agent.py 的手写循环做对照。

为什么要有这个文件
------------------
``app/agent.py`` 的 ReAct 循环是**手写**的：for 循环 + 轮次上限 + 手工拼
tool 消息。它跑得很好，也是默认实现。手写的价值在于「每一行都清楚为什么这样写」，
代价是新增分支（比如「校验失败 → 换个工具重试」）时要自己维护状态机。

LangGraph 把同一套逻辑表达成**图**：节点是「调模型」「执行工具」「强制收尾」，
边是「下一步去哪」的判断。对照实现的意义：

| 维度 | 手写循环（agent.py） | LangGraph（本文件） |
| --- | --- | --- |
| 控制流 | for + else | 条件边 |
| 状态 | 局部变量 | 显式 State |
| 加分支 | 改循环体 | 加节点/边 |
| 可观测 | 自己记 trace | 图执行轨迹 + checkpointer |

**刻意保持一致的部分**：同一套 Prompt、同一套工具 Schema、同一个 OpenAI 客户端、
同一套格式校验与纠错重试。这样两者的差异只来自「编排层」，对比才公平。

用法::

    from app.agent_langgraph import run_agent_langgraph
    result, usage, trace = run_agent_langgraph(logs)

注意：``langgraph`` 是**可选依赖**（见 requirements.txt 的 agent 附录）。
没有安装时本模块会给出明确报错，主流程（手写循环）不受影响。
"""
from __future__ import annotations

from typing import TypedDict

from app.agent import (
    DEFAULT_MAX_TOOL_ROUNDS,
    TRACE_RESULT_EXCERPT,
    AgentTrace,
    build_agent_messages,
)
from app.diagnose import (
    CORRECTION_PROMPT,
    Usage,
    _extract_usage,
    _parse_json,
    _sum_usage,
    current_model,
    get_client,
    schema_problems,
)
from app.parser import ParsedLog
from app.tools import TOOL_SCHEMAS, dispatch_tool


class AgentState(TypedDict):
    """图的显式状态。

    与手写版的局部变量一一对应：messages / rounds / tool_calls_total / usage。
    区别是它**可以被检查点持久化**（本实现没挂 checkpointer，因为一次诊断
    生命周期很短，挂上只是徒增复杂度）。
    """

    messages: list
    rounds: int
    tool_calls_total: int
    usage: Usage
    forced: bool
    round_details: list  # 每轮工具调用的明细（与手写版 trace.rounds 对齐）


def _build_graph(max_rounds: int, logs: list[ParsedLog]):
    """编译一张「模型 ⇄ 工具」的图。每次调用重建，成本可忽略（纯内存结构）。"""
    try:
        from langgraph.graph import END, StateGraph
    except ImportError as exc:  # 可选依赖缺失时给出可执行提示
        raise RuntimeError(
            "未安装 langgraph，无法使用对比实现。安装："
            "pip install langgraph -i https://pypi.tuna.tsinghua.edu.cn/simple"
        ) from exc

    def agent_node(state: AgentState) -> dict:
        resp = get_client().chat.completions.create(
            model=current_model(),
            messages=state["messages"],
            tools=TOOL_SCHEMAS,
            temperature=0.2,
            response_format={"type": "json_object"},
        )
        return {
            "messages": [*state["messages"], resp.choices[0].message],
            "rounds": state["rounds"] + 1,
            "usage": _sum_usage(state["usage"], _extract_usage(resp.usage)),
        }

    def tools_node(state: AgentState) -> dict:
        last = state["messages"][-1]
        messages = list(state["messages"])
        calls_detail = []
        executed = 0
        for call in getattr(last, "tool_calls", None) or []:
            result_text = dispatch_tool(call.function.name, call.function.arguments, logs)
            messages.append({"role": "tool", "tool_call_id": call.id, "content": result_text})
            calls_detail.append(
                {
                    "name": call.function.name,
                    "arguments": call.function.arguments,
                    "result_excerpt": result_text[:TRACE_RESULT_EXCERPT],
                }
            )
            executed += 1
        return {
            "messages": messages,
            "tool_calls_total": state["tool_calls_total"] + executed,
            "round_details": [
                *state["round_details"],
                {"round": state["rounds"], "decision": "tool_call", "tool_calls": calls_detail},
            ],
        }

    def force_node(state: AgentState) -> dict:
        """轮次用尽：不给工具，逼模型收尾。与手写版 else 分支等价。"""
        messages = [
            *state["messages"],
            {"role": "user", "content": "工具调用轮次已用完。请基于目前已有信息直接输出最终诊断 JSON。"},
        ]
        resp = get_client().chat.completions.create(
            model=current_model(),
            messages=messages,
            temperature=0.2,
            response_format={"type": "json_object"},
        )
        return {
            "messages": [*messages, resp.choices[0].message],
            "usage": _sum_usage(state["usage"], _extract_usage(resp.usage)),
            "forced": True,
        }

    def route(state: AgentState) -> str:
        """agent 节点之后去哪。

        注意边界：``rounds <= max_rounds`` 时仍要执行工具 —— 手写版的 for 循环
        是在每一轮**末尾**判断轮次，所以第 max_rounds 轮的工具也会执行；
        如果这里写成 ``<``，图版会比手写版少执行一轮工具（行为不等价）。
        """
        last = state["messages"][-1]
        if not getattr(last, "tool_calls", None):
            return "done"          # 模型直接回答
        if state["rounds"] <= max_rounds:
            return "tools"
        return "force"

    def after_tools(state: AgentState) -> str:
        """工具执行完：预算用完就去强制收尾，否则回到模型。"""
        return "force" if state["rounds"] >= max_rounds else "agent"

    graph = StateGraph(AgentState)
    graph.add_node("agent", agent_node)
    graph.add_node("tools", tools_node)
    graph.add_node("force", force_node)
    graph.set_entry_point("agent")
    graph.add_conditional_edges("agent", route, {"tools": "tools", "force": "force", "done": END})
    graph.add_conditional_edges("tools", after_tools, {"agent": "agent", "force": "force"})
    graph.add_edge("force", END)
    return graph.compile()


def run_agent_langgraph(
    logs: list[ParsedLog],
    max_rounds: int = DEFAULT_MAX_TOOL_ROUNDS,
) -> tuple[dict, Usage, AgentTrace]:
    """用 LangGraph 跑一次 Agent 诊断，返回与 ``run_agent`` 相同的三元组。

    校验/纠错逻辑与 ``run_agent`` 一致：不合格反馈重试一次，仍不合格抛错。
    """
    from app.diagnose import DiagnosisFormatError

    graph = _build_graph(max_rounds, logs)
    state = graph.invoke(
        {
            "messages": build_agent_messages(logs),
            "rounds": 0,
            "tool_calls_total": 0,
            "usage": Usage(),
            "forced": False,
            "round_details": [],
        }
    )

    content = getattr(state["messages"][-1], "content", "") or ""
    usage: Usage = state["usage"]

    result: dict | None = None
    try:
        result = _parse_json(content)
        problems = schema_problems(result)
    except Exception:
        problems = ["返回的内容不是合法 JSON"]

    if problems:
        resp = get_client().chat.completions.create(
            model=current_model(),
            messages=[
                *state["messages"],
                {"role": "assistant", "content": content[:2000]},
                {"role": "user", "content": CORRECTION_PROMPT.format(problem="；".join(problems))},
            ],
            temperature=0.2,
            response_format={"type": "json_object"},
        )
        usage = _sum_usage(usage, _extract_usage(resp.usage))
        content = resp.choices[0].message.content or ""
        try:
            result = _parse_json(content)
            problems = schema_problems(result)
        except Exception:
            problems = ["返回的内容不是合法 JSON"]

    if problems or result is None:
        raise DiagnosisFormatError(
            f"LangGraph 版最终输出不符合格式（{'；'.join(problems)}）。"
            f"它最后一次说的是：{content.strip()[:200] or '（空）'}"
        )

    trace = AgentTrace(forced_answer=bool(state["forced"]))
    trace.rounds = list(state["round_details"])
    trace.rounds.append({"round": state["rounds"] + 1, "decision": "answer"})
    return result, usage, trace


__all__ = ["run_agent_langgraph"]
