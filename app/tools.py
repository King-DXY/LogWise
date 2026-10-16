"""Agent 可调用的工具层（Function Calling 的本地执行端）。

设计要点
--------
1. **工具只做「模型做不到的事」**：检索私有案例库、统计本次日志。
   通用故障模式（磁盘满、NPE）模型本来就会，不给工具 —— 给了也是白花钱，
   这是 RAG 实验已经验证过的结论。
2. **返回值刻意瘦身**：只回 id / fault_type / summary / root_cause / solution_steps，
   不把整条日志原文塞回去 —— 日志已经在 messages 里了，再塞一遍是纯浪费 token。
3. **执行永不抛异常到模型侧**：工具内部出错（如知识库缺失）时返回
   ``{"error": ...}`` 给模型，让它带着这个信息继续推理，
   而不是让整条诊断链路 500。
"""
from __future__ import annotations

import json
from collections import Counter

from app.parser import ParsedLog
from app.retriever import get_retriever

# --------------------------------------------------------------------------- #
# 工具的 JSON Schema（发给模型，告诉它有哪些工具、怎么调）
# --------------------------------------------------------------------------- #
TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "search_fault_cases",
            "description": (
                "在历史故障案例库中检索与指定关键词相关的案例。"
                "适用于：遇到不熟悉的错误码、专有组件名、或想参考类似故障的处置方法时。"
                "常见故障（磁盘满、空指针、连接被拒等）不需要调用，直接诊断即可。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "检索关键词，通常是错误码、异常类名或组件名",
                    },
                    "k": {
                        "type": "integer",
                        "description": "返回案例数，默认 3，最多 5",
                        "default": 3,
                    },
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_log_overview",
            "description": (
                "获取本次日志的整体统计：各级别条数、涉及的模块、时间跨度、ERROR 列表。"
                "适用于：日志量较大、需要先了解全貌再定位时。"
            ),
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "lookup_error_type",
            "description": (
                "按错误类型在历史故障案例库里精确查找处置规范。"
                "适用于：已判断出错误类型（如 disk_full、oom），想查该类故障的标准处置流程时。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "error_type": {
                        "type": "string",
                        "description": "错误类型标识（fault_type），如 disk_full、oom、connection_refused",
                    }
                },
                "required": ["error_type"],
            },
        },
    },
]

TOOL_NAMES = {t["function"]["name"] for t in TOOL_SCHEMAS}

# 单个工具返回值的最大字符数：防止检索出超长内容把上下文撑爆
MAX_TOOL_RESULT_CHARS = 3000


# --------------------------------------------------------------------------- #
# 工具实现
# --------------------------------------------------------------------------- #
def search_fault_cases(query: str, k: int = 3) -> dict:
    """BM25 检索案例库，返回瘦身后的案例列表。"""
    k = max(1, min(5, int(k or 3)))
    hits = get_retriever().search(query, k=k)
    return {
        "matched": len(hits),
        "cases": [
            {
                "id": h.id,
                "fault_type": h.case.get("fault_type", ""),
                "summary": h.case.get("summary", ""),
                "root_cause": h.case["diagnosis"].get("root_cause", ""),
                "solution_steps": h.case["diagnosis"].get("solution_steps", []),
            }
            for h in hits
        ],
    }


def get_log_overview(logs: list[ParsedLog]) -> dict:
    """统计本次日志的全貌。日志由 Agent 循环传入，不需要模型提供。"""
    if not logs:
        return {"error": "本次没有解析到日志"}
    levels = Counter(log.level for log in logs)
    modules = sorted({log.module for log in logs if log.module})
    errors = [
        {"index": i, "module": log.module, "message": log.message[:120]}
        for i, log in enumerate(logs)
        if log.level in ("ERROR", "CRITICAL", "FATAL")
    ]
    return {
        "total": len(logs),
        "levels": dict(levels),
        "modules": modules[:20],
        "time_span": {"first": logs[0].timestamp, "last": logs[-1].timestamp},
        "errors": errors[:10],
    }


def lookup_error_type(error_type: str) -> dict:
    """按 fault_type 精确匹配案例库。"""
    wanted = (error_type or "").strip().lower()
    if not wanted:
        return {"error": "error_type 不能为空"}
    matches = [
        {
            "id": c["id"],
            "fault_type": c.get("fault_type", ""),
            "summary": c.get("summary", ""),
            "root_cause": c["diagnosis"].get("root_cause", ""),
            "solution_steps": c["diagnosis"].get("solution_steps", []),
        }
        for c in get_retriever().cases
        if c.get("fault_type", "").lower() == wanted
    ]
    if not matches:
        # 给出可用的 fault_type 列表，让模型下次能查对 —— 比一句"没找到"有用得多
        available = sorted({c.get("fault_type", "") for c in get_retriever().cases})
        return {"matched": 0, "available_types": available}
    return {"matched": len(matches), "cases": matches}


# --------------------------------------------------------------------------- #
# 统一分发
# --------------------------------------------------------------------------- #
def dispatch_tool(name: str, arguments: str | dict, logs: list[ParsedLog] | None = None) -> str:
    """执行一次工具调用，返回给模型看的 JSON 字符串。

    参数
    ----
    name
        模型给出的工具名。不在 TOOL_NAMES 里时返回错误信息而不是抛异常 ——
        让模型知道"这个工具不存在"，它通常会换个工具或直接回答。
    arguments
        模型给出的参数（JSON 字符串或已解析的 dict）。解析失败同样回错误。
    logs
        本次诊断的日志上下文，只有 get_log_overview 会用到。
    """
    if name not in TOOL_NAMES:
        return json.dumps(
            {"error": f"未知工具 {name!r}，可用工具：{sorted(TOOL_NAMES)}"},
            ensure_ascii=False,
        )
    if isinstance(arguments, str):
        try:
            args = json.loads(arguments) if arguments.strip() else {}
        except json.JSONDecodeError as exc:
            return json.dumps({"error": f"工具参数不是合法 JSON：{exc}"}, ensure_ascii=False)
    else:
        args = dict(arguments or {})

    try:
        if name == "search_fault_cases":
            result = search_fault_cases(args.get("query", ""), args.get("k", 3))
        elif name == "get_log_overview":
            result = get_log_overview(logs or [])
        else:  # lookup_error_type
            result = lookup_error_type(args.get("error_type", ""))
    except Exception as exc:  # 工具出错不该让诊断链路整体失败
        result = {"error": f"{type(exc).__name__}: {exc}"}

    text = json.dumps(result, ensure_ascii=False)
    if len(text) > MAX_TOOL_RESULT_CHARS:
        text = text[:MAX_TOOL_RESULT_CHARS] + "...（结果过长已截断）"
    return text
