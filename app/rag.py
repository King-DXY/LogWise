"""把检索到的历史故障案例注入 Prompt（RAG 的注入端）。

三种模式，都要用数据比出来
--------------------------
| 模式 | 注入内容 | 回答什么问题 |
| --- | --- | --- |
| `fixed`   | 只用 Prompt 里固定的 4 个 Few-shot | 基线（现状） |
| `dynamic` | 只用检索到的 top-k 历史案例 | RAG 单独有多大用 |
| `hybrid`  | 固定 Few-shot + 检索案例 | 两者互补还是互相干扰 |

固定示例提供**输出格式与判断思路的稳定性**，动态案例提供**本次故障的具体参照**。
它们不是简单的替代关系，谁更好必须测，不能拍脑袋。

为什么注入的案例要专门写一段说明
--------------------------------
System Prompt 里已经有一段约束，说对话历史中的示例**只是格式示范、与当前日志无关**。
但检索到的案例性质不同 —— 它们是该系统**真实发生过的历史故障**，是**应该**被参考的。

如果不加区分，模型会沿用前一条约束把历史案例也当成格式示范而忽略掉；
反过来，如果不加限制地让模型参考，它又会直接把历史案例的结论套到当前日志上。

所以这里单独注入一条 system 消息，把两件事同时说清楚：**可以参考思路，不能照搬结论**。
"""
from __future__ import annotations

import os

from app.prompts import FEW_SHOT_EXAMPLES
from app.retriever import RetrievedCase, get_retriever

MODE_FIXED = "fixed"
MODE_DYNAMIC = "dynamic"
MODE_HYBRID = "hybrid"
MODES = (MODE_FIXED, MODE_DYNAMIC, MODE_HYBRID)

# 默认保持 fixed —— 接入新能力默认不改变现有行为，
# 等三组对比的数据出来后再决定要不要换默认值。
DEFAULT_MODE = MODE_FIXED
DEFAULT_TOP_K = 3

RAG_NOTICE = """【本次附带的参考资料】
上面几条对话记录是该系统的**历史故障案例**，由检索从历史故障库中挑出，与本次故障可能相关。

使用要求：
1. 可以参考它们的排查思路、验证手段与处置步骤 —— 这是它们的主要价值。
2. 但结论必须基于**本次日志的实际证据**。现象相似不等于根因相同，
   直接把历史案例的结论套到本次故障上是错误做法。
3. 历史案例里的时间、服务名、数值都是过去的记录，不代表当前状态，
   不要把其中内容当作本次故障的事实。
4. 如果历史案例与本次日志并不匹配，忽略它们即可，不要为了呼应而强行关联。
"""


def rag_enabled() -> bool:
    return os.getenv("LOGWISE_RAG_ENABLED", "true").strip().lower() in ("1", "true", "yes", "on")


def rag_mode() -> str:
    mode = os.getenv("LOGWISE_RAG_MODE", DEFAULT_MODE).strip().lower()
    return mode if mode in MODES else DEFAULT_MODE


def rag_top_k() -> int:
    try:
        return max(1, min(10, int(os.getenv("LOGWISE_RAG_TOP_K", DEFAULT_TOP_K))))
    except (TypeError, ValueError):
        return DEFAULT_TOP_K


def retrieve(query: str, k: int | None = None) -> list[RetrievedCase]:
    """按当前配置检索。fixed 模式下不检索，直接返回空。"""
    if not rag_enabled() or rag_mode() == MODE_FIXED:
        return []
    return get_retriever().search(query, k=k or rag_top_k())


def build_examples(
    query: str,
    mode: str | None = None,
    k: int | None = None,
    fixed_examples: list[dict] | None = None,
) -> tuple[list[dict], list[str]]:
    """组装要注入 Prompt 的示例，返回 (示例列表, 检索到的案例 id 列表)。

    示例统一成 ``{"log": ..., "diagnosis": ...}`` 结构 —— 与 ``FEW_SHOT_EXAMPLES``
    一致，这样上层用同一段代码就能把它们拼进 messages。

    第二个返回值是**检索结果指纹**，调用方要用它参与缓存键计算：
    检索到的案例变了，同样的输入会产生不同的诊断，旧缓存必须失效。
    """
    mode = mode or rag_mode()
    fixed = FEW_SHOT_EXAMPLES if fixed_examples is None else fixed_examples

    examples: list[dict] = []
    retrieved_ids: list[str] = []

    if mode in (MODE_FIXED, MODE_HYBRID):
        examples.extend(fixed)

    if mode in (MODE_DYNAMIC, MODE_HYBRID) and rag_enabled():
        cases = get_retriever().search(query, k=k or rag_top_k())
        retrieved_ids = [c.id for c in cases]
        for case in cases:
            log_text, diagnosis_text = case.as_few_shot()
            examples.append({"log": log_text, "diagnosis": diagnosis_text})

    return examples, retrieved_ids


def signature(mode: str | None = None, retrieved_ids: list[str] | None = None, k: int | None = None) -> str:
    """检索配置与结果的指纹，用于缓存键。

    为什么必须带上：改了模式（fixed → hybrid）或检索到的案例换了之后，
    同样的日志也应该得到不同的诊断。不纳入缓存键的话，
    调参会以为「模式改了没效果」—— 这是缓存类功能最隐蔽的坑。
    """
    mode = mode or rag_mode()
    if mode == MODE_FIXED:
        return MODE_FIXED
    ids = ",".join(sorted(retrieved_ids or []))
    return f"{mode}|k{k or rag_top_k()}|{ids}"


def describe(mode: str | None = None) -> dict:
    """给 /stats 用的配置回显。"""
    from app.retriever import describe_backend  # 延迟导入，避免模块间耦合

    mode = mode or rag_mode()
    return {
        "enabled": rag_enabled(),
        "mode": mode,
        "top_k": rag_top_k(),
        "knowledge_base_size": get_retriever().size(),
        "retriever": describe_backend(),
        "note": "mode=fixed 时不检索；dynamic 只用检索案例；hybrid 为固定示例 + 检索案例。"
                "retriever.requested / actual 不一致表示发生了降级（如向量索引未构建）",
    }
