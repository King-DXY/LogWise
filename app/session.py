"""诊断会话：支撑「多轮追问」。

为什么需要这个模块
------------------
诊断本身是一次性的（无状态）：日志进、诊断出。但用户看完诊断后自然的
下一个动作是**追问** —— "第三步具体怎么操作？"、"如果是生产环境呢？"。
没有会话记忆，模型根本不记得刚才的日志和结论，追问无从谈起。

设计要点
--------
1. **会话内容 = 诊断时的完整 messages + 诊断结论**。
   这样追问时模型能看到与诊断完全一致的上下文（含固定 Few-shot），
   不会出现「诊断一套上下文、追问另一套」的割裂。
2. **追问回答是自由文本**，不要求 JSON —— 追问是给人看的解释，
   结构化输出反而碍事。
3. **内存实现 + TTL + LRU**：与 cache.py / guard.py 同一套取舍，
   单实例够用，多副本应换 Redis。
4. **追问不调工具、不走 RAG**：它的定位是「解释刚才的诊断」，
   引入工具会让成本与复杂度失控。需要重新分析就发新的诊断请求。
"""
from __future__ import annotations

import json
import threading
import time
import uuid
from collections import OrderedDict
from dataclasses import dataclass, field

from app.diagnose import Usage, _extract_usage, current_model, get_client

DEFAULT_TTL = 1800.0        # 会话 30 分钟不说话就过期
DEFAULT_MAX_SESSIONS = 100  # 最多同时存多少个会话
MAX_FOLLOWUP_ROUNDS = 6     # 最多保留几轮追问（防止上下文无限膨胀）
MAX_QUESTION_CHARS = 2000   # 单个问题长度上限（防止把整篇文档贴进来烧钱）


@dataclass
class Session:
    """一次诊断的会话上下文。"""

    messages: list[dict]           # 诊断 messages + 诊断结论 + 追问记录
    created: float
    touched: float
    followups: int = field(default=0)


class SessionStore:
    """带 TTL 与 LRU 淘汰的线程安全会话存储（与 DiagnosisCache 同款结构）。"""

    def __init__(self, ttl: float = DEFAULT_TTL, max_sessions: int = DEFAULT_MAX_SESSIONS) -> None:
        self._lock = threading.Lock()
        self._data: OrderedDict[str, Session] = OrderedDict()
        self._ttl = ttl
        self._max = max_sessions

    def create(self, messages: list[dict], diagnosis: dict) -> str:
        """以「诊断时的 messages + 诊断结论」为底子建会话，返回 session_id。"""
        base = [*messages, {"role": "assistant", "content": json.dumps(diagnosis, ensure_ascii=False)}]
        now = time.time()
        sid = uuid.uuid4().hex[:12]
        with self._lock:
            self._data[sid] = Session(messages=base, created=now, touched=now)
            self._data.move_to_end(sid)
            while len(self._data) > self._max:
                self._data.popitem(last=False)
        return sid

    def _get_fresh(self, sid: str) -> Session | None:
        """取会话并顺手维护 TTL/LRU。调用方需持有锁。"""
        s = self._data.get(sid)
        if s is None:
            return None
        if time.time() - s.touched > self._ttl:
            self._data.pop(sid, None)
            return None
        self._data.move_to_end(sid)
        return s

    def ask(self, sid: str, question: str) -> tuple[str, Usage]:
        """在指定会话里追问一个问题，返回 (回答文本, token 用量)。

        会话不存在或已过期时抛 ``KeyError``（接口层转成 404）。
        """
        question = (question or "").strip()
        if not question:
            raise ValueError("问题不能为空")
        if len(question) > MAX_QUESTION_CHARS:
            raise ValueError(f"问题过长（{len(question)} 字，最多 {MAX_QUESTION_CHARS} 字）")

        with self._lock:
            session = self._get_fresh(sid)
            if session is None:
                raise KeyError(sid)
            session.messages.append({"role": "user", "content": question})
            messages = list(session.messages)  # 拷贝一份再调模型，别在锁里等网络

        try:
            resp = get_client().chat.completions.create(
                model=current_model(),
                messages=messages,
                temperature=0.3,  # 追问允许稍活的表达，但仍以稳定为主
            )
        except Exception:
            # 调用失败要把问题撤回：否则下一次追问前面会挂着一条没有回答的
            # user 消息，模型会把它当成「用户又问了但没被回应」的悬空上下文。
            with self._lock:
                s = self._get_fresh(sid)
                if (
                    s is not None
                    and s.messages
                    and s.messages[-1].get("role") == "user"
                    and s.messages[-1].get("content") == question
                ):
                    s.messages.pop()
            raise
        answer = resp.choices[0].message.content or ""
        usage = _extract_usage(resp.usage)

        with self._lock:
            session = self._get_fresh(sid)
            if session is not None:
                session.messages.append({"role": "assistant", "content": answer})
                session.followups += 1
                session.touched = time.time()
                # 只保留最近 N 轮追问：头部（system + few-shot + 日志 + 诊断）不动，
                # 从尾部往前裁 —— 越早的追问价值越低，且上下文不能无限膨胀。
                head_len = len(session.messages) - session.followups * 2
                overflow = session.followups - MAX_FOLLOWUP_ROUNDS
                if overflow > 0:
                    drop = overflow * 2
                    session.messages = session.messages[:head_len] + session.messages[head_len + drop :]
                    session.followups = MAX_FOLLOWUP_ROUNDS
        return answer, usage

    def snapshot(self) -> dict:
        with self._lock:
            return {"active_sessions": len(self._data), "ttl_seconds": self._ttl}


# 全局单例
store = SessionStore()
