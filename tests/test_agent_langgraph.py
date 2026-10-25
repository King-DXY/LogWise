"""LangGraph 对比实现的测试（app/agent_langgraph.py）。

与手写版用同一套 FakeAgentClient 脚本，验证**行为等价**：
直接回答、工具调用、强制收尾、格式纠错四条路径都要跑通。
langgraph 是可选依赖，未安装时跳过。
"""
from __future__ import annotations

import json

import pytest

pytest.importorskip("langgraph", reason="langgraph 是可选依赖")

from app import agent_langgraph
from app.parser import ParsedLog
from tests.test_agent import GOOD, GOOD_JSON, FakeAgentClient, _tool_call

LOG = ParsedLog(
    timestamp="2026-10-17 10:00:00,000",
    level="ERROR",
    module="ns-admin",
    message="创建命名空间失败 code=NS-1042",
)


def _run(monkeypatch, script):
    fake = FakeAgentClient(script)
    monkeypatch.setattr(agent_langgraph, "get_client", lambda: fake)
    return agent_langgraph.run_agent_langgraph([LOG]), fake


class TestLangGraphParity:
    def test_direct_answer(self, monkeypatch):
        (result, usage, trace), fake = _run(monkeypatch, [GOOD_JSON])
        assert result["error_type"] == GOOD["error_type"]
        assert fake.calls == 1, "直接回答只该调一次模型"
        assert trace.tool_call_count == 0
        assert trace.forced_answer is False

    def test_tool_call_then_answer(self, monkeypatch):
        script = [
            {"tool_calls": [_tool_call("c1", "lookup_error_type", {"error_type": "quota"})]},
            GOOD_JSON,
        ]
        (result, usage, trace), fake = _run(monkeypatch, script)
        assert result["error_type"] == GOOD["error_type"]
        assert fake.calls == 2
        assert trace.tool_call_count == 1
        assert trace.rounds[0]["tool_calls"][0]["name"] == "lookup_error_type", "trace 要有逐轮明细"
        # 工具结果必须真的回到了模型上下文里（messages 里要有 role=tool）
        assert any(
            isinstance(m, dict) and m.get("role") == "tool"
            for m in fake.seen_messages[-1]
        )

    def test_forced_answer_when_rounds_exhausted(self, monkeypatch):
        script = [{"tool_calls": [_tool_call("c1", "lookup_error_type", {"error_type": "x"})]}] * 3
        (result, usage, trace), fake = _run(monkeypatch, script + [GOOD_JSON])
        assert trace.forced_answer is True
        assert fake.calls == 4, "3 轮工具 + 1 次强制收尾"
        assert trace.tool_call_count == 3, "第 3 轮的工具也要执行 —— 与手写版一致"

    def test_format_retry_recovers(self, monkeypatch):
        (result, _, _), fake = _run(monkeypatch, ["不是 JSON", GOOD_JSON])
        assert result["error_type"] == GOOD["error_type"]
        assert fake.calls == 2
