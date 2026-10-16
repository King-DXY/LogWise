"""Agent 模式（app/agent.py + app/tools.py）的测试。

模型侧全部用 FakeAgentClient 脚本化返回，不打真实网络。
脚本条目两种形态：
- ``str``                 → 模型直接输出该内容（最终回答）
- ``{"tool_calls": [...]}`` → 模型返回工具调用请求
"""
from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from app import agent, tools
from app.diagnose import DiagnosisFormatError
from app.parser import ParsedLog

GOOD = {
    "error_type": "命名服务配额耗尽",
    "root_cause": "NS-1042 表示租户配额达到上限",
    "evidence": ["code=NS-1042"],
    "impact": "新命名空间创建失败",
    "solution_steps": ["查配额", "申请扩容"],
    "confidence": 0.8,
}
GOOD_JSON = json.dumps(GOOD, ensure_ascii=False)


def _resp(content=None, tool_calls=None, prompt=100, completion=50, cached=0):
    return SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(content=content, tool_calls=tool_calls),
            )
        ],
        usage=SimpleNamespace(
            prompt_tokens=prompt,
            completion_tokens=completion,
            prompt_tokens_details=SimpleNamespace(cached_tokens=cached),
        ),
    )


def _tool_call(call_id: str, name: str, arguments: dict):
    return SimpleNamespace(
        id=call_id,
        function=SimpleNamespace(name=name, arguments=json.dumps(arguments)),
    )


class FakeAgentClient:
    """按脚本返回；脚本条目是 str（直接回答）或 dict（tool_calls）。"""

    def __init__(self, script: list):
        self.script = script
        self.calls = 0
        self.seen_messages: list[list] = []
        self.seen_tools: list = []
        outer = self

        class _Completions:
            def create(self, **kwargs):
                outer.seen_messages.append(list(kwargs["messages"]))
                outer.seen_tools.append(kwargs.get("tools"))
                idx = min(outer.calls, len(outer.script) - 1)
                outer.calls += 1
                item = outer.script[idx]
                if isinstance(item, dict) and "tool_calls" in item:
                    return _resp(tool_calls=item["tool_calls"])
                return _resp(content=item)

        self.chat = SimpleNamespace(completions=_Completions())


@pytest.fixture
def err_log() -> ParsedLog:
    return ParsedLog(
        timestamp="2026-10-16 10:00:00,000",
        level="ERROR",
        module="ns-admin",
        message="创建命名空间失败 code=NS-1042",
    )


@pytest.fixture
def use_fake(monkeypatch):
    def _install(script: list) -> FakeAgentClient:
        fake = FakeAgentClient(script)
        monkeypatch.setattr(agent, "get_client", lambda: fake)
        return fake

    return _install


# --------------------------------------------------------------------------- #
# 循环行为
# --------------------------------------------------------------------------- #
class TestAgentLoop:
    def test_direct_answer_without_tools(self, use_fake, err_log):
        """常见故障：模型第一轮直接回答，一次调用结束。"""
        fake = use_fake([GOOD_JSON])
        result, usage, trace = agent.run_agent([err_log])

        assert result == GOOD
        assert fake.calls == 1
        assert trace.tool_call_count == 0
        assert trace.as_dict()["forced_answer"] is False
        assert trace.rounds[-1]["decision"] == "answer"
        assert usage.prompt_tokens == 100

    def test_tool_call_then_answer(self, use_fake, err_log, monkeypatch):
        """模型先查工具，工具结果塞回 messages，第二轮给结论。"""
        dispatched = []
        monkeypatch.setattr(
            agent,
            "dispatch_tool",
            lambda name, args, logs: dispatched.append(name)
            or json.dumps({"matched": 1, "cases": [{"id": "ns-01"}]}, ensure_ascii=False),
        )
        fake = use_fake(
            [
                {"tool_calls": [_tool_call("c1", "search_fault_cases", {"query": "NS-1042"})]},
                GOOD_JSON,
            ]
        )

        result, usage, trace = agent.run_agent([err_log])

        assert result == GOOD
        assert dispatched == ["search_fault_cases"]
        assert trace.tool_call_count == 1

        # 第二轮请求的 messages 必须带：assistant 的 tool_calls + tool 结果
        second = fake.seen_messages[1]
        roles = [m["role"] if isinstance(m, dict) else getattr(m, "role", None) for m in second]
        assert "tool" in roles
        tool_msg = next(m for m in second if isinstance(m, dict) and m.get("role") == "tool")
        assert tool_msg["tool_call_id"] == "c1"
        assert "ns-01" in tool_msg["content"]

        # 用量两轮累加
        assert usage.prompt_tokens == 200

    def test_round_limit_forces_answer(self, use_fake, err_log, monkeypatch):
        """模型每轮都要查工具：达到上限后强制无工具收尾。"""
        monkeypatch.setattr(agent, "dispatch_tool", lambda n, a, logs: "{}")
        script = [
            {"tool_calls": [_tool_call(f"c{i}", "search_fault_cases", {"query": "x"})]}
            for i in range(3)
        ] + [GOOD_JSON]
        fake = use_fake(script)

        result, usage, trace = agent.run_agent([err_log], max_rounds=3)

        assert result == GOOD
        assert trace.as_dict()["forced_answer"] is True
        assert trace.tool_call_count == 3
        # 最后一次调用不得携带 tools —— 强制回答
        assert fake.seen_tools[-1] is None
        # 3 轮工具 + 1 次强制 = 4 次调用
        assert fake.calls == 4

    def test_bad_format_triggers_one_retry(self, use_fake, err_log):
        """最终输出不合规时按 CORRECTION_PROMPT 重试一次。"""
        fake = use_fake(["这不是 JSON", GOOD_JSON])

        result, _, _ = agent.run_agent([err_log])

        assert result == GOOD
        assert fake.calls == 2
        last_roles = [m["role"] for m in fake.seen_messages[1] if isinstance(m, dict)]
        assert last_roles[-1] == "user"  # 纠错指令

    def test_persistent_bad_format_raises(self, use_fake, err_log):
        use_fake(["不是 JSON", "还不是 JSON"])
        with pytest.raises(DiagnosisFormatError):
            agent.run_agent([err_log])


# --------------------------------------------------------------------------- #
# 工具层
# --------------------------------------------------------------------------- #
class TestTools:
    def test_unknown_tool_returns_error(self, err_log):
        out = json.loads(tools.dispatch_tool("delete_everything", "{}", [err_log]))
        assert "error" in out
        assert "delete_everything" in out["error"]

    def test_bad_arguments_json_returns_error(self, err_log):
        out = json.loads(tools.dispatch_tool("search_fault_cases", "{oops", [err_log]))
        assert "error" in out

    def test_log_overview(self):
        logs = [
            ParsedLog(timestamp="t1", level="INFO", module="a", message="启动"),
            ParsedLog(timestamp="t2", level="ERROR", module="b", message="挂了"),
            ParsedLog(timestamp="t3", level="WARN", module="a", message="慢了"),
        ]
        out = json.loads(tools.dispatch_tool("get_log_overview", "{}", logs))
        assert out["total"] == 3
        assert out["levels"] == {"INFO": 1, "ERROR": 1, "WARN": 1}
        assert out["errors"][0]["module"] == "b"
        assert out["time_span"] == {"first": "t1", "last": "t3"}

    def test_lookup_error_type_hit(self):
        """用真实知识库：disk_full 必须能查到 disk-01。"""
        out = json.loads(tools.dispatch_tool("lookup_error_type", {"error_type": "disk_full"}))
        assert out["matched"] >= 1
        assert any(c["id"] == "disk-01" for c in out["cases"])

    def test_lookup_error_type_miss_lists_available(self):
        """查不到时给出可用的 fault_type 列表，而不是一句干巴巴的没找到。"""
        out = json.loads(tools.dispatch_tool("lookup_error_type", {"error_type": "nope"}))
        assert out["matched"] == 0
        assert "disk_full" in out["available_types"]

    def test_search_fault_cases_hit(self):
        """真实知识库 + BM25：'No space left on device' 必须命中磁盘案例。"""
        out = json.loads(
            tools.dispatch_tool("search_fault_cases", {"query": "No space left on device"})
        )
        assert out["matched"] >= 1
        assert any("disk" in c["id"] for c in out["cases"])

    def test_search_result_is_lean(self):
        """工具返回值不含日志原文 —— 那是纯浪费 token。"""
        out = json.loads(
            tools.dispatch_tool("search_fault_cases", {"query": "No space left on device"})
        )
        for case in out["cases"]:
            assert "log" not in case
            assert set(case) <= {"id", "fault_type", "summary", "root_cause", "solution_steps"}


# --------------------------------------------------------------------------- #
# HTTP 接口层
# --------------------------------------------------------------------------- #
class TestAgentEndpoint:
    @pytest.fixture
    def client(self, monkeypatch):
        monkeypatch.setenv("LOGWISE_RATE_LIMIT", "0")
        from fastapi.testclient import TestClient

        from app.main import app

        return TestClient(app)

    def test_endpoint_returns_trace(self, client, monkeypatch, err_log):
        """接口必须带回 diagnosis + trace，且鉴权限流与老接口一致。"""
        from app import main

        fake_trace = agent.AgentTrace(rounds=[{"round": 1, "decision": "answer"}])
        monkeypatch.setattr(
            agent, "run_agent", lambda logs: (GOOD, agent.Usage(), fake_trace)
        )
        # run_agent 是在端点函数内延迟导入的，patch 模块属性即可生效

        r = client.post(
            "/diagnose/agent",
            json={"text": "2026-10-16 10:00:00,000 ERROR [ns-admin] a - 创建失败 code=NS-1042"},
        )
        assert r.status_code == 200
        data = r.json()
        assert data["diagnosis"] == GOOD
        assert data["trace"]["rounds"][0]["decision"] == "answer"
        assert data["trace"]["forced_answer"] is False
        assert data["cached"] is False

    def test_endpoint_model_error_is_502(self, client, monkeypatch):
        """模型侧异常对齐老接口：502，不裸奔。"""
        monkeypatch.setattr(
            agent, "run_agent", lambda logs: (_ for _ in ()).throw(RuntimeError("boom"))
        )
        r = client.post(
            "/diagnose/agent",
            json={"text": "2026-10-16 10:00:00,000 ERROR [ns-admin] a - 创建失败"},
        )
        assert r.status_code == 502

    def test_endpoint_records_agent_metrics(self, client, monkeypatch):
        fake_trace = agent.AgentTrace(
            rounds=[{"round": 1, "decision": "answer"}]
        )
        monkeypatch.setattr(
            agent, "run_agent", lambda logs: (GOOD, agent.Usage(), fake_trace)
        )
        client.post(
            "/diagnose/agent",
            json={"text": "2026-10-16 10:00:00,000 ERROR [ns-admin] a - 创建失败 code=NS-1042"},
        )
        stats = client.get("/stats").json()
        assert stats["metrics"]["agent"]["requests"] >= 1
        assert stats["metrics"]["agent"]["direct_answer_rate"] > 0


# --------------------------------------------------------------------------- #
# /stats 的 Agent 指标
# --------------------------------------------------------------------------- #
class TestAgentMetrics:
    def test_agent_summary_counts(self):
        from app import metrics as metrics_mod

        s = metrics_mod.MetricsStore()
        # 两条直接回答 + 一条带 2 次工具调用
        s.record(
            metrics_mod.CallRecord(
                endpoint="/diagnose/agent", duration_ms=10, agent=True, agent_rounds=1
            )
        )
        s.record(
            metrics_mod.CallRecord(
                endpoint="/diagnose/agent", duration_ms=10, agent=True, agent_rounds=1
            )
        )
        s.record(
            metrics_mod.CallRecord(
                endpoint="/diagnose/agent",
                duration_ms=10,
                agent=True,
                agent_rounds=3,
                agent_tool_calls=2,
                agent_forced=True,
            )
        )
        # 非 Agent 请求不进 Agent 统计
        s.record(metrics_mod.CallRecord(endpoint="/diagnose", duration_ms=10))

        a = s.summary()["agent"]
        assert a["requests"] == 3
        assert a["direct_answer_rate"] == round(2 / 3, 4)
        assert a["tool_call_rate"] == round(1 / 3, 4)
        assert a["avg_tool_calls"] == round(2 / 3, 2)
        assert a["avg_rounds"] == round(5 / 3, 2)
        assert a["forced_answer_count"] == 1

    def test_empty_agent_summary(self):
        from app import metrics as metrics_mod

        s = metrics_mod.MetricsStore()
        assert s.summary()["agent"] == {"requests": 0}
