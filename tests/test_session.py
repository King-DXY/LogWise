"""多轮追问（app/session.py + POST /diagnose/followup）的测试。

模型侧全部用 FakeClient 脚本化返回，不打真实网络。
"""
from __future__ import annotations

import json
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from app import session as session_mod
from app.main import app
from app.session import SessionStore

LOG = "2026-10-17 10:00:00,000 ERROR [x] com.example.A - 磁盘写入失败"
DIAGNOSIS = {
    "error_type": "磁盘空间不足",
    "root_cause": "日志分区写满",
    "evidence": ["No space left on device"],
    "impact": "写入失败",
    "solution_steps": ["df -h"],
    "confidence": 0.9,
}


def _fake_client(answer: str):
    class _Completions:
        def create(self, **kwargs):
            return SimpleNamespace(
                choices=[SimpleNamespace(message=SimpleNamespace(content=answer))],
                usage=SimpleNamespace(
                    prompt_tokens=200, completion_tokens=60,
                    prompt_tokens_details=SimpleNamespace(cached_tokens=150),
                ),
            )

    return SimpleNamespace(chat=SimpleNamespace(completions=_Completions()))


@pytest.fixture
def store(monkeypatch) -> SessionStore:
    """每个测试用独立的 SessionStore，并替换全局单例。"""
    s = SessionStore(ttl=1800, max_sessions=10)
    monkeypatch.setattr(session_mod, "store", s)
    return s


class TestSessionStore:
    def test_ask_appends_qa_and_returns_answer(self, store, monkeypatch):
        monkeypatch.setattr(session_mod, "get_client", lambda: _fake_client("执行 df -h 即可"))
        sid = store.create([{"role": "user", "content": "日志"}], DIAGNOSIS)
        answer, usage = store.ask(sid, "怎么查？")
        assert answer == "执行 df -h 即可"
        assert usage.cached_tokens == 150
        assert store._data[sid].followups == 1

    def test_unknown_session_raises_keyerror(self, store):
        with pytest.raises(KeyError):
            store.ask("不存在", "问题")

    def test_expired_session_raises_keyerror(self, store):
        sid = store.create([], DIAGNOSIS)
        store._data[sid].touched -= 9999  # 手动让它过期
        with pytest.raises(KeyError):
            store.ask(sid, "问题")
        assert sid not in store._data, "过期会话应被顺手清掉"

    def test_empty_question_rejected(self, store):
        sid = store.create([], DIAGNOSIS)
        with pytest.raises(ValueError):
            store.ask(sid, "   ")

    def test_overlong_question_rejected(self, store):
        sid = store.create([], DIAGNOSIS)
        with pytest.raises(ValueError, match="问题过长"):
            store.ask(sid, "问" * (session_mod.MAX_QUESTION_CHARS + 1))

    def test_failed_call_rolls_back_the_question(self, store, monkeypatch):
        """模型调用失败时，未回答的问题必须撤回，否则会在上下文里悬空。"""

        def _boom(**kwargs):
            raise RuntimeError("network down")

        class _BadClient:
            class chat:
                class completions:
                    create = staticmethod(_boom)

        monkeypatch.setattr(session_mod, "get_client", lambda: _BadClient())
        sid = store.create([{"role": "user", "content": "日志"}], DIAGNOSIS)
        before = list(store._data[sid].messages)

        with pytest.raises(RuntimeError):
            store.ask(sid, "这一步怎么做？")

        assert store._data[sid].messages == before, "失败的问题不该留在会话里"
        assert store._data[sid].followups == 0

    def test_followup_rounds_are_capped(self, store, monkeypatch):
        """追问轮数有上限：越早的追问先被裁掉，上下文不能无限膨胀。"""
        monkeypatch.setattr(session_mod, "get_client", lambda: _fake_client("答"))
        sid = store.create([{"role": "user", "content": "日志"}], DIAGNOSIS)
        base_len = len(store._data[sid].messages)
        for i in range(session_mod.MAX_FOLLOWUP_ROUNDS + 3):
            store.ask(sid, f"第{i}问")
        s = store._data[sid]
        assert s.followups == session_mod.MAX_FOLLOWUP_ROUNDS
        assert len(s.messages) == base_len + session_mod.MAX_FOLLOWUP_ROUNDS * 2
        # 头部（诊断上下文）不能被裁掉
        assert s.messages[0]["content"] == "日志"

    def test_lru_eviction(self, monkeypatch):
        s = SessionStore(ttl=1800, max_sessions=2)
        monkeypatch.setattr(session_mod, "get_client", lambda: _fake_client("答"))
        ids = [s.create([], DIAGNOSIS) for _ in range(3)]
        assert ids[0] not in s._data, "超容量后最旧的会话应被淘汰"
        assert ids[2] in s._data


@pytest.fixture(scope="module")
def client() -> TestClient:
    return TestClient(app)


@pytest.fixture(autouse=True)
def _no_rate_limit(monkeypatch):
    monkeypatch.setenv("LOGWISE_RATE_LIMIT", "0")


class TestFollowupEndpoint:
    def test_unknown_session_is_404(self, client: TestClient):
        r = client.post("/diagnose/followup", json={"session_id": "x", "question": "?"})
        assert r.status_code == 404

    def test_empty_question_is_422(self, client: TestClient, store):
        sid = store.create([], DIAGNOSIS)
        r = client.post("/diagnose/followup", json={"session_id": sid, "question": " "})
        assert r.status_code == 422

    def test_happy_path(self, client: TestClient, store, monkeypatch):
        monkeypatch.setattr(session_mod, "get_client", lambda: _fake_client("答案文本"))
        sid = store.create([{"role": "user", "content": "日志"}], DIAGNOSIS)
        r = client.post("/diagnose/followup", json={"session_id": sid, "question": "怎么做？"})
        assert r.status_code == 200
        body = r.json()
        assert body["answer"] == "答案文本"
        assert body["usage"]["prompt_tokens"] == 200
