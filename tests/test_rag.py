"""RAG 注入层测试：三种模式的组装、以及缓存键必须随检索结果变化。

最要紧的一条是缓存键
--------------------
改了 mode（fixed → hybrid）或知识库换了一批案例之后，
同样的日志**应该**得到不同的诊断。如果缓存键没把检索结果算进去，
调参会看到「模式改了但输出一模一样」—— 然后花很久才想到是命中了旧缓存。

这个坑在纯 Prompt 时代就踩过一次（PROMPT_VERSION 没进缓存键），
RAG 让它的触发面变大了：Prompt 没变、日志没变，只有检索结果变了。
"""
from __future__ import annotations

import pytest

from app import rag
from app.cache import make_key
from app.context import build_context
from app.diagnose import build_messages
from app.parser import parse_text

LOG = """2026-08-03 11:22:41,308 ERROR [metrics-writer] com.example.metrics.MetricsWriter - 日志文件滚动失败
java.io.IOException: Disk quota exceeded
	at java.base/java.io.UnixFileSystem.createFileExclusively(Native Method)
	at com.example.metrics.MetricsWriter.rotate(MetricsWriter.java:88)"""


@pytest.fixture
def logs():
    return build_context(parse_text(LOG)).logs


class TestModeConfig:
    def test_default_mode_is_fixed(self, monkeypatch):
        """默认必须是 fixed —— 接入新能力不该悄悄改变现有行为。"""
        monkeypatch.delenv("LOGWISE_RAG_MODE", raising=False)
        assert rag.rag_mode() == rag.MODE_FIXED

    def test_unknown_mode_falls_back(self, monkeypatch):
        monkeypatch.setenv("LOGWISE_RAG_MODE", "banana")
        assert rag.rag_mode() == rag.MODE_FIXED

    def test_top_k_is_clamped(self, monkeypatch):
        monkeypatch.setenv("LOGWISE_RAG_TOP_K", "999")
        assert rag.rag_top_k() == 10
        monkeypatch.setenv("LOGWISE_RAG_TOP_K", "0")
        assert rag.rag_top_k() == 1
        monkeypatch.setenv("LOGWISE_RAG_TOP_K", "abc")
        assert rag.rag_top_k() == rag.DEFAULT_TOP_K

    def test_disabled_rag_returns_nothing(self, monkeypatch):
        monkeypatch.setenv("LOGWISE_RAG_MODE", "dynamic")
        monkeypatch.setenv("LOGWISE_RAG_ENABLED", "false")
        assert rag.retrieve("java.io.IOException: Disk quota exceeded") == []


class TestExampleAssembly:
    def test_fixed_mode_uses_only_fixed_examples(self, logs):
        examples, retrieved = rag.build_examples(rag_text(logs), mode=rag.MODE_FIXED)
        assert retrieved == []
        assert len(examples) == len(rag.FEW_SHOT_EXAMPLES)

    def test_dynamic_mode_uses_only_retrieved(self, logs):
        examples, retrieved = rag.build_examples(rag_text(logs), mode=rag.MODE_DYNAMIC, k=3)
        assert retrieved, "应当检索到案例"
        assert len(retrieved) <= 3
        assert len(examples) == len(retrieved)
        assert not any(ex in rag.FEW_SHOT_EXAMPLES for ex in examples)

    def test_hybrid_mode_combines_both(self, logs):
        examples, retrieved = rag.build_examples(rag_text(logs), mode=rag.MODE_HYBRID, k=2)
        assert len(examples) == len(rag.FEW_SHOT_EXAMPLES) + len(retrieved)

    def test_examples_share_the_same_shape(self, logs):
        """固定示例与检索案例的结构必须一致，上层才能用同一段代码拼 messages。"""
        examples, _ = rag.build_examples(rag_text(logs), mode=rag.MODE_HYBRID, k=2)
        for ex in examples:
            assert set(ex) == {"log", "diagnosis"}


class TestMessageStructure:
    def test_fixed_mode_matches_old_behaviour(self, logs):
        """fixed 模式的 messages 应与加 RAG 之前完全一致（1 条 system，无参考资料说明）。"""
        messages = build_messages(logs, mode=rag.MODE_FIXED)
        assert sum(1 for m in messages if m["role"] == "system") == 1
        assert messages[0]["role"] == "system"
        assert messages[-1]["role"] == "user"

    def test_dynamic_mode_adds_notice(self, logs):
        messages = build_messages(logs, mode=rag.MODE_DYNAMIC, top_k=2)
        systems = [m["content"] for m in messages if m["role"] == "system"]
        assert len(systems) == 2, "应额外注入一条参考资料说明"
        assert rag.RAG_NOTICE in systems

    def test_notice_only_when_cases_injected(self, logs):
        """没有注入检索案例时不该出现参考资料说明 —— 否则会和「示例仅示意格式」的约束打架。"""
        messages = build_messages(logs, mode=rag.MODE_FIXED)
        assert rag.RAG_NOTICE not in [m["content"] for m in messages]

    def test_last_message_is_the_log_under_diagnosis(self, logs):
        for mode in rag.MODES:
            messages = build_messages(logs, mode=mode, top_k=2)
            assert messages[-1]["role"] == "user"
            assert "Disk quota exceeded" in messages[-1]["content"]


class TestSignatureAndCacheKey:
    def test_signature_is_stable_for_same_input(self):
        assert rag.signature("hybrid", ["a", "b"], 3) == rag.signature("hybrid", ["b", "a"], 3)

    def test_signature_changes_with_mode(self):
        assert rag.signature("fixed") != rag.signature("hybrid", [], 3)

    def test_signature_changes_with_retrieved_cases(self):
        assert rag.signature("dynamic", ["disk-01"], 3) != rag.signature("dynamic", ["disk-02"], 3)

    def test_signature_changes_with_top_k(self):
        assert rag.signature("dynamic", ["disk-01"], 3) != rag.signature("dynamic", ["disk-01"], 5)

    def test_cache_key_changes_with_rag_signature(self):
        """回归测试：检索结果必须进缓存键，否则「改了模式没效果」会把人带进沟里。"""
        base = make_key("same log text", model="m", prompt_version="v3", context_signature="ctx")
        other = make_key(
            "same log text",
            model="m",
            prompt_version="v3",
            context_signature="ctx",
            rag_signature="hybrid|k3|disk-01",
        )
        assert base != other, "RAG 签名变了但缓存键没变 —— 会命中旧缓存"

    def test_cache_key_stable_when_rag_unchanged(self):
        kw = dict(model="m", prompt_version="v3", context_signature="ctx", rag_signature="hybrid|k3|disk-01")
        assert make_key("same log text", **kw) == make_key("same log text", **kw)


def rag_text(logs) -> str:
    from app.diagnose import format_logs

    return format_logs(logs)
