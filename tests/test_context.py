"""上下文聚合逻辑的单元测试。

这些测试守护的是本项目最重要的一个修复：原来只把「第一条 ERROR」送给模型，
多日志故障场景下命中率只有 16.7%（详见 README）。聚合逻辑一旦被改坏，
评测分数会掉，但代码不会报错 —— 所以必须固化下来。
"""
from __future__ import annotations

import pytest

from app import context as ctx_mod
from app.context import build_context, select_anchor_index
from app.parser import ParsedLog


def make_log(level: str, message: str, ts: str = "2026-09-18 10:00:00,000") -> ParsedLog:
    return ParsedLog(timestamp=ts, level=level, module="test", message=message, stack_trace="")


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    """默认开启聚合、参数用默认值，避免受本机 .env 影响。"""
    monkeypatch.setenv("LOGWISE_CONTEXT_ENABLED", "true")
    monkeypatch.delenv("LOGWISE_CONTEXT_WINDOW", raising=False)
    monkeypatch.delenv("LOGWISE_CONTEXT_MAX_LOGS", raising=False)


# --------------------------------------------------------------------------- #
# 锚点选择
# --------------------------------------------------------------------------- #
class TestAnchorSelection:
    def test_picks_first_error(self):
        logs = [make_log("INFO", "a"), make_log("ERROR", "b"), make_log("ERROR", "c")]
        assert select_anchor_index(logs) == 1

    def test_error_levels_all_qualify(self):
        for level in ("ERROR", "CRITICAL", "FATAL"):
            logs = [make_log("INFO", "a"), make_log(level, "b")]
            assert select_anchor_index(logs) == 1

    def test_falls_back_to_first_warn_when_no_error(self):
        logs = [make_log("INFO", "a"), make_log("WARN", "b"), make_log("WARN", "c")]
        assert select_anchor_index(logs) == 1

    def test_falls_back_to_first_log(self):
        logs = [make_log("INFO", "a"), make_log("DEBUG", "b")]
        assert select_anchor_index(logs) == 0


# --------------------------------------------------------------------------- #
# 关闭时退回旧行为
# --------------------------------------------------------------------------- #
class TestDisabled:
    def test_disabled_returns_only_anchor(self, monkeypatch):
        monkeypatch.setenv("LOGWISE_CONTEXT_ENABLED", "false")
        logs = [make_log("WARN", "a"), make_log("ERROR", "b"), make_log("ERROR", "c")]
        result = build_context(logs)
        assert result.size == 1
        assert result.logs[0].message == "b"
        assert result.enabled is False

    def test_disabled_signature_differs(self, monkeypatch):
        logs = [make_log("ERROR", "b")]
        monkeypatch.setenv("LOGWISE_CONTEXT_ENABLED", "false")
        off = build_context(logs).cache_signature()
        monkeypatch.setenv("LOGWISE_CONTEXT_ENABLED", "true")
        on = build_context(logs).cache_signature()
        assert off != on


# --------------------------------------------------------------------------- #
# 聚合行为
# --------------------------------------------------------------------------- #
class TestAggregation:
    def test_includes_logs_around_anchor(self):
        logs = [
            make_log("INFO", "1"),
            make_log("INFO", "2"),
            make_log("ERROR", "3"),   # 锚点
            make_log("INFO", "4"),
            make_log("INFO", "5"),
        ]
        result = build_context(logs)
        assert [l.message for l in result.logs] == ["1", "2", "3", "4", "5"]

    def test_window_limits_reach(self, monkeypatch):
        monkeypatch.setenv("LOGWISE_CONTEXT_WINDOW", "1")
        logs = [make_log("INFO", str(i)) for i in range(7)]
        logs[3] = make_log("ERROR", "3")   # 锚点在第 3 位
        result = build_context(logs)
        assert sorted(l.message for l in result.logs) == ["2", "3", "4"]

    def test_window_at_list_start_does_not_overflow(self, monkeypatch):
        monkeypatch.setenv("LOGWISE_CONTEXT_WINDOW", "3")
        logs = [make_log("ERROR", "0"), make_log("INFO", "1")]
        result = build_context(logs)
        assert [l.message for l in result.logs] == ["0", "1"]

    def test_window_at_list_end_does_not_overflow(self, monkeypatch):
        monkeypatch.setenv("LOGWISE_CONTEXT_WINDOW", "3")
        logs = [make_log("INFO", "1"), make_log("ERROR", "2")]
        result = build_context(logs)
        assert [l.message for l in result.logs] == ["1", "2"]

    def test_anomaly_outside_window_is_included(self, monkeypatch):
        """远处的 WARN 可能就是真因，必须纳入 —— 这是修复的核心。"""
        monkeypatch.setenv("LOGWISE_CONTEXT_WINDOW", "1")
        monkeypatch.setenv("LOGWISE_CONTEXT_MAX_LOGS", "12")
        logs = [
            make_log("WARN", "远处真因"),
            make_log("INFO", "噪音1"),
            make_log("INFO", "噪音2"),
            make_log("INFO", "噪音3"),
            make_log("ERROR", "锚点"),
        ]
        result = build_context(logs)
        messages = [l.message for l in result.logs]
        assert "远处真因" in messages
        assert "锚点" in messages
        # 窗口外的纯 INFO 噪音不应被纳入
        assert "噪音1" not in messages

    def test_non_anomaly_outside_window_excluded(self, monkeypatch):
        monkeypatch.setenv("LOGWISE_CONTEXT_WINDOW", "0")
        logs = [make_log("INFO", "远"), make_log("ERROR", "锚点")]
        result = build_context(logs)
        assert [l.message for l in result.logs] == ["锚点"]

    def test_output_keeps_original_order(self):
        """时间线顺序本身就是诊断线索，不能按级别重排。"""
        logs = [
            make_log("INFO", "1"),
            make_log("ERROR", "2"),
            make_log("INFO", "3"),
            make_log("WARN", "4"),
        ]
        result = build_context(logs)
        assert [l.message for l in result.logs] == ["1", "2", "3", "4"]


# --------------------------------------------------------------------------- #
# 上限裁剪
# --------------------------------------------------------------------------- #
class TestMaxLogs:
    def test_anchor_always_kept_even_when_capped(self, monkeypatch):
        monkeypatch.setenv("LOGWISE_CONTEXT_WINDOW", "4")
        monkeypatch.setenv("LOGWISE_CONTEXT_MAX_LOGS", "2")
        logs = [make_log("ERROR", str(i)) for i in range(9)]
        result = build_context(logs)
        assert result.truncated is True
        assert result.size == 2
        assert result.anchor.message in [l.message for l in result.logs]

    def test_respects_max_logs(self, monkeypatch):
        monkeypatch.setenv("LOGWISE_CONTEXT_WINDOW", "10")
        monkeypatch.setenv("LOGWISE_CONTEXT_MAX_LOGS", "5")
        logs = [make_log("WARN", str(i)) for i in range(30)]
        result = build_context(logs)
        assert result.size <= 5
        assert result.truncated is True

    def test_not_truncated_when_within_cap(self):
        logs = [make_log("ERROR", "a"), make_log("WARN", "b")]
        result = build_context(logs)
        assert result.truncated is False

    def test_window_logs_win_over_distant_anomalies_when_capped(self, monkeypatch):
        """上限收紧时，应优先保留锚点附近的日志，而不是远处的异常。"""
        monkeypatch.setenv("LOGWISE_CONTEXT_WINDOW", "2")
        monkeypatch.setenv("LOGWISE_CONTEXT_MAX_LOGS", "3")
        logs = [
            make_log("WARN", "很远的异常"),
            make_log("INFO", "x1"),
            make_log("INFO", "x2"),
            make_log("INFO", "靠近1"),
            make_log("ERROR", "锚点"),
            make_log("INFO", "靠近2"),
        ]
        result = build_context(logs)
        messages = [l.message for l in result.logs]
        assert "锚点" in messages
        assert "靠近1" in messages or "靠近2" in messages
        assert "很远的异常" not in messages

    def test_max_logs_at_least_one(self, monkeypatch):
        monkeypatch.setenv("LOGWISE_CONTEXT_MAX_LOGS", "0")
        logs = [make_log("ERROR", "a"), make_log("WARN", "b")]
        result = build_context(logs)
        assert result.size >= 1


# --------------------------------------------------------------------------- #
# 边界情况
# --------------------------------------------------------------------------- #
class TestEdgeCases:
    def test_empty_input(self):
        result = build_context([])
        assert result.size == 0
        assert result.anchor is None

    def test_single_log(self):
        result = build_context([make_log("ERROR", "only")])
        assert result.size == 1
        assert result.anchor.message == "only"
        assert result.truncated is False

    def test_total_parsed_recorded(self):
        logs = [make_log("INFO", str(i)) for i in range(5)] + [make_log("ERROR", "boom")]
        result = build_context(logs)
        assert result.total_parsed == 6


# --------------------------------------------------------------------------- #
# 缓存签名
# --------------------------------------------------------------------------- #
class TestCacheSignature:
    def test_signature_changes_with_window(self, monkeypatch):
        logs = [make_log("ERROR", "a")]
        monkeypatch.setenv("LOGWISE_CONTEXT_WINDOW", "4")
        s1 = build_context(logs).cache_signature()
        monkeypatch.setenv("LOGWISE_CONTEXT_WINDOW", "8")
        s2 = build_context(logs).cache_signature()
        assert s1 != s2

    def test_signature_changes_with_max_logs(self, monkeypatch):
        logs = [make_log("ERROR", "a")]
        monkeypatch.setenv("LOGWISE_CONTEXT_MAX_LOGS", "12")
        s1 = build_context(logs).cache_signature()
        monkeypatch.setenv("LOGWISE_CONTEXT_MAX_LOGS", "20")
        s2 = build_context(logs).cache_signature()
        assert s1 != s2

    def test_signature_stable_for_same_params(self):
        logs = [make_log("ERROR", "a")]
        assert build_context(logs).cache_signature() == build_context(logs).cache_signature()


# --------------------------------------------------------------------------- #
# 与缓存键联动
# --------------------------------------------------------------------------- #
class TestCacheKeyIntegration:
    def test_context_signature_affects_cache_key(self, monkeypatch):
        """改了聚合参数必须让缓存失效，否则调参会以为「没效果」。"""
        from app.cache import make_key

        monkeypatch.setenv("LOGWISE_CONTEXT_WINDOW", "4")
        sig_a = build_context([make_log("ERROR", "a")]).cache_signature()
        monkeypatch.setenv("LOGWISE_CONTEXT_WINDOW", "8")
        sig_b = build_context([make_log("ERROR", "a")]).cache_signature()

        key_a = make_key("日志内容", "model", "v3", sig_a)
        key_b = make_key("日志内容", "model", "v3", sig_b)
        assert key_a != key_b

    def test_summary_describe_shape(self):
        result = build_context([make_log("ERROR", "a"), make_log("WARN", "b")])
        info = ctx_mod.describe(result)
        assert info["enabled"] is True
        assert info["selected"] == result.size
        assert set(info) >= {"enabled", "total_parsed", "selected", "truncated", "window", "max_logs"}
