"""防护层与度量层的单元测试。

覆盖：令牌桶限流、缓存键归一化与淘汰、度量聚合与成本计算、鉴权与真实 IP 识别。
"""
from __future__ import annotations

import asyncio
import time

import pytest
from fastapi import HTTPException

from app import cache as cache_mod
from app import guard, metrics


# --------------------------------------------------------------------------- #
# 测试替身
# --------------------------------------------------------------------------- #
class _FakeClient:
    def __init__(self, host: str) -> None:
        self.host = host


class _FakeRequest:
    """最小化的 Request 替身：只提供 guard 用到的两个属性。"""

    def __init__(self, host: str, headers: dict | None = None) -> None:
        self.client = _FakeClient(host)
        self.headers = headers or {}


# --------------------------------------------------------------------------- #
# 令牌桶限流
# --------------------------------------------------------------------------- #
class TestTokenBucketLimiter:
    def test_allows_up_to_capacity_then_blocks(self, monkeypatch):
        monkeypatch.setenv("LOGWISE_RATE_LIMIT", "3")
        monkeypatch.setenv("LOGWISE_RATE_WINDOW", "60")
        limiter = guard.TokenBucketLimiter()

        allowed = [limiter.allow("1.2.3.4")[0] for _ in range(5)]
        assert allowed == [True, True, True, False, False]

    def test_different_keys_are_isolated(self, monkeypatch):
        monkeypatch.setenv("LOGWISE_RATE_LIMIT", "1")
        monkeypatch.setenv("LOGWISE_RATE_WINDOW", "60")
        limiter = guard.TokenBucketLimiter()

        assert limiter.allow("a")[0] is True
        assert limiter.allow("a")[0] is False
        # 另一个来源 IP 不应受影响
        assert limiter.allow("b")[0] is True

    def test_retry_after_is_positive_when_blocked(self, monkeypatch):
        monkeypatch.setenv("LOGWISE_RATE_LIMIT", "1")
        monkeypatch.setenv("LOGWISE_RATE_WINDOW", "60")
        limiter = guard.TokenBucketLimiter()

        limiter.allow("ip")
        allowed, retry_after = limiter.allow("ip")
        assert allowed is False
        assert retry_after > 0

    def test_refills_over_time(self, monkeypatch):
        # 窗口最短 1 秒（rate_window 有下限），所以这里睡 1.05 秒
        monkeypatch.setenv("LOGWISE_RATE_LIMIT", "1")
        monkeypatch.setenv("LOGWISE_RATE_WINDOW", "1")
        limiter = guard.TokenBucketLimiter()

        assert limiter.allow("ip")[0] is True
        assert limiter.allow("ip")[0] is False
        time.sleep(1.05)
        assert limiter.allow("ip")[0] is True

    def test_snapshot_counts_clients(self, monkeypatch):
        monkeypatch.setenv("LOGWISE_RATE_LIMIT", "5")
        limiter = guard.TokenBucketLimiter()
        limiter.allow("a")
        limiter.allow("b")
        assert limiter.snapshot()["active_clients"] == 2


# --------------------------------------------------------------------------- #
# 鉴权
# --------------------------------------------------------------------------- #
class TestAuth:
    def test_auth_disabled_when_key_absent(self, monkeypatch):
        monkeypatch.delenv("LOGWISE_API_KEY", raising=False)
        assert guard.auth_enabled() is False
        # 未配置 Key 时，任何来源都应放行
        assert asyncio.run(guard.require_api_key(_FakeRequest("8.8.8.8"))) is None

    def test_rejects_missing_key_from_remote(self, monkeypatch):
        monkeypatch.setenv("LOGWISE_API_KEY", "secret")
        with pytest.raises(HTTPException) as exc:
            asyncio.run(guard.require_api_key(_FakeRequest("8.8.8.8")))
        assert exc.value.status_code == 401

    def test_rejects_wrong_key(self, monkeypatch):
        monkeypatch.setenv("LOGWISE_API_KEY", "secret")
        req = _FakeRequest("8.8.8.8", {"x-api-key": "wrong"})
        with pytest.raises(HTTPException) as exc:
            asyncio.run(guard.require_api_key(req))
        assert exc.value.status_code == 401

    def test_accepts_correct_key(self, monkeypatch):
        monkeypatch.setenv("LOGWISE_API_KEY", "secret")
        req = _FakeRequest("8.8.8.8", {"x-api-key": "secret"})
        assert asyncio.run(guard.require_api_key(req)) is None

    def test_localhost_bypasses_key(self, monkeypatch):
        monkeypatch.setenv("LOGWISE_API_KEY", "secret")
        monkeypatch.setenv("LOGWISE_TRUST_LOCALHOST", "true")
        # 没有带 Key，但来源是本机 —— 应放行
        assert asyncio.run(guard.require_api_key(_FakeRequest("127.0.0.1"))) is None

    def test_localhost_bypass_can_be_disabled(self, monkeypatch):
        monkeypatch.setenv("LOGWISE_API_KEY", "secret")
        monkeypatch.setenv("LOGWISE_TRUST_LOCALHOST", "false")
        with pytest.raises(HTTPException):
            asyncio.run(guard.require_api_key(_FakeRequest("127.0.0.1")))


class TestClientIp:
    def test_ignores_forwarded_header_by_default(self, monkeypatch):
        """默认不采信 X-Forwarded-For，否则限流和本机豁免都能被伪造绕过。"""
        monkeypatch.delenv("LOGWISE_TRUST_PROXY_HEADERS", raising=False)
        req = _FakeRequest("10.0.0.5", {"x-forwarded-for": "1.2.3.4"})
        assert guard.client_ip(req) == "10.0.0.5"

    def test_uses_forwarded_header_when_explicitly_enabled(self, monkeypatch):
        monkeypatch.setenv("LOGWISE_TRUST_PROXY_HEADERS", "true")
        req = _FakeRequest("10.0.0.5", {"x-forwarded-for": "1.2.3.4, 5.6.7.8"})
        assert guard.client_ip(req) == "1.2.3.4"

    def test_is_localhost(self, monkeypatch):
        monkeypatch.delenv("LOGWISE_TRUST_PROXY_HEADERS", raising=False)
        assert guard.is_localhost(_FakeRequest("127.0.0.1")) is True
        assert guard.is_localhost(_FakeRequest("::1")) is True
        assert guard.is_localhost(_FakeRequest("10.0.0.5")) is False


# --------------------------------------------------------------------------- #
# 缓存
# --------------------------------------------------------------------------- #
class TestDiagnosisCache:
    def test_normalize_folds_whitespace(self):
        a = cache_mod.normalize("ERROR  foo\r\n\r\n\r\n   bar  ")
        b = cache_mod.normalize("ERROR foo\nbar")
        assert a == b

    def test_same_log_different_whitespace_shares_key(self):
        """用户从不同地方复制的日志，空格/空行差异不应导致缓存失效。"""
        assert cache_mod.make_key("a  b\n\nc", "m", "v1") == cache_mod.make_key("a b\nc", "m", "v1")

    def test_key_varies_with_model_and_prompt_version(self):
        base = cache_mod.make_key("x", "m1", "v1")
        assert base != cache_mod.make_key("x", "m2", "v1")
        assert base != cache_mod.make_key("x", "m1", "v2")

    def test_set_and_get(self, monkeypatch):
        monkeypatch.setenv("LOGWISE_CACHE_TTL", "60")
        c = cache_mod.DiagnosisCache()
        c.set("k", "value")
        assert c.get("k") == "value"

    def test_miss_returns_none(self, monkeypatch):
        c = cache_mod.DiagnosisCache()
        assert c.get("never-set") is None

    def test_expired_entry_is_treated_as_miss(self, monkeypatch):
        monkeypatch.setenv("LOGWISE_CACHE_TTL", "0")
        c = cache_mod.DiagnosisCache()
        c.set("k", "value")
        assert c.get("k") is None

    def test_lru_eviction(self, monkeypatch):
        monkeypatch.setenv("LOGWISE_CACHE_TTL", "60")
        monkeypatch.setenv("LOGWISE_CACHE_SIZE", "2")
        c = cache_mod.DiagnosisCache()

        c.set("a", "1")
        c.set("b", "2")
        c.get("a")          # a 变为最近使用
        c.set("c", "3")     # 应淘汰 b

        assert c.get("a") == "1"
        assert c.get("b") is None
        assert c.get("c") == "3"

    def test_snapshot_reports_hit_rate(self, monkeypatch):
        monkeypatch.setenv("LOGWISE_CACHE_TTL", "60")
        c = cache_mod.DiagnosisCache()
        c.set("k", "v")
        c.get("k")          # 命中
        c.get("nope")       # 未命中
        snap = c.snapshot()
        assert snap["hits"] == 1
        assert snap["misses"] == 1
        assert snap["hit_rate"] == 0.5

    def test_chunk_text_splits_into_pieces(self):
        pieces = cache_mod.chunk_text("abcdefghij", size=4)
        assert pieces == ["abcd", "efgh", "ij"]
        assert cache_mod.chunk_text("") == [""]
        assert "".join(cache_mod.chunk_text("任意内容" * 50)) == "任意内容" * 50


# --------------------------------------------------------------------------- #
# 度量
# --------------------------------------------------------------------------- #
class TestMetrics:
    def test_cjk_estimation(self):
        assert metrics.estimate_tokens("") == 0
        assert metrics.estimate_tokens("你好世界") == 4        # 4 个汉字 → 4 token
        assert metrics.estimate_tokens("abcdefgh") == 2        # 8 个 ASCII → 2 token

    def test_cost_uses_configured_prices(self, monkeypatch):
        monkeypatch.setenv("DEEPSEEK_PRICE_INPUT", "1.0")
        monkeypatch.setenv("DEEPSEEK_PRICE_OUTPUT", "2.0")
        rec = metrics.CallRecord(
            endpoint="/diagnose",
            duration_ms=100,
            prompt_tokens=1_000_000,
            completion_tokens=1_000_000,
        )
        assert rec.cost() == pytest.approx(3.0)

    def test_cache_hit_costs_nothing_even_with_tokens(self, monkeypatch):
        monkeypatch.setenv("DEEPSEEK_PRICE_INPUT", "1.0")
        monkeypatch.setenv("DEEPSEEK_PRICE_OUTPUT", "2.0")
        rec = metrics.CallRecord(
            endpoint="/diagnose",
            duration_ms=5,
            prompt_tokens=999_999,
            completion_tokens=999_999,
            cache_hit=True,
        )
        assert rec.cost() == 0.0

    def test_summary_aggregates(self, monkeypatch):
        monkeypatch.setenv("DEEPSEEK_PRICE_INPUT", "1.0")
        monkeypatch.setenv("DEEPSEEK_PRICE_OUTPUT", "2.0")
        store = metrics.MetricsStore()

        store.record(metrics.CallRecord(
            endpoint="/diagnose", duration_ms=100,
            prompt_tokens=1000, completion_tokens=500, llm_call=True))
        store.record(metrics.CallRecord(
            endpoint="/diagnose", duration_ms=10,
            llm_call=True, cache_hit=True))
        store.record(metrics.CallRecord(
            endpoint="/parse", duration_ms=2))

        out = store.summary()
        assert out["requests"]["total"] == 3
        assert out["requests"]["llm_calls"] == 2
        assert out["requests"]["cache_hits"] == 1
        assert out["requests"]["cache_hit_rate"] == 0.5
        assert out["requests"]["billable_calls"] == 1
        assert out["tokens"]["total"] == 1500
        # 1000/1e6*1.0 + 500/1e6*2.0 = 0.002
        assert out["cost"]["total"] == pytest.approx(0.002)
        assert out["cost"]["avg_per_billable_call"] == pytest.approx(0.002)

    def test_error_counting_and_reset(self):
        store = metrics.MetricsStore()
        store.record(metrics.CallRecord(endpoint="/diagnose", duration_ms=1, error="boom"))
        assert store.summary()["requests"]["errors"] == 1

        store.reset()
        assert store.summary()["requests"]["total"] == 0
        assert store.summary()["cost"]["total"] == 0.0

    def test_summary_empty_store_is_safe(self):
        """没有任何请求时 /stats 不应崩（除零）。"""
        out = metrics.MetricsStore().summary()
        assert out["latency_ms"]["avg"] == 0.0
        assert out["cost"]["avg_per_billable_call"] == 0.0
        assert out["requests"]["cache_hit_rate"] == 0.0

    def test_cached_input_uses_cheaper_price(self, monkeypatch):
        """命中上下文缓存的输入 token 必须按更低单价计算，否则成本会高估。"""
        monkeypatch.setenv("DEEPSEEK_PRICE_INPUT", "1.0")
        monkeypatch.setenv("DEEPSEEK_PRICE_INPUT_CACHED", "0.2")
        monkeypatch.setenv("DEEPSEEK_PRICE_OUTPUT", "2.0")

        rec = metrics.CallRecord(
            endpoint="/diagnose",
            duration_ms=10,
            prompt_tokens=1_000_000,
            cached_prompt_tokens=800_000,
            completion_tokens=0,
        )
        # 20 万未命中 × 1.0 + 80 万命中 × 0.2 = 0.2 + 0.16
        assert rec.uncached_prompt_tokens == 200_000
        assert rec.cost() == pytest.approx(0.36)

    def test_cached_tokens_never_make_uncached_negative(self):
        rec = metrics.CallRecord(
            endpoint="/x", duration_ms=1,
            prompt_tokens=100, cached_prompt_tokens=500,
        )
        assert rec.uncached_prompt_tokens == 0

    def test_summary_reports_prompt_cache_rate(self):
        store = metrics.MetricsStore()
        store.record(metrics.CallRecord(
            endpoint="/diagnose", duration_ms=10, llm_call=True,
            prompt_tokens=1000, cached_prompt_tokens=700, completion_tokens=100,
        ))
        tokens = store.summary()["tokens"]
        assert tokens["cached_prompt"] == 700
        assert tokens["uncached_prompt"] == 300
        assert tokens["cache_rate"] == 0.7

    def test_summary_exposes_three_price_tiers(self, monkeypatch):
        monkeypatch.setenv("DEEPSEEK_PRICE_INPUT_CACHED", "0.2")
        price = metrics.MetricsStore().summary()["price"]
        assert price["input_cached"] == 0.2
        assert "input_uncached" in price
        assert "output" in price


# --------------------------------------------------------------------------- #
# usage 解析（token 用量提取）
# --------------------------------------------------------------------------- #
class TestUsageExtraction:
    def test_reads_openai_style_prompt_tokens_details(self):
        from app.diagnose import _extract_usage

        class _Details:
            cached_tokens = 700

        class _Raw:
            prompt_tokens = 1000
            completion_tokens = 50
            prompt_tokens_details = _Details()

        got = _extract_usage(_Raw())
        assert (got.prompt_tokens, got.completion_tokens, got.cached_tokens) == (1000, 50, 700)
        assert got.uncached_prompt_tokens == 300

    def test_falls_back_to_deepseek_top_level_field(self):
        from app.diagnose import _extract_usage

        class _Raw:
            prompt_tokens = 1000
            completion_tokens = 50
            prompt_tokens_details = None
            prompt_cache_hit_tokens = 640

        assert _extract_usage(_Raw()).cached_tokens == 640

    def test_none_usage_is_safe(self):
        from app.diagnose import _extract_usage

        got = _extract_usage(None)
        assert (got.prompt_tokens, got.completion_tokens, got.cached_tokens) == (0, 0, 0)
        assert got.total == 0

    def test_cached_tokens_are_clamped_to_prompt_tokens(self):
        """防御：命中数不该超过输入总数，否则未命中会算成负数。"""
        from app.diagnose import _extract_usage

        class _Raw:
            prompt_tokens = 100
            completion_tokens = 1
            prompt_tokens_details = None
            prompt_cache_hit_tokens = 999

        assert _extract_usage(_Raw()).cached_tokens == 100
