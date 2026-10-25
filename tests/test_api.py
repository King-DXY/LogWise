"""HTTP 接口层的测试。

重点守护一个真实踩过的坑：**日志不能放在 URL query string 里传**。
前端原本用 `GET /diagnose/stream?text=...`，实测 query string 长度上限约 32K 字符，
再长会被 ASGI 服务器直接以 400 拒绝 —— 而「日志长」恰恰是上下文聚合最有价值的场景。

现在同时提供 GET 和 POST 两个版本，前端统一走 POST。
"""
from __future__ import annotations

import json

import pytest
from fastapi.testclient import TestClient

from app.main import app

SMALL_LOG = """2026-09-18 21:15:03,440 INFO  [health] com.example.Health - 健康检查通过
2026-09-18 21:15:04,550 ERROR [gateway] com.example.OrderClient - 调用订单服务失败: connection refused
2026-09-18 21:15:05,660 WARN  [gateway] com.example.CircuitBreaker - 熔断器已打开
"""

# 约 60K 字符 —— 远超 query string 的上限，但 POST 应该毫无压力
BIG_LOG = "2026-09-18 21:00:00,000 INFO  [x] com.example.A - 填充日志内容\n" * 1500


@pytest.fixture(scope="module")
def client() -> TestClient:
    return TestClient(app)


@pytest.fixture(autouse=True)
def _no_rate_limit(monkeypatch):
    """测试里关掉限流，避免连续请求被 429 干扰。"""
    monkeypatch.setenv("LOGWISE_RATE_LIMIT", "0")


class TestContextEndpoint:
    def test_post_context_returns_selection(self, client: TestClient):
        r = client.post("/context", json={"text": SMALL_LOG})
        assert r.status_code == 200
        data = r.json()
        assert data["total_parsed"] == 3
        assert data["anchor_index"] == 1          # 第一条 ERROR
        assert data["selected"] == 3              # 窗口够大，三条全选
        assert len(data["logs"]) == 3
        assert sum(1 for log in data["logs"] if log["is_anchor"]) == 1

    def test_get_context_still_works(self, client: TestClient):
        """GET 版保留，避免破坏已有调用方。"""
        r = client.get("/context", params={"text": SMALL_LOG})
        assert r.status_code == 200
        assert r.json()["total_parsed"] == 3

    def test_post_context_handles_long_log(self, client: TestClient):
        """核心回归：长日志必须走 POST 才不会被 URL 长度限制挡住。"""
        assert len(BIG_LOG) > 50_000
        r = client.post("/context", json={"text": BIG_LOG})
        assert r.status_code == 200, "长日志通过 POST 提交不应该失败"
        data = r.json()
        assert data["total_parsed"] == 1500
        assert data["selected"] <= data["max_logs"], "选中的条数不应超过上限"

    def test_blank_log_returns_error(self, client: TestClient):
        """纯空白才解析不出日志。

        注意解析器是「宽容」的：任何非空文本都会兜底产出一条日志
        （这是 `tests/test_parser.py::test_unknown_format_graceful` 保证的行为），
        所以这里只能用空白文本来触发「解析不到日志」这条分支。
        """
        r = client.post("/context", json={"text": "   \n  \n"})
        assert r.status_code == 200
        assert "error" in r.json()

    def test_context_is_counted_in_request_total(self, client: TestClient):
        """``/context`` 必须计入「请求总数」。

        它本来是唯一一个不写度量的业务接口，于是 total 并不是真的 total。
        （``/stats`` 自己不写度量，所以取 stats 不会污染计数。）
        """
        before = client.get("/stats").json()["metrics"]["requests"]["total"]
        client.post("/context", json={"text": SMALL_LOG})
        after = client.get("/stats").json()["metrics"]["requests"]["total"]
        assert after == before + 1

    def test_context_does_not_touch_the_model(self, client: TestClient):
        """聚合预览必须是纯本地的 —— 它不调用大模型，因此不产生任何费用。

        ⚠️ 这条断言原本是 ``assert after >= before``，恒为真 —— 等于什么都没测，
        而且当时 /context 压根没写度量，after == before 是恒定的。
        现在改成逐项比对，任何一项变了都会失败。
        """
        before = client.get("/stats").json()["metrics"]["requests"]
        client.post("/context", json={"text": SMALL_LOG})
        after = client.get("/stats").json()["metrics"]["requests"]

        assert after["total"] == before["total"] + 1
        assert after["llm_calls"] == before["llm_calls"], "/context 绝不该调用模型"
        assert after["cache_hits"] == before["cache_hits"]
        assert after["billable_calls"] == before["billable_calls"]


class TestContextDisplayCap:
    """日志很大时只展示一部分 —— 否则响应体和 DOM 行数会把浏览器拖垮。

    实测：2 万条日志的响应体约 4MB，前端要渲染 2 万行 DOM。
    **注意这只影响展示，不影响实际送给模型的日志。**
    """

    def _big(self, n: int) -> str:
        return "2026-09-18 21:00:00,000 INFO  [x] com.example.A - 填充日志内容\n" * n

    def test_small_log_is_fully_returned(self, client: TestClient):
        r = client.post("/context", json={"text": self._big(20)})
        data = r.json()
        assert data["display_limited"] is False
        assert data["displayed"] == 20

    def test_huge_log_is_capped(self, client: TestClient):
        from app.main import MAX_CONTEXT_DISPLAY

        r = client.post("/context", json={"text": self._big(20_000)})
        data = r.json()
        assert data["total_parsed"] == 20_000
        assert data["display_limited"] is True
        assert data["displayed"] <= MAX_CONTEXT_DISPLAY
        assert len(data["logs"]) <= MAX_CONTEXT_DISPLAY
        assert len(r.content) < 1_000_000, "响应体不应膨胀到 MB 级"

    def test_cap_never_hides_the_anchor(self, client: TestClient):
        r = client.post("/context", json={"text": self._big(5_000)})
        data = r.json()
        anchors = [log for log in data["logs"] if log["is_anchor"]]
        assert len(anchors) == 1, "裁展示时锚点必须可见"

    def test_cap_never_hides_selected_logs(self, client: TestClient):
        """被选中的日志是这块 UI 的核心信息，一个都不能少。"""
        r = client.post("/context", json={"text": self._big(5_000)})
        data = r.json()
        visible_selected = sum(1 for log in data["logs"] if log["selected"])
        assert visible_selected == data["selected"]

    def test_display_indices_are_ascending(self, client: TestClient):
        r = client.post("/context", json={"text": self._big(5_000)})
        indices = [log["index"] for log in r.json()["logs"]]
        assert indices == sorted(indices)
        assert len(indices) == len(set(indices)), "不应出现重复行"


class TestMissingTextRejected:
    def test_post_without_text_is_422(self, client: TestClient):
        assert client.post("/context", json={}).status_code == 422

    def test_post_with_wrong_type_is_422(self, client: TestClient):
        assert client.post("/context", json={"text": 123}).status_code == 422


class TestBadModelOutput:
    """模型返回非 JSON 时，接口要给一句人话，而不是把 JSONDecodeError 甩出来。

    对应用户实测遇到的一次 502：

        {"detail":"模型调用失败：JSONDecodeError: Expecting value: line 1 column 1 (char 0)"}

    修复后：校验结构 → 不合格就把问题反馈给模型重试一次 → 仍不合格才报错，
    且报错内容里带上模型的原话，便于判断到底发生了什么。
    """

    LOG = "2026-09-20 10:00:00,000 ERROR [x] com.example.A - 磁盘写入失败"
    GOOD = {
        "error_type": "磁盘空间不足",
        "root_cause": "日志分区写满",
        "evidence": ["No space left on device"],
        "impact": "写入失败",
        "solution_steps": ["df -h"],
        "confidence": 0.9,
    }
    PROSE = "抱歉，这段日志信息不足，我无法给出结构化诊断。"

    def _fake(self, contents: list[str]):
        import json as _json
        from types import SimpleNamespace
        from app import diagnose

        outer = {"i": 0}

        class _Completions:
            def create(self, **kwargs):
                idx = min(outer["i"], len(contents) - 1)
                outer["i"] += 1
                return SimpleNamespace(
                    choices=[SimpleNamespace(
                        message=SimpleNamespace(content=contents[idx]))],
                    usage=SimpleNamespace(prompt_tokens=100, completion_tokens=50,
                                          prompt_tokens_details=None),
                )

        return SimpleNamespace(chat=SimpleNamespace(completions=_Completions())), _json, diagnose

    def test_retry_recovers_so_the_request_succeeds(self, client: TestClient, monkeypatch):
        import json as _json

        from app import diagnose

        fake, _, _ = self._fake([self.PROSE, _json.dumps(self.GOOD)])
        monkeypatch.setattr(diagnose, "get_client", lambda: fake)

        r = client.post("/diagnose", json={"text": self.LOG})
        assert r.status_code == 200
        assert r.json()["diagnosis"]["error_type"] == "磁盘空间不足"

    def test_exhausted_retries_give_a_readable_error(self, client: TestClient, monkeypatch):
        from app import cache as cache_mod
        from app import diagnose

        fake, _, _ = self._fake([self.PROSE])   # 一直不合格
        monkeypatch.setattr(diagnose, "get_client", lambda: fake)
        cache_mod.cache.reset()

        r = client.post("/diagnose", json={"text": self.LOG + " 换个内容避开缓存"})
        assert r.status_code == 502
        detail = r.json()["detail"]
        assert "JSONDecodeError" not in detail, "不该把 Python 异常名甩给用户"
        assert "无法给出结构化诊断" in detail, "应带上模型的原话，便于判断"


class TestCacheKeyRetrieval:
    """fixed 模式下算缓存键不该发起检索。

    ``_cache_key`` 曾经无条件调 ``rag.retrieve``：fixed 模式下它本来就返回空，
    调用是纯浪费；dense/hybrid 后端时更是一次真实的 embedding 网络请求。
    """

    def test_fixed_mode_does_not_retrieve(self, monkeypatch):
        from app import main, rag
        from app.context import build_context
        from app.parser import parse_text

        monkeypatch.setenv("LOGWISE_RAG_MODE", "fixed")

        def _boom(query):
            raise AssertionError("fixed 模式不该发起检索")

        monkeypatch.setattr(rag, "retrieve", _boom)
        ctx = build_context(parse_text(SMALL_LOG))
        key = main._cache_key(SMALL_LOG, ctx)
        assert isinstance(key, str) and len(key) == 64

    def test_fixed_mode_signature_is_stable(self, monkeypatch):
        """签名必须与旧实现一致 —— 否则优化一次，线上缓存全部失效。"""
        from app import main, rag
        from app.context import build_context
        from app.parser import parse_text

        monkeypatch.setenv("LOGWISE_RAG_MODE", "fixed")
        ctx = build_context(parse_text(SMALL_LOG))
        key = main._cache_key(SMALL_LOG, ctx)
        assert rag.signature() == "fixed"
        # 同样输入重算一次，键必须相同（签名里没有随机成分）
        assert main._cache_key(SMALL_LOG, ctx) == key


class TestStreamCacheSafety:
    """SSE 流被客户端中途断开时，**半截结果不能写入缓存**。

    曾经的 bug：``finally`` 里只判断 ``not error and full_text``，
    而客户端断开（GeneratorExit）时 error 为空、full_text 是截断的 JSON ——
    这份残缺结果会被写进缓存，下次同一段日志命中后回放出来的就是截断的诊断。
    """

    LOG = SMALL_LOG

    @staticmethod
    def _fake_stream(chunks: list[str]):
        async def _gen(logs, usage):
            for c in chunks:
                yield c

        return _gen

    def _drive(self, monkeypatch, chunks: list[str], take: int | None):
        """跑一遍流式接口。take=None 表示读到结束；否则读 take 个 chunk 后断开。"""
        import asyncio

        from app import cache as cache_mod
        from app import main

        monkeypatch.setattr(main, "stream_diagnosis_multi", self._fake_stream(chunks))

        async def _run():
            resp = main._stream_response(self.LOG)
            it = resp.body_iterator
            if take is None:
                async for _ in it:
                    pass
            else:
                for _ in range(take):
                    await it.__anext__()
                await it.aclose()  # 模拟客户端中途断开

        asyncio.run(_run())
        return cache_mod.cache

    def test_partial_stream_is_not_cached(self, client: TestClient, monkeypatch):
        from app import cache as cache_mod

        cache_mod.cache.reset()
        cache = self._drive(monkeypatch, ['{"error_type": "x",', ' "root_cause":'], take=1)
        assert cache.snapshot()["size"] == 0, "半截结果不该入缓存"

    def test_completed_stream_is_cached(self, client: TestClient, monkeypatch):
        from app import cache as cache_mod

        cache_mod.cache.reset()
        payload = json.dumps(TestBadModelOutput.GOOD, ensure_ascii=False)
        cache = self._drive(monkeypatch, [payload[:30], payload[30:]], take=None)
        assert cache.snapshot()["size"] == 1, "完整且符合 schema 的流应该入缓存"

    @pytest.mark.parametrize(
        "chunks",
        [
            ["这不是 JSON"],
            ['{"error_type": "x",'],
            ['{"error_type": "x"}'],
        ],
        ids=["non-json", "truncated-json", "missing-fields"],
    )
    def test_invalid_completed_stream_is_not_cached(self, client: TestClient, monkeypatch, chunks):
        from app import cache as cache_mod

        cache_mod.cache.reset()
        cache = self._drive(monkeypatch, chunks, take=None)
        assert cache.snapshot()["size"] == 0, "流结束不代表结果合格，非法诊断不能入缓存"

    def test_corrupt_cache_is_rebuilt_instead_of_500(self, client: TestClient, monkeypatch):
        from app import cache as cache_mod
        from app import main
        from app.context import build_context
        from app.diagnose import Usage
        from app.parser import parse_text

        cache_mod.cache.reset()
        monkeypatch.setenv("LOGWISE_RAG_MODE", "fixed")
        context = build_context(parse_text(self.LOG))
        key = main._cache_key(self.LOG, context)
        cache_mod.cache.set(key, '{"error_type": "x"}')  # 合法 JSON，但缺诊断字段
        monkeypatch.setattr(
            main, "diagnose_multi_ex", lambda logs: (TestBadModelOutput.GOOD, Usage())
        )

        r = client.post("/diagnose", json={"text": self.LOG})
        assert r.status_code == 200
        assert r.json()["cached"] is False
        assert r.json()["diagnosis"] == TestBadModelOutput.GOOD
        assert main._read_cached_diagnosis(key) == TestBadModelOutput.GOOD


class TestAgentStreamEvents:
    """``/diagnose/agent/stream`` 的事件必须是 **JSON 对象**。

    真实踩过的坑：接口里写成 ``_sse(json.dumps(event))`` 而 ``_sse`` 内部又
    ``json.dumps`` 一次 —— 线上传出去的是 ``data: "{\\"type\\": ...}"``，
    也就是一个 JSON **字符串**。前端 ``JSON.parse`` 得到字符串，``ev.type``
    永远是 undefined，整个 Agent 时间线静默不渲染。这类 bug 单测不覆盖接口层就发现不了。
    """

    LOG = "2026-10-17 10:00:00,000 ERROR [t] com.example.A - 编码回归用例"
    DIAG = {
        "error_type": "磁盘不足",
        "root_cause": "分区写满",
        "evidence": ["No space left on device"],
        "impact": "写入失败",
        "solution_steps": ["df -h"],
        "confidence": 0.9,
    }

    @staticmethod
    def _events(r) -> list:
        import json as _json

        out = []
        for line in r.text.splitlines():
            if not line.startswith("data: "):
                continue
            payload = line[6:]
            if payload == "[DONE]":
                continue
            out.append(_json.loads(payload))
        return out

    def test_events_are_objects_not_double_encoded(self, client: TestClient, monkeypatch):
        from app import cache as cache_mod

        cache_mod.cache.reset()

        def fake_stream(logs, max_rounds=None):
            yield {"type": "tool_call", "round": 1,
                   "tool_calls": [{"name": "search_fault_cases", "arguments": "{}", "result_excerpt": ""}]}
            yield {"type": "delta", "text": '{"error_type": "磁盘不足"}'}
            yield {"type": "done", "diagnosis": self.DIAG,
                   "usage": {"prompt_tokens": 10, "cached_tokens": 0, "completion_tokens": 5},
                   "trace": {"rounds": [], "tool_calls_total": 1, "forced_answer": False}}

        monkeypatch.setattr("app.agent.stream_agent", fake_stream)
        r = client.post("/diagnose/agent/stream", json={"text": self.LOG})
        assert r.status_code == 200

        events = self._events(r)
        assert all(isinstance(e, dict) for e in events), "事件必须是对象，不能是双重编码的字符串"
        assert [e["type"] for e in events] == ["tool_call", "delta", "done"]
        assert events[-1]["session_id"], "done 事件应带上会话 id（用于追问）"
        assert events[-1]["usage"] == {
            "prompt_tokens": 10,
            "cached_tokens": 0,
            "completion_tokens": 5,
            "cost_yuan": 0.00002,
        }

        # 同一 Agent 缓存从流式接口回放时，usage 契约不能漂移。
        replay = client.post("/diagnose/agent/stream", json={"text": self.LOG})
        replay_done = self._events(replay)[-1]
        assert replay_done["cached"] is True
        assert replay_done["usage"] == events[-1]["usage"]

    def test_corrupt_agent_cache_is_discarded(self, client: TestClient):
        from app import cache as cache_mod
        from app import main

        cache_mod.cache.reset()
        cache_mod.cache.set("broken-agent", json.dumps({"diagnosis": self.DIAG}))
        assert main._read_cached_agent("broken-agent") is None
        assert cache_mod.cache.snapshot()["size"] == 0

    def test_empty_log_error_is_also_object(self, client: TestClient):
        r = client.post("/diagnose/agent/stream", json={"text": "   "})
        events = self._events(r)
        assert len(events) == 1 and events[0]["type"] == "error"


class TestNewEndpointsProtected:
    """新增的 Agent 流式 / 追问接口必须挂上鉴权，别漏了 PROTECTED。"""

    def test_agent_stream_requires_key(self, client: TestClient, monkeypatch):
        monkeypatch.setenv("LOGWISE_API_KEY", "secret-key")
        r = client.post("/diagnose/agent/stream", json={"text": "x"})
        assert r.status_code == 401

    def test_followup_requires_key(self, client: TestClient, monkeypatch):
        monkeypatch.setenv("LOGWISE_API_KEY", "secret-key")
        r = client.post("/diagnose/followup", json={"session_id": "x", "question": "?"})
        assert r.status_code == 401

    def test_correct_key_passes(self, client: TestClient, monkeypatch):
        monkeypatch.setenv("LOGWISE_API_KEY", "secret-key")
        r = client.post(
            "/diagnose/followup",
            json={"session_id": "x", "question": "?"},
            headers={"X-API-Key": "secret-key"},
        )
        # 带了正确 Key → 走到业务逻辑，会话不存在应 404 而不是 401
        assert r.status_code == 404


class TestHealthHandshake:
    """``/health`` 兼作接口契约版本握手。

    前端靠它判断「跑着的后端是不是旧代码」。之所以不拿业务接口来探测：
    ``/health`` 不计度量、不受限流，页面每次刷新探测一下不会污染 /stats，
    也不会把限流额度吃掉。
    """

    def test_reports_api_revision(self, client: TestClient):
        from app.main import API_REVISION

        r = client.get("/health")
        assert r.status_code == 200
        assert r.json() == {"status": "ok", "api": API_REVISION}

    def test_health_is_not_counted_in_metrics(self, client: TestClient):
        before = client.get("/stats").json()["metrics"]["requests"]["total"]
        client.get("/health")
        after = client.get("/stats").json()["metrics"]["requests"]["total"]
        assert after == before, "/health 不该计入请求数，否则版本探测会污染统计"
