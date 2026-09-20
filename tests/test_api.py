"""HTTP 接口层的测试。

重点守护一个真实踩过的坑：**日志不能放在 URL query string 里传**。
前端原本用 `GET /diagnose/stream?text=...`，实测 query string 长度上限约 32K 字符，
再长会被 ASGI 服务器直接以 400 拒绝 —— 而「日志长」恰恰是上下文聚合最有价值的场景。

现在同时提供 GET 和 POST 两个版本，前端统一走 POST。
"""
from __future__ import annotations

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
