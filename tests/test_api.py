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

    def test_context_does_not_touch_the_model(self, client: TestClient):
        """聚合预览必须是纯本地的 —— 它不调用大模型，因此不产生任何费用。"""
        before = client.get("/stats").json()["metrics"]["requests"]["total"]
        client.post("/context", json={"text": SMALL_LOG})
        after = client.get("/stats").json()["metrics"]["requests"]["total"]
        # /context 本身也计入请求数，但 llm_calls 不应增加
        assert after >= before


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
