"""检索层测试：分词、打分、以及两个踩过的坑。

为什么这两个坑值得单独写测试
----------------------------
它们都是**不会报错的性能问题** —— 检索照样返回结果，只是返回的是错的。
不测的话，只会在某次评测里看到"分数怎么有点低"，然后花很久才能定位。

坑 1：query 侧词频被线性放大
    日志里"健康检查通过"重复 10 次，早期实现按出现次数累加，
    该词权重被放大 10 倍，把一条恰好含相同噪音的案例顶到检索第一（760 分），
    真正该命中的案例排第 4。
坑 2：无信息量的行没有被过滤
    同上，这类行在故障上下文里没有任何区分度，却能在数量上压倒异常行。
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from app.retriever import (
    BM25Retriever,
    DenseRetriever,
    HybridRetriever,
    IndexNotBuiltError,
    _NOISE_LINE,
    load_cases,
    make_retriever,
    reset_retriever,
    rrf_fuse,
    tokenize,
    weighted_tokens,
)
from tests._helpers import fake_vector

KB_PATH = Path(__file__).resolve().parent.parent / "data" / "fault_cases.jsonl"


class TestTokenize:
    def test_splits_camel_case(self):
        """FileNotFoundException 必须能拆成 file / not / found / exception。"""
        toks = tokenize("java.io.FileNotFoundException")
        assert "file" in toks
        assert "not" in toks
        assert "found" in toks
        assert "exception" in toks

    def test_keeps_whole_identifier(self):
        """整体也要保留，让完整类名本身成为可精确匹配的词。"""
        toks = tokenize("HikariPool")
        assert "hikaripool" in toks
        assert "hikari" in toks and "pool" in toks

    def test_splits_underscore_and_dash(self):
        toks = tokenize("ARRAY_INDEX_OUT_OF_BOUNDS")
        assert "array" in toks and "index" in toks and "bounds" in toks

    def test_chinese_two_gram(self):
        toks = tokenize("磁盘空间不足")
        assert "磁盘" in toks
        assert "空间" in toks
        assert "不足" in toks

    def test_digits_are_kept_as_part_of_identifier(self):
        toks = tokenize("HDFS_2k")
        assert any(t.startswith("hdfs") for t in toks)


class TestNoiseFilter:
    def test_noise_lines_are_dropped(self):
        """健康检查 / 指标上报这类行不含区分度，检索时必须跳过。"""
        text = "\n".join([
            "2025-01-01 10:00:00,000 INFO  [health] com.example.health.HealthChecker - 健康检查通过",
            "2025-01-01 10:00:01,000 INFO  [metrics] com.example.metrics.Reporter - 指标上报成功",
        ])
        assert weighted_tokens(text) == []

    def test_exception_lines_survive(self):
        text = "java.io.IOException: No space left on device"
        assert weighted_tokens(text)


@pytest.fixture(scope="module")
def retriever() -> BM25Retriever:
    return BM25Retriever.from_jsonl(KB_PATH)


class TestBM25Search:
    def test_loads_knowledge_base(self, retriever: BM25Retriever):
        assert retriever.size() >= 40

    def test_empty_query_returns_nothing(self, retriever: BM25Retriever):
        assert retriever.search("", k=3) == []

    def test_finds_same_fault_group(self, retriever: BM25Retriever):
        """一段磁盘写满的日志，应该检索到 disk 类的历史案例。"""
        query = (
            "2026-01-01 10:00:00,000 ERROR [writer] com.example.log.FileWriter - 写入失败\n"
            "java.io.IOException: No space left on device\n"
            "\tat com.example.log.FileWriter.flush(FileWriter.java:50)"
        )
        hits = retriever.search(query, k=3)
        assert hits, "应当至少召回一条"
        assert any(h.fault_group == "disk" for h in hits), (
            f"未召回 disk 类案例，实际召回: {[(h.id, h.fault_group) for h in hits]}"
        )

    def test_exclude_removes_case(self, retriever: BM25Retriever):
        """留一法要用到这个参数：评测某条案例时把它自己排除。"""
        query = "java.io.IOException: No space left on device"
        first = retriever.search(query, k=5)
        assert first
        excluded = {first[0].id}
        again = retriever.search(query, k=5, exclude=excluded)
        assert all(h.id not in excluded for h in again)

    def test_results_are_sorted_by_score(self, retriever: BM25Retriever):
        hits = retriever.search("connection refused", k=5)
        scores = [h.score for h in hits]
        assert scores == sorted(scores, reverse=True)

    def test_as_few_shot_is_prompt_ready(self, retriever: BM25Retriever):
        """检索结果要能直接拼进 Prompt：user 是日志，assistant 是合法 JSON。"""
        hit = retriever.search("java.lang.OutOfMemoryError: Java heap space", k=1)[0]
        user, assistant = hit.as_few_shot()
        assert user == hit.case["log"]
        parsed = json.loads(assistant)
        assert {"error_type", "root_cause", "evidence", "impact", "solution_steps", "confidence"} == set(parsed)


class TestQueryTermFrequencyRegression:
    """回归测试：query 侧不得按词频放大权重（坑 1）。"""

    def test_repeated_noise_cannot_dominate(self):
        """20 条健康检查 + 1 条磁盘错误 —— 结果必须是磁盘类，不能被噪音带偏。"""
        retriever = BM25Retriever.from_jsonl(KB_PATH)
        noise = "\n".join(
            "2026-01-01 10:00:%02d,000 INFO  [health] com.example.health.HealthChecker - 健康检查通过" % i
            for i in range(20)
        )
        signal = (
            "2026-01-01 10:01:00,000 ERROR [writer] com.example.log.FileWriter - 写入失败\n"
            "java.io.IOException: No space left on device"
        )
        hits = retriever.search(noise + "\n" + signal, k=3)
        assert hits, "应当至少召回一条"
        assert hits[0].fault_group == "disk", (
            f"重复的噪音行把结果带偏了，top-1 是 {hits[0].id}({hits[0].fault_group})。"
            "检查 search() 里 query 侧是否仍在使用 Counter（应为 set 二值化）。"
        )

    def test_noise_regex_matches_known_forms(self):
        for text in (
            "com.example.health.HealthChecker - 健康检查通过",
            "com.example.metrics.Reporter - 指标上报成功",
            "心跳正常",
        ):
            assert _NOISE_LINE.search(text), f"未识别为噪音行: {text}"

    def test_noise_regex_does_not_eat_real_errors(self):
        """不能误伤真正的错误行 —— 这里出现过 HealthChecker 但后面接的是故障描述。"""
        assert not _NOISE_LINE.search("com.example.api.HealthChecker - 本地磁盘缓存子系统不可用")
        assert not _NOISE_LINE.search("java.io.IOException: No space left on device")


class TestSingleton:
    def test_get_retriever_caches_and_resets(self):
        from app.retriever import get_retriever

        reset_retriever()
        a = get_retriever()
        b = get_retriever()
        assert a is b, "同一进程内应复用同一个索引，避免每次请求重建"
        reset_retriever()
        assert get_retriever() is not a


# --------------------------------------------------------------------------- #
# 多后端：RRF 融合、Dense、Hybrid
#
# 这些测试全部用「假向量」跑，不需要 API Key、不联网、不花钱。
# 目的是把逻辑先验证到位 —— 填上 Key 之后只需跑一条命令确认。
# --------------------------------------------------------------------------- #
class TestRrfFusion:
    def test_single_ranking_preserves_order(self):
        fused = rrf_fuse([["a", "b", "c"]])
        assert list(fused) == ["a", "b", "c"]

    def test_scores_decrease_with_rank(self):
        fused = rrf_fuse([["a", "b", "c"]])
        assert fused["a"] > fused["b"] > fused["c"]

    def test_agreement_between_backends_wins(self):
        """两个后端都排第一的文档，融合后必须还是第一 —— 这正是 RRF 存在的意义。"""
        fused = rrf_fuse([["a", "b", "c"], ["a", "c", "b"]])
        assert max(fused, key=fused.get) == "a"

    def test_empty_ranking_is_ignored(self):
        """某个后端不可用时返回空列表，融合不该崩 —— 等价于不参与。"""
        fused = rrf_fuse([["a", "b"], []])
        assert set(fused) == {"a", "b"}

    def test_no_rankings(self):
        assert rrf_fuse([]) == {}


@pytest.fixture(scope="module")
def dense_retriever() -> DenseRetriever:
    """用假向量构造 —— 不需要 API Key，也不产生任何费用。"""
    cases = load_cases(KB_PATH)
    return DenseRetriever(cases, [fake_vector(c["log"]) for c in cases])


@pytest.fixture
def fake_embedding(monkeypatch):
    """把向量接口换成假实现。"""
    from app import embedding as emb

    monkeypatch.setenv("EMBEDDING_API_KEY", "test-key")
    monkeypatch.setattr(emb, "embed_one", lambda text: fake_vector(text))


class TestDenseRetriever:
    def test_size_matches_knowledge_base(self, dense_retriever):
        assert dense_retriever.size() >= 50

    def test_mismatched_length_raises(self):
        cases = load_cases(KB_PATH)[:3]
        with pytest.raises(ValueError):
            DenseRetriever(cases, [[1.0, 0.0]])

    def test_results_sorted_by_similarity(self, dense_retriever, fake_embedding):
        hits = dense_retriever.search("java.io.IOException: No space left on device", k=5)
        assert hits
        scores = [h.score for h in hits]
        assert scores == sorted(scores, reverse=True)

    def test_empty_when_not_configured(self, dense_retriever, monkeypatch):
        monkeypatch.delenv("EMBEDDING_API_KEY", raising=False)
        assert dense_retriever.search("任意查询") == []

    def test_service_error_returns_empty_instead_of_raising(self, dense_retriever, monkeypatch):
        """向量服务抖动时返回空，让上层回退到词法结果，而不是把整条诊断链路带崩。"""
        from app import embedding as emb

        monkeypatch.setenv("EMBEDDING_API_KEY", "test-key")

        def boom(_):
            raise RuntimeError("接口 500")

        monkeypatch.setattr(emb, "embed_one", boom)
        assert dense_retriever.search("任意查询") == []

    def test_exclude_is_respected(self, dense_retriever, fake_embedding):
        first = dense_retriever.search("disk quota exceeded", k=3)
        assert first
        again = dense_retriever.search("disk quota exceeded", k=3, exclude={first[0].id})
        assert all(h.id != first[0].id for h in again)


class TestHybridRetriever:
    @pytest.fixture
    def hybrid(self, dense_retriever, fake_embedding) -> HybridRetriever:
        return HybridRetriever(BM25Retriever(load_cases(KB_PATH)), dense_retriever)

    def test_returns_k_results(self, hybrid):
        assert len(hybrid.search("java.io.IOException: No space left on device", k=3)) == 3

    def test_results_come_from_knowledge_base(self, hybrid):
        valid = {c["id"] for c in hybrid.cases}
        hits = hybrid.search("redis connection refused", k=5)
        assert hits and all(h.case["id"] in valid for h in hits)

    def test_finds_disk_case(self, hybrid):
        """融合之后仍要能找到同类案例 —— 别为了融合把正确结果挤掉。"""
        hits = hybrid.search(
            "java.io.IOException: No space left on device\n"
            "\tat com.example.metrics.MetricsWriter.flush(MetricsWriter.java:88)",
            k=5,
        )
        assert hits, "融合后不该返回空"
        assert any(h.fault_group == "disk" for h in hits), (
            f"未召回 disk 类，实际召回: {[(h.id, h.fault_group) for h in hits]}"
        )

    def test_degrades_to_sparse_when_dense_missing(self, monkeypatch):
        """Dense 侧不可用时，Hybrid 应退化为纯 BM25 的结果，而不是返回空。"""
        monkeypatch.delenv("EMBEDDING_API_KEY", raising=False)
        sparse = BM25Retriever(load_cases(KB_PATH))
        hybrid = HybridRetriever(sparse, None)
        query = "java.io.IOException: No space left on device"
        assert [h.id for h in hybrid.search(query, k=3)] == [h.id for h in sparse.search(query, k=3)]


class TestBackendSelection:
    def test_default_backend_is_bm25(self, monkeypatch):
        from app.retriever import current_backend

        monkeypatch.delenv("LOGWISE_RETRIEVER_BACKEND", raising=False)
        assert current_backend() == "bm25"

    def test_unknown_backend_falls_back_to_default(self, monkeypatch):
        from app.retriever import current_backend

        monkeypatch.setenv("LOGWISE_RETRIEVER_BACKEND", "banana")
        assert current_backend() == "bm25"

    def test_make_retriever_bm25(self):
        assert isinstance(make_retriever("bm25"), BM25Retriever)

    def test_make_retriever_dense_without_index_raises(self, tmp_path):
        """严格模式：索引没建就明确报错。

        脚本与评测走这条路，是为了避免「以为在测 dense，其实跑的是 bm25」
        这种静默错误 —— 那会让整轮对比数据失去意义。
        """
        with pytest.raises(IndexNotBuiltError):
            make_retriever("dense", index_path=tmp_path / "nope.json")

    def test_get_retriever_degrades_instead_of_crashing(self, tmp_path, monkeypatch):
        """服务端必须降级：向量没配好不该让整个诊断服务起不来。"""
        from app import retriever as ret

        monkeypatch.setenv("LOGWISE_RETRIEVER_BACKEND", "dense")
        monkeypatch.setattr(ret, "INDEX_PATH", tmp_path / "nope.json")
        reset_retriever()
        try:
            assert isinstance(ret.get_retriever(), BM25Retriever)
            info = ret.describe_backend()
            assert info["degraded"] is True
            assert info["requested"] == "dense" and info["actual"] == "bm25"
            assert info["reason"], "降级原因要能报出来，否则用户不知道走的是哪条路"
        finally:
            reset_retriever()
