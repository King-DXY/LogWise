"""向量化客户端测试 —— 全程不碰真实接口。

用「假向量 + 假客户端」验证链路，理由很实际：
真实接口要花钱、要联网、要你的 Key，而这三样都不该成为"代码是否正确"的前提。
填 Key 之前先把逻辑验证到位，填上之后只需跑一条命令。
"""
from __future__ import annotations

from types import SimpleNamespace

import pytest

from app import embedding
from tests._helpers import fake_vector

__all__ = ["fake_vector"]  # 供其他测试模块复用


class FakeEmbeddingsAPI:
    """替身：记录调用次数与文本，返回确定性向量。"""

    def __init__(self) -> None:
        self.calls = 0
        self.texts: list[str] = []

    def create(self, model: str, input: list[str], **kwargs):  # noqa: A002
        self.calls += 1
        self.texts.extend(input)
        return SimpleNamespace(
            data=[
                SimpleNamespace(index=i, embedding=fake_vector(t))
                for i, t in enumerate(input)
            ]
        )


@pytest.fixture(autouse=True)
def isolated_cache(tmp_path, monkeypatch):
    """把缓存指到临时目录，避免测试污染真实的 data/index/。"""
    monkeypatch.setattr(embedding, "CACHE_PATH", tmp_path / "embedding_cache.json")
    monkeypatch.setattr(embedding, "INDEX_DIR", tmp_path)
    embedding.reset_cache()
    yield
    embedding.reset_cache()


@pytest.fixture
def fake_api(monkeypatch):
    api = FakeEmbeddingsAPI()
    monkeypatch.setattr(embedding, "get_client", lambda: SimpleNamespace(embeddings=api))
    monkeypatch.setenv("EMBEDDING_API_KEY", "test-key")
    monkeypatch.setenv("EMBEDDING_MODEL", "embedding-3")
    monkeypatch.delenv("EMBEDDING_DIMENSIONS", raising=False)
    return api


class TestConfig:
    def test_not_configured_without_key(self, monkeypatch):
        monkeypatch.delenv("EMBEDDING_API_KEY", raising=False)
        assert embedding.embedding_configured() is False

    def test_placeholder_key_counts_as_unconfigured(self, monkeypatch):
        """填了占位符等于没填 —— 不能因为有个非空字符串就以为配好了。"""
        monkeypatch.setenv("EMBEDDING_API_KEY", "在这里填你的智谱Key")
        assert embedding.embedding_configured() is False

    def test_configured_with_real_looking_key(self, monkeypatch):
        monkeypatch.setenv("EMBEDDING_API_KEY", "abc123")
        assert embedding.embedding_configured() is True

    def test_dimensions_empty_means_default(self, monkeypatch):
        monkeypatch.setenv("EMBEDDING_DIMENSIONS", "")
        assert embedding.embedding_dimensions() is None

    def test_dimensions_parsed_and_bounded(self, monkeypatch):
        monkeypatch.setenv("EMBEDDING_DIMENSIONS", "1024")
        assert embedding.embedding_dimensions() == 1024
        monkeypatch.setenv("EMBEDDING_DIMENSIONS", "abc")
        assert embedding.embedding_dimensions() is None
        monkeypatch.setenv("EMBEDDING_DIMENSIONS", "99999")
        assert embedding.embedding_dimensions() is None


class TestEmbed:
    def test_unconfigured_raises_clear_error(self, monkeypatch):
        monkeypatch.delenv("EMBEDDING_API_KEY", raising=False)
        with pytest.raises(embedding.EmbeddingUnavailable) as exc:
            embedding.embed(["任意文本"])
        assert "EMBEDDING_API_KEY" in str(exc.value)

    def test_returns_one_vector_per_text(self, fake_api):
        vectors = embedding.embed(["第一条", "第二条", "第三条"])
        assert len(vectors) == 3
        assert all(isinstance(v, list) and v for v in vectors)

    def test_empty_input_short_circuits(self, fake_api):
        assert embedding.embed([]) == []
        assert fake_api.calls == 0, "空输入不该调用接口"

    def test_cache_prevents_duplicate_calls(self, fake_api):
        embedding.embed(["同一条文本"])
        embedding.embed(["同一条文本"])
        assert fake_api.calls == 1, "第二次应命中缓存，不重复花钱"

    def test_cache_survives_reset(self, fake_api):
        embedding.embed(["持久化的文本"])
        embedding.reset_cache()  # 模拟进程重启（内存缓存清空）
        embedding.embed(["持久化的文本"])
        assert fake_api.calls == 1, "磁盘缓存应让重启后也不重复调用"

    def test_batching_splits_large_input(self, fake_api, monkeypatch):
        monkeypatch.setattr(embedding, "BATCH_SIZE", 4)
        embedding.embed([f"文本 {i}" for i in range(10)])
        assert fake_api.calls == 3, "10 条、每批 4 条 → 应为 3 次请求"

    def test_cache_key_changes_with_model(self, fake_api, monkeypatch):
        embedding.embed(["同一段文本"])
        monkeypatch.setenv("EMBEDDING_MODEL", "另一个模型")
        embedding.embed(["同一段文本"])
        assert fake_api.calls == 2, "换模型后旧向量必须失效，不能拿来混用"

    def test_embed_one(self, fake_api):
        assert len(embedding.embed_one("单条")) > 0

    def test_cache_stats(self, fake_api):
        embedding.embed(["a", "b"])
        stats = embedding.cache_stats()
        assert stats["cached_vectors"] == 2


class TestCosine:
    def test_identical_vectors_score_one(self):
        v = [0.1, 0.2, 0.3]
        assert embedding.cosine(v, v) == pytest.approx(1.0)

    def test_orthogonal_vectors_score_zero(self):
        assert embedding.cosine([1.0, 0.0], [0.0, 1.0]) == pytest.approx(0.0)

    def test_zero_vector_is_safe(self):
        assert embedding.cosine([0.0, 0.0], [1.0, 1.0]) == 0.0

    def test_similar_text_scores_higher(self):
        """假向量也要保持基本性质，否则测不出检索是否有效。"""
        a = fake_vector("redis connection refused")
        b = fake_vector("redis connection refused timeout")
        c = fake_vector("disk quota exceeded")
        assert embedding.cosine(a, b) > embedding.cosine(a, c)
