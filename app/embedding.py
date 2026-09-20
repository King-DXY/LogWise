"""向量化客户端：对接 OpenAI 兼容格式的 embedding 接口。

为什么单独成模块
----------------
诊断用 DeepSeek，向量化得用另一家 —— 本项目实测 DeepSeek **不提供 embedding**
（`POST /embeddings` 返回 404，模型列表只有对话模型）。

所以这里做成**通用 OpenAI 兼容客户端**：智谱 / 阿里百炼 / 硅基流动 都是这个格式，
换厂商只需要改 `.env` 三行，代码不动。

三个设计决定
------------
1. **结果落盘缓存**。embedding 虽便宜（智谱 0.5 元/百万 token），
   但知识库没变时重启服务、重跑脚本都不该重复花钱。
   缓存键是 ``(模型, 文本)`` 的哈希，与配置绑定 —— 换模型会自动失效。
2. **未配置时优雅降级**。没填 Key 就明确报告"不可用"，
   而不是抛异常把整个服务带崩。上层先问 ``embedding_configured()`` 再决定用不用。
3. **批量 + 分批**。一次请求传多条减少往返，同时避免单请求过大被拒。
"""
from __future__ import annotations

import hashlib
import json
import os
import threading
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
INDEX_DIR = ROOT / "data" / "index"
CACHE_PATH = INDEX_DIR / "embedding_cache.json"

# ⚠️ 本模块自己加载 .env，不能依赖「别处已经加载过」。
# 踩过的坑：build_index.py 只导入 app.embedding / metrics / retriever，
# 不经过 app.diagnose —— 而 load_dotenv() 原本只写在 diagnose.py 里，
# 于是脚本读不到 EMBEDDING_API_KEY，报「未配置向量化服务」，
# 但同一台机器上诊断功能和手工测试都正常，很容易看懵。
# 谁消费配置谁负责加载，比依赖导入顺序可靠。
load_dotenv(ROOT / ".env")

# 单次请求最多提交多少条文本。太小则往返多，太大可能被服务端拒绝。
BATCH_SIZE = 32

DEFAULT_MODEL = "embedding-3"


class EmbeddingUnavailable(RuntimeError):
    """向量化服务未配置或不可用时抛出。"""


# --------------------------------------------------------------------------- #
# 配置
# --------------------------------------------------------------------------- #
def embedding_base_url() -> str:
    return os.getenv("EMBEDDING_BASE_URL", "https://open.bigmodel.cn/api/paas/v4").strip()


def embedding_model() -> str:
    return os.getenv("EMBEDDING_MODEL", DEFAULT_MODEL).strip() or DEFAULT_MODEL


def embedding_dimensions() -> int | None:
    """自定义向量维度；留空则用服务端默认值。

    智谱 embedding-3 支持 256~2048。维度越低存储越省、精度略降。
    默认不传该参数 —— 不同厂商对这个字段的支持程度不一致，
    留空最保险；需要时在 .env 里显式指定。
    """
    raw = os.getenv("EMBEDDING_DIMENSIONS", "").strip()
    if not raw:
        return None
    try:
        value = int(raw)
    except ValueError:
        return None
    return value if 1 <= value <= 8192 else None


def embedding_configured() -> bool:
    """是否已配置可用的向量化服务。未配置时 Dense 检索应被标记为不可用。"""
    key = os.getenv("EMBEDDING_API_KEY", "").strip()
    return bool(key) and key != "在这里填你的智谱Key"


# --------------------------------------------------------------------------- #
# 客户端
# --------------------------------------------------------------------------- #
_client = None
_client_lock = threading.Lock()


def get_client():
    if not embedding_configured():
        raise EmbeddingUnavailable(
            "未配置 EMBEDDING_API_KEY，向量化不可用。"
            "请在 .env 中填写后重试（诊断功能不受影响）。"
        )
    global _client
    with _client_lock:
        if _client is None:
            from openai import OpenAI  # 延迟导入，未用向量时不必加载

            _client = OpenAI(api_key=os.getenv("EMBEDDING_API_KEY"), base_url=embedding_base_url())
    return _client


# --------------------------------------------------------------------------- #
# 落盘缓存
# --------------------------------------------------------------------------- #
_cache: dict[str, list[float]] | None = None
_cache_lock = threading.Lock()
_cache_dirty = False


def _cache_key(text: str) -> str:
    """缓存键含模型名 —— 换模型后旧向量自动失效，不会拿错的向量去比对。"""
    payload = f"{embedding_model()}\x00{embedding_dimensions() or 'default'}\x00{text}"
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()


def _load_cache() -> dict[str, list[float]]:
    global _cache
    if _cache is not None:
        return _cache
    with _cache_lock:
        if _cache is None:
            if CACHE_PATH.exists():
                try:
                    raw = json.loads(CACHE_PATH.read_text(encoding="utf-8"))
                    _cache = raw.get("vectors", {}) if isinstance(raw, dict) else {}
                except (json.JSONDecodeError, OSError):
                    _cache = {}  # 缓存坏了不该让功能不可用，重建即可
            else:
                _cache = {}
    return _cache


def _save_cache() -> None:
    global _cache_dirty
    cache = _load_cache()
    with _cache_lock:
        INDEX_DIR.mkdir(parents=True, exist_ok=True)
        CACHE_PATH.write_text(
            json.dumps({"model": embedding_model(), "count": len(cache), "vectors": cache},
                       ensure_ascii=False),
            encoding="utf-8",
        )
        _cache_dirty = False


def cache_stats() -> dict:
    cache = _load_cache()
    return {
        "path": str(CACHE_PATH),
        "cached_vectors": len(cache),
        "model": embedding_model(),
    }


def reset_cache(memory_only: bool = True) -> None:
    """测试用：清掉内存缓存（默认不动磁盘文件）。"""
    global _cache, _cache_dirty
    with _cache_lock:
        _cache = None
        _cache_dirty = False


# --------------------------------------------------------------------------- #
# 对外接口
# --------------------------------------------------------------------------- #
def embed(texts: list[str]) -> list[list[float]]:
    """把一批文本转成向量。命中缓存的不会重复调用接口。

    未配置向量化服务时抛 ``EmbeddingUnavailable``。
    """
    if not texts:
        return []

    cache = _load_cache()
    keys = [_cache_key(t) for t in texts]

    missing_idx = [i for i, k in enumerate(keys) if k not in cache]
    if missing_idx:
        pending = [texts[i] for i in missing_idx]
        client = get_client()
        vectors: list[list[float]] = []
        for start in range(0, len(pending), BATCH_SIZE):
            batch = pending[start:start + BATCH_SIZE]
            kwargs = {"model": embedding_model(), "input": batch}
            dims = embedding_dimensions()
            if dims:
                kwargs["dimensions"] = dims
            resp = client.embeddings.create(**kwargs)
            # 服务端不保证返回顺序，按 index 排序更稳妥
            items = sorted(resp.data, key=lambda d: d.index)
            vectors.extend([list(d.embedding) for d in items])

        global _cache_dirty
        with _cache_lock:
            for i, vec in zip(missing_idx, vectors):
                cache[keys[i]] = vec
            _cache_dirty = True
        _save_cache()

    return [cache[k] for k in keys]


def embed_one(text: str) -> list[float]:
    return embed([text])[0]


def cosine(a: list[float], b: list[float]) -> float:
    """余弦相似度。纯 Python 实现 —— 53×1024 维只有 5 万次乘加，毫秒级，
    为这点计算量引入 numpy 不划算。"""
    dot = na = nb = 0.0
    for x, y in zip(a, b):
        dot += x * y
        na += x * x
        nb += y * y
    if na <= 0 or nb <= 0:
        return 0.0
    return dot / ((na ** 0.5) * (nb ** 0.5))
