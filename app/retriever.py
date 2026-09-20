"""可插拔的故障案例检索层（RAG 的检索端）。

为什么默认用 BM25 而不是向量检索
--------------------------------
不是"退而求其次"，而是基于这个场景的判断：

1. **日志检索靠的是精确 token，不是语义相似**
   `HikariPool`、`No space left on device`、`OOMKilled`、`code=9F27`
   这类词要么全中要么全不中。BM25 天生擅长这种匹配。
2. **向量的"相近"在诊断场景下可能有害**
   向量会把「连接被拒」与「服务超时」放得很近（语义确实像），
   但这两者的根因完全不同，误召回比不召回更糟。
3. **DeepSeek 没有 embedding 接口**（实测 `/embeddings` 返回 404），
   要上向量就得额外下载本地模型或接入第三方服务。
   在拿到「BM25 不够用」的实测数据之前，不值得引入这个依赖。

因此这里把后端抽象成 `Retriever`，`BM25Retriever` 是默认实现。
等评测数据说明词法检索确实不够时，再加 `DenseRetriever` 并做 A/B —— 接口已经留好。

针对日志场景的分词
------------------
通用分词器在日志上表现很差。`FileNotFoundException` 必须能匹配到
`FileNotFound`、`not found` 等写法，所以要做 CamelCase / 下划线 / 点号拆词。
中文部分按 2-gram 切分（"磁盘写满" → 磁盘/盘写/写满）。
"""
from __future__ import annotations

import json
import math
import os
import re
from abc import ABC, abstractmethod
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_KB_PATH = ROOT / "data" / "fault_cases.jsonl"

# CamelCase / 下划线 / 点号 / 连字符 的拆分点
_SPLIT = re.compile(r"(?<=[a-z0-9])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])|[_\-.]")
_IDENT = re.compile(r"[A-Za-z][A-Za-z0-9_]*")
_CJK = re.compile(r"[\u4e00-\u9fa5]+")
_STACK_FRAME = re.compile(r"^\s*at\s", re.M)

# BM25 参数：k1 控制词频饱和，b 控制文档长度归一化（0.75 是通用默认值）
K1 = 1.5
B = 0.75

# 检索后端
BACKEND_BM25 = "bm25"
BACKEND_DENSE = "dense"
BACKEND_HYBRID = "hybrid"
BACKENDS = (BACKEND_BM25, BACKEND_DENSE, BACKEND_HYBRID)

# 默认仍是 bm25 —— 接入新后端默认不改变现有行为，等实测数据出来再决定要不要换。
DEFAULT_BACKEND = BACKEND_BM25

# RRF 的平滑常数。60 是检索工程里的通用默认值：
# 它决定"排名靠前"能带来多大优势 —— 越小越激进。
RRF_K = 60

# 融合前每个后端各取多少条候选。取得比最终 k 大，避免某个后端的好结果被早早截掉。
CANDIDATE_DEPTH = 20

INDEX_PATH = ROOT / "data" / "index" / "embeddings.json"

# 行的权重：异常消息行最能说明问题，日志头信息量最低
W_EXCEPTION = 3.0
W_BUSINESS = 2.0
W_DEFAULT = 1.0

# 无信息量的日志行 —— 检索时在**查询与文档两端**都要跳过。
#
# 为什么必须过滤：这类行不含任何区分度，但数量常常压倒真正的异常行。
# 实测长日志噪音用例里有 10 条"健康检查通过"，它们把一条含相同噪音的案例
# 直接顶到检索第一名（分数 760，而真正该命中的案例排第 4）。
# 过滤掉之后 Recall@3 从 81.2% 提升到 96.9%。
_NOISE_LINE = re.compile(
    r"健康检查(通过|正常|成功)"
    r"|指标上报成功"
    r"|心跳(正常|成功)"
    r"|HealthChecker\s*[-:]\s*健康检查"
    r"|Reporter\s*[-:]\s*指标上报",
)


def tokenize(text: str) -> list[str]:
    """日志专用分词：拆 CamelCase、保留整体标识符、中文 2-gram。"""
    out: list[str] = []

    for seg in _CJK.findall(text):
        if len(seg) == 1:
            out.append(seg)
        else:
            out.extend(seg[i:i + 2] for i in range(len(seg) - 1))

    for raw in _IDENT.findall(text):
        for part in _SPLIT.split(raw):
            p = part.lower()
            if len(p) >= 2:
                out.append(p)
        # 同时保留整体，让 `filenotfoundexception` 也能精确命中
        low = raw.lower()
        if low not in {x.lower() for x in _SPLIT.split(raw)}:
            out.append(low)

    return out


def _line_weight(line: str) -> float:
    if "Exception" in line or "Error" in line or "错误" in line:
        return W_EXCEPTION
    if "com.example." in line or re.search(r"[\u4e00-\u9fa5]", line):
        return W_BUSINESS
    return W_DEFAULT


def weighted_tokens(text: str) -> list[str]:
    """按行的信息量做加权采样 —— 等价于把重要的行"复制"若干份。

    无信息量的行（健康检查、指标上报等）直接跳过，原因见 `_NOISE_LINE`。
    """
    out: list[str] = []
    for line in text.split("\n"):
        if _NOISE_LINE.search(line):
            continue
        w = _line_weight(line)
        out.extend(tokenize(line) * int(w))
    return out


@dataclass
class RetrievedCase:
    """一条检索结果。"""

    score: float
    case: dict

    @property
    def id(self) -> str:
        return self.case["id"]

    @property
    def fault_group(self) -> str:
        return self.case["fault_group"]

    def as_few_shot(self) -> tuple[str, str]:
        """转成可直接注入 Prompt 的 (user, assistant) 内容。"""
        return self.case["log"], json.dumps(self.case["diagnosis"], ensure_ascii=False)


class Retriever(ABC):
    """检索接口。换后端只需实现这个类，调用方不用改。"""

    @abstractmethod
    def search(self, query: str, k: int = 3, exclude: set[str] | None = None) -> list[RetrievedCase]:
        ...

    @abstractmethod
    def size(self) -> int:
        ...


class BM25Retriever(Retriever):
    """Okapi BM25，纯 Python 实现，零依赖。

    文档 = 一条知识库案例的日志正文。
    查询 = 本次待诊断的日志（通常是上下文聚合后的多行文本）。
    """

    def __init__(self, cases: list[dict]):
        self._cases = cases
        self._docs: list[list[str]] = [weighted_tokens(c["log"]) for c in cases]
        self._tf: list[Counter[str]] = [Counter(d) for d in self._docs]
        self._len: list[int] = [len(d) for d in self._docs]
        self._avgdl = (sum(self._len) / len(self._len)) if self._len else 0.0

        df: Counter[str] = Counter()
        for tf in self._tf:
            df.update(tf.keys())
        n = len(self._docs)
        # 加 1 防止 df=0 时取对数出错；BM25+ 形式的 IDF 恒为正
        self._idf: dict[str, float] = {
            t: math.log(1 + (n - d + 0.5) / (d + 0.5)) for t, d in df.items()
        }

    @classmethod
    def from_jsonl(cls, path: Path | str | None = None) -> "BM25Retriever":
        return cls(load_cases(path))

    @property
    def cases(self) -> list[dict]:
        return self._cases

    def size(self) -> int:
        return len(self._cases)

    def search(self, query: str, k: int = 3, exclude: set[str] | None = None) -> list[RetrievedCase]:
        exclude = exclude or set()
        # ⚠️ query 侧**必须二值化**（用 set 而不是 Counter）。
        # 踩过的坑：日志里同一句"健康检查通过"重复十几次时，
        # 若按出现次数线性放大权重，噪音会直接把某条含相同噪音的案例顶到第一，
        # 实测噪声用例的分数被推到 760，而真正的异常行被淹没。
        # 检索只关心"出现了哪些错误"，不关心同一个词重复了几次 ——
        # 词频的作用体现在文档侧（tf），不体现在查询侧。
        q_terms = set(weighted_tokens(query))
        if not q_terms:
            return []

        scored: list[tuple[float, int]] = []
        for i, tf in enumerate(self._tf):
            if self._cases[i]["id"] in exclude:
                continue
            dl = self._len[i] or 1
            score = 0.0
            for term in q_terms:
                f = tf.get(term)
                if not f:
                    continue
                idf = self._idf.get(term, 0.0)
                denom = f + K1 * (1 - B + B * dl / (self._avgdl or 1))
                score += idf * (f * (K1 + 1)) / denom
            if score > 0:
                scored.append((score, i))

        scored.sort(key=lambda x: (-x[0], self._cases[x[1]]["id"]))
        return [RetrievedCase(score=round(s, 6), case=self._cases[i]) for s, i in scored[:k]]


class IndexNotBuiltError(RuntimeError):
    """向量索引不存在或与知识库 / 当前配置不匹配 —— 需要先跑 build_index.py。"""


def _configured_embedding_model() -> str:
    """当前配置的 embedding 模型名；取不到就返回空串（表示跳过该校验）。

    用延迟导入而不是模块级导入：没用到向量检索时不必加载 embedding 模块
    （它会连带 import dotenv）。
    """
    try:
        from app import embedding
    except ImportError:
        return ""
    return embedding.embedding_model()


class DenseRetriever(Retriever):
    """余弦相似度检索，向量来自 OpenAI 兼容的 embedding 接口（如智谱）。

    索引由 ``build_index.py`` 预先算好并落到 ``data/index/embeddings.json``，
    检索时**只对 query 调一次接口** —— 知识库那侧的向量早已算过。

    注意分数尺度：这里是余弦相似度（约 -1~1），而 BM25 是无界的正值。
    两者**不能直接加权合并**，这正是 Hybrid 要用 RRF（只看排名）的原因。
    """

    def __init__(self, cases: list[dict], vectors: list[list[float]]):
        if len(cases) != len(vectors):
            raise ValueError(f"案例数与向量数不匹配: {len(cases)} vs {len(vectors)}")
        dims = {len(v) for v in vectors}
        if len(dims) > 1:
            raise ValueError(f"向量维度不一致: {sorted(dims)}，索引可能损坏，请重建")
        self._cases = cases
        self._vectors = vectors
        self._dim = dims.pop() if dims else 0
        self._norms = [math.sqrt(sum(x * x for x in v)) or 1.0 for v in vectors]

    @property
    def cases(self) -> list[dict]:
        return self._cases

    @classmethod
    def from_index(
        cls,
        cases: list[dict] | None = None,
        path: Path | str | None = None,
        kb_path: Path | str | None = None,
    ) -> "DenseRetriever":
        # ⚠️ 默认值写 None 而不是直接引用 INDEX_PATH ——
        # Python 的函数默认参数在**定义时**求值，写成 `path=INDEX_PATH` 之后，
        # 测试里 monkeypatch 模块属性就再也不生效了（拿到的是定义时那个对象）。
        # 这个坑是被 test_get_retriever_degrades_instead_of_crashing 抓出来的。
        index_path = Path(INDEX_PATH if path is None else path)
        if not index_path.exists():
            raise IndexNotBuiltError(
                f"向量索引不存在: {index_path}\n请先运行:  python build_index.py"
            )
        raw = json.loads(index_path.read_text(encoding="utf-8"))
        cases = load_cases(kb_path) if cases is None else cases

        stored: dict = raw.get("vectors", {})
        vectors = [stored.get(c["id"]) for c in cases]
        missing = [c["id"] for c, v in zip(cases, vectors) if not v]
        if missing:
            raise IndexNotBuiltError(
                f"索引缺少 {len(missing)} 条案例的向量（例如 {missing[:3]}）。"
                "知识库应该是改过了，请重新运行:  python build_index.py"
            )

        # ------------------------------------------------------------------ #
        # 校验索引与当前配置是否匹配
        #
        # 为什么必须在**加载阶段**拦下来：余弦相似度是 element-wise 的 zip 点积，
        # 两边长度不一致时 ```zip`` 会**静默截断**到较短的那个 —— 不报错，
        # 只是算出一个"看起来正常但实际错误"的分数。实测 4 维 query 对 8 维索引
        # 会正常返回结果，完全没有提示。
        #
        # 触发场景很现实：改了 .env 里的 EMBEDDING_MODEL 或 EMBEDDING_DIMENSIONS
        # 之后没重建索引。这里报错 → get_retriever() 降级回 BM25 并把原因写进
        # /stats，比悄悄给出错误检索结果好得多。
        # ------------------------------------------------------------------ #
        dims = {len(v) for v in vectors}
        if len(dims) != 1:
            raise IndexNotBuiltError(
                f"索引内向量维度不一致（{sorted(dims)}），文件可能损坏。"
                "请重新运行:  python build_index.py"
            )
        actual_dim = dims.pop()

        declared_dim = raw.get("dimensions")
        if isinstance(declared_dim, int) and declared_dim != actual_dim:
            raise IndexNotBuiltError(
                f"索引声明的维度是 {declared_dim}，实际是 {actual_dim}，文件已损坏。"
                "请重新运行:  python build_index.py"
            )

        stored_model = raw.get("model")
        current_model = _configured_embedding_model()
        if stored_model and current_model and stored_model != current_model:
            raise IndexNotBuiltError(
                f"索引是用 {stored_model} 建的，当前配置的模型是 {current_model}。"
                "不同模型的向量空间不可比，换个模型必须重建索引：\n"
                "    python build_index.py --force"
            )

        return cls(cases, vectors)

    def size(self) -> int:
        return len(self._cases)

    def search(self, query: str, k: int = 3, exclude: set[str] | None = None) -> list[RetrievedCase]:
        from app import embedding  # 延迟导入：没用到向量时不必加载

        if not embedding.embedding_configured():
            return []
        try:
            qv = embedding.embed_one(query)
        except Exception:
            # 向量服务抖动不该让整条诊断链路失败：返回空，让上层回退到词法结果
            return []

        # 最后一道防线：query 与索引维度不一致时，下面的 zip 点积会静默截断，
        # 算出错误的相似度。加载阶段已校验过配置，能走到这里说明服务端返回的
        # 维度与建索引时不同 —— 宁可不返回结果（上层会退化成只用词法结果），
        # 也不要返回错的。
        if self._dim and len(qv) != self._dim:
            return []

        qn = math.sqrt(sum(x * x for x in qv)) or 1.0
        exclude = exclude or set()
        scored: list[tuple[float, int]] = []
        for i, vec in enumerate(self._vectors):
            if self._cases[i]["id"] in exclude:
                continue
            dot = 0.0
            for a, b in zip(qv, vec):
                dot += a * b
            scored.append((dot / (qn * self._norms[i]), i))

        scored.sort(key=lambda x: (-x[0], self._cases[x[1]]["id"]))
        return [RetrievedCase(score=round(s, 6), case=self._cases[i]) for s, i in scored[:k]]


def rrf_fuse(rankings: list[list[str]], k: int = RRF_K) -> dict[str, float]:
    """Reciprocal Rank Fusion：按**排名**融合多个后端的结果。

    ``score(d) = Σ_i 1 / (k + rank_i(d))``

    为什么不用分数加权：BM25 的分数是 0 到 +∞ 的无界值，余弦相似度在 -1~1 之间，
    两者尺度完全不同 —— 直接相加等于让 BM25 单方面主导融合结果。
    RRF 只看排名，天然免疫尺度问题，这是它成为检索工程通用做法的原因。
    """
    fused: dict[str, float] = {}
    for ranking in rankings:
        for rank, doc_id in enumerate(ranking, start=1):
            fused[doc_id] = fused.get(doc_id, 0.0) + 1.0 / (k + rank)
    return fused


class HybridRetriever(Retriever):
    """BM25 与 Dense 的 RRF 融合。

    两个后端各取 ``CANDIDATE_DEPTH`` 条候选（比最终 k 大，避免某侧的好结果被早早截掉），
    融合后返回前 k 条。某一侧不可用时自动退化为另一侧 —— 不做特殊处理，
    空排名参与融合等价于不参与。
    """

    def __init__(self, sparse: Retriever, dense: Retriever | None):
        self._sparse = sparse
        self._dense = dense
        self._cases = list(getattr(sparse, "cases", []) or [])
        self._by_id = {c["id"]: c for c in self._cases}

    @property
    def cases(self) -> list[dict]:
        return self._cases

    def size(self) -> int:
        return self._sparse.size()

    def search(self, query: str, k: int = 3, exclude: set[str] | None = None) -> list[RetrievedCase]:
        exclude = exclude or set()
        rankings: list[list[str]] = [
            [h.id for h in self._sparse.search(query, k=CANDIDATE_DEPTH, exclude=exclude)]
        ]
        if self._dense is not None:
            rankings.append(
                [h.id for h in self._dense.search(query, k=CANDIDATE_DEPTH, exclude=exclude)]
            )

        fused = rrf_fuse(rankings)
        ordered = sorted(fused.items(), key=lambda kv: (-kv[1], kv[0]))
        # 先过滤再截断。反过来的话（先 ordered[:k] 再过滤），遇到不在知识库里的
        # doc_id 就会白白占掉名额，最终返回的条数少于 k。
        picked: list[RetrievedCase] = []
        for doc_id, score in ordered:
            case = self._by_id.get(doc_id)
            if case is None:
                continue
            picked.append(RetrievedCase(score=round(score, 6), case=case))
            if len(picked) >= k:
                break
        return picked


# --------------------------------------------------------------------------- #
# 工厂与单例
# --------------------------------------------------------------------------- #
def load_cases(path: Path | str | None = None) -> list[dict]:
    """读取知识库。默认值写 None 的原因同 ``make_retriever``：便于测试替换路径。"""
    resolved = DEFAULT_KB_PATH if path is None else path
    lines = Path(resolved).read_text(encoding="utf-8").splitlines()
    return [json.loads(x) for x in lines if x.strip()]


def current_backend() -> str:
    """当前配置的检索后端（环境变量 LOGWISE_RETRIEVER_BACKEND）。"""
    raw = os.getenv("LOGWISE_RETRIEVER_BACKEND", DEFAULT_BACKEND).strip().lower()
    return raw if raw in BACKENDS else DEFAULT_BACKEND


def make_retriever(
    backend: str | None = None,
    kb_path: Path | str | None = None,
    index_path: Path | str | None = None,
) -> Retriever:
    """按后端名构造检索器。**严格模式**：要求的后端不可用就抛异常。

    脚本与评测请用这个 —— 免得出现「以为在测 hybrid，其实跑的是 bm25」。
    服务端用 ``get_retriever()``，它会降级。

    路径参数的默认值是 ``None`` 而不是直接写模块常量，原因同 ``from_index``：
    默认参数在函数定义时求值，写死之后 monkeypatch 模块属性就不生效了。
    """
    backend = (backend or current_backend()).strip().lower()
    kb_path = DEFAULT_KB_PATH if kb_path is None else kb_path
    index_path = INDEX_PATH if index_path is None else index_path
    cases = load_cases(kb_path)
    if backend == BACKEND_BM25:
        return BM25Retriever(cases)
    if backend == BACKEND_DENSE:
        return DenseRetriever.from_index(cases=cases, path=index_path)
    if backend == BACKEND_HYBRID:
        return HybridRetriever(
            BM25Retriever(cases),
            DenseRetriever.from_index(cases=cases, path=index_path),
        )
    raise ValueError(f"未知检索后端: {backend}（可选 {list(BACKENDS)}）")


_retriever: Retriever | None = None
_retriever_backend: str | None = None
_retriever_path: str | None = None
_fallback_reason: str = ""
_kb_ok: bool = True


def get_retriever(path: Path | str | None = None) -> Retriever:
    """服务端用的单例。

    与 ``make_retriever`` 的区别：后端不可用时**降级回 BM25** 并记下原因
    （``describe_backend()`` 会报出来）。理由很实际 ——
    向量服务没配好或索引没建，不该让整个诊断服务起不来。
    """
    global _retriever, _retriever_backend, _retriever_path, _fallback_reason, _kb_ok
    kb_path = DEFAULT_KB_PATH if path is None else path
    cache_id = str(kb_path)
    backend = current_backend()
    # 知识库路径也要进缓存键 —— 否则测试里换个路径拿到的还是上一次那个实例
    if _retriever is not None and _retriever_backend == backend and _retriever_path == cache_id:
        return _retriever

    _fallback_reason = ""
    _kb_ok = True
    try:
        _retriever = make_retriever(backend, kb_path)
        _retriever_backend = backend
    except Exception as exc:
        _fallback_reason = f"{type(exc).__name__}: {exc}"
        try:
            _retriever = BM25Retriever.from_jsonl(kb_path)
            _retriever_backend = BACKEND_BM25
        except Exception as fallback_exc:
            # ------------------------------------------------------------------ #
            # 连词法检索也起不来（知识库文件缺失或损坏）。
            #
            # 为什么还要再兜一层：原来只有一层 except，而它内部又调 from_jsonl
            # 去读**同一个**读不到的文件 —— 于是异常直接穿出去。
            # 后果很严重：/stats 会调 get_retriever().size()，整个接口 500。
            # 实测就踩过：.dockerignore 排除了 data/，容器里没有知识库，
            # 于是容器内的 /stats 直接挂掉。
            #
            # 用一个空知识库兜底：检索返回空结果，诊断照常工作（RAG 本来就默认关闭），
            # 但服务不会因为一个可选功能的数据文件缺失而整体不可用。
            # ------------------------------------------------------------------ #
            _fallback_reason = (
                f"{type(exc).__name__}: {exc}"
                f"｜知识库不可用（{type(fallback_exc).__name__}: {fallback_exc}）"
            )
            _retriever = BM25Retriever([])
            _retriever_backend = BACKEND_BM25
            _kb_ok = False
    _retriever_path = cache_id
    return _retriever


def describe_backend() -> dict:
    """当前生效的检索后端，以及是否发生过降级。"""
    requested = current_backend()
    actual = _retriever_backend or requested
    return {
        "requested": requested,
        "actual": actual,
        "degraded": bool(_fallback_reason),
        "reason": _fallback_reason,
        "knowledge_base_size": _retriever.size() if _retriever else 0,
        # 单独给一个布尔：知识库文件读不到时，上面那层的降级也救不了，
        # 会退化成一个空知识库（检索返回空）。这种情况必须一眼看得出来。
        "knowledge_base_ok": _kb_ok,
    }


def reset_retriever() -> None:
    """用于测试：清掉单例，强制下次重新加载。"""
    global _retriever, _retriever_backend, _retriever_path, _fallback_reason, _kb_ok
    _retriever = None
    _retriever_backend = None
    _retriever_path = None
    _fallback_reason = ""
    _kb_ok = True
