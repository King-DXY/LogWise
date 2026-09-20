"""构建向量索引：把知识库里每条日志算成向量，存到 data/index/embeddings.json。

用法
----
    python build_index.py             # 增量构建（已有缓存的条目会跳过）
    python build_index.py --force     # 忽略缓存，全量重建
    python build_index.py --check     # 只看状态，不调用接口

为什么做成独立脚本而不是塞进服务启动
------------------------------------
建索引要花钱（虽然很少），而且需要联网。放进启动流程会有两个问题：
1. 每次重启都调一遍接口，哪怕知识库一个字没变
2. 没配 Key 或网络不通时，整个服务起不来

所以做成显式的一次性动作，并在结束时把**实际花了多少钱**报出来。
底层的去重做得比较细：``app.embedding`` 按 ``(模型, 文本)`` 做落盘缓存，
所以重复运行不会重复计费，只有新增或改动过的案例才会真正调接口。
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

if __name__ == "__main__":
    # ⚠️ 必须在导入任何项目模块**之前**切到 .venv。
    # 双击运行时用的是系统全局 Python，依赖只装在 .venv 里 ——
    # 放到文件末尾调用就太晚了，import 阶段就会因找不到依赖而崩。
    from app.launcher import reenter_venv

    reenter_venv()

from app import embedding  # noqa: E402
from app.metrics import estimate_tokens_from_chars  # noqa: E402
from app.retriever import DEFAULT_KB_PATH, INDEX_PATH, load_cases  # noqa: E402

# 智谱 embedding-3 的官方单价（元/百万 token）。
# 会调价，所以做成可配置并在输出里回显 —— 你一眼能发现配置过时了。
DEFAULT_PRICE = 0.5


def price_per_million() -> float:
    raw = os.getenv("EMBEDDING_PRICE_PER_MILLION", "").strip()
    try:
        return float(raw) if raw else DEFAULT_PRICE
    except ValueError:
        return DEFAULT_PRICE


def load_index() -> dict:
    if not INDEX_PATH.exists():
        return {}
    try:
        return json.loads(INDEX_PATH.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {}


def show_status(cases: list[dict]) -> None:
    idx = load_index()
    stored: dict = idx.get("vectors", {})
    have = [c["id"] for c in cases if c.get("id") in stored]
    missing = [c["id"] for c in cases if c.get("id") not in stored]

    print("=" * 74)
    print("向量索引状态")
    print("=" * 74)
    print(f"  知识库      : {DEFAULT_KB_PATH}（{len(cases)} 条）")
    print(f"  索引文件    : {INDEX_PATH}")
    print(f"  索引状态    : {'存在' if idx else '不存在'}")
    if idx:
        print(f"  建库模型    : {idx.get('model', '(未知)')}   维度 {idx.get('dimensions', '?')}")
        print(f"  构建时间    : {idx.get('built_at', '(未知)')}")
    print(f"  已有向量    : {len(have)} 条")
    print(f"  缺少向量    : {len(missing)} 条" + (f"  例如 {missing[:3]}" if missing else ""))
    print()
    print(f"  向量缓存    : {embedding.cache_stats()['cached_vectors']} 条  ({embedding.CACHE_PATH})")
    print(f"  接口配置    : {'已配置' if embedding.embedding_configured() else '未配置 EMBEDDING_API_KEY'}")
    print(f"  模型 / 地址 : {embedding.embedding_model()}  @  {embedding.embedding_base_url()}")
    print(f"  单价回显    : {price_per_million()} 元 / 百万 token")
    if missing:
        print()
        print("  → 运行 python build_index.py 补齐")


def main() -> int:
    ap = argparse.ArgumentParser(description="构建故障案例的向量索引")
    ap.add_argument("--force", action="store_true", help="忽略缓存全量重建")
    ap.add_argument("--check", action="store_true", help="只看状态，不调用接口")
    ap.add_argument("--limit", type=int, default=0, help="只处理前 N 条（用于小规模试跑）")
    args = ap.parse_args()

    cases = load_cases()
    if args.limit:
        cases = cases[: args.limit]

    if args.check:
        show_status(cases)
        return 0

    if not embedding.embedding_configured():
        print("=" * 74)
        print("无法建索引：未配置向量化服务")
        print("=" * 74)
        print("  请在 .env 里填入以下三项后重试：")
        print()
        print("    EMBEDDING_BASE_URL=https://open.bigmodel.cn/api/paas/v4")
        print("    EMBEDDING_API_KEY=你的智谱Key")
        print("    EMBEDDING_MODEL=embedding-3")
        print()
        print("  （不配也不影响诊断功能，只是 Dense / Hybrid 检索不可用）")
        return 1

    if args.force and embedding.CACHE_PATH.exists():
        embedding.CACHE_PATH.unlink()
        embedding.reset_cache()
        print("  已清除向量缓存，将全量重建")

    print("=" * 74)
    print("构建向量索引")
    print("=" * 74)
    print(f"  知识库   : {len(cases)} 条")
    print(f"  模型     : {embedding.embedding_model()}")
    print(f"  单价回显 : {price_per_million()} 元 / 百万 token")
    print()

    # 先算一下"需要真正调用接口"的有几条，避免缓存命中时误报花费
    cache = embedding._load_cache()  # noqa: SLF001  同包内读取状态，用于报账
    pending = [c for c in cases if embedding._cache_key(c["log"]) not in cache]  # noqa: SLF001
    # 注意传的是**字符数**不是文本 —— 这个函数名里的 from_chars 就是这个意思
    est_tokens = sum(estimate_tokens_from_chars(len(c["log"])) for c in pending)
    est_cost = est_tokens / 1_000_000 * price_per_million()

    print(f"  缓存命中 : {len(cases) - len(pending)} 条（不调用接口、不计费）")
    print(f"  待处理   : {len(pending)} 条  预计 {est_tokens:,} token ≈ {est_cost:.5f} 元")
    print()

    if not pending:
        print("  没有需要处理的条目，索引已是最新。")
        vectors = {c["id"]: embedding.embed_one(c["log"]) for c in cases}
    else:
        t0 = time.perf_counter()
        texts = [c["log"] for c in cases]
        try:
            all_vectors = embedding.embed(texts)
        except Exception as exc:
            print(f"  ❌ 调用失败: {type(exc).__name__}: {exc}")
            return 1
        elapsed = time.perf_counter() - t0
        print(f"  完成，耗时 {elapsed:.1f}s")
        vectors = {c["id"]: v for c, v in zip(cases, all_vectors)}

    dims = len(next(iter(vectors.values()))) if vectors else 0
    INDEX_PATH.parent.mkdir(parents=True, exist_ok=True)
    INDEX_PATH.write_text(
        json.dumps(
            {
                "model": embedding.embedding_model(),
                "dimensions": dims,
                "built_at": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
                "knowledge_base": str(DEFAULT_KB_PATH.relative_to(ROOT)).replace("\\", "/"),
                "count": len(vectors),
                "vectors": {k: [round(x, 6) for x in v] for k, v in vectors.items()},
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    size_mb = INDEX_PATH.stat().st_size / 1024 / 1024
    print()
    print("=" * 74)
    print("索引已写入")
    print("=" * 74)
    print(f"  文件     : {INDEX_PATH}")
    print(f"  条数     : {len(vectors)}   维度 {dims}   体积 {size_mb:.2f} MB")
    print(f"  本次花费 : 约 {est_cost:.5f} 元（{len(pending)} 条实际调用了接口）")
    print()
    print("  接下来可以跑对比：")
    print("    python eval_retrieval.py --backend bm25")
    print("    python eval_retrieval.py --backend dense")
    print("    python eval_retrieval.py --backend hybrid")
    return 0


if __name__ == "__main__":
    from app.launcher import pause_if_console

    exit_code = main()
    pause_if_console()
    raise SystemExit(exit_code)
