"""检索质量评测：知识库里找不找得到同类故障。

为什么单独测检索
----------------
RAG 由两段组成 —— **检索** 与 **注入**。如果检索本身找错了案例，
Prompt 再讲究也救不回来。分开测才能知道问题出在哪一段。

而且检索评测**完全不需要调用大模型**：有没有召回同类案例，是纯本地可算的。
所以这一步能在花任何钱之前先把关 —— 检索不过关，就不必谈注入。

判据
----
对每条评测用例，看检索结果里有没有 `fault_group` 与它真因同类别的案例。
`fault_group` 是知识库上的人工标注（disk / memory / db_conn / …），
与评测集的 `category`（决定它属于哪一档难度）是两个正交的维度 ——
比如 multi-01 的分组是"多日志连锁"，但真因属于 memory（Pod 因内存压力被驱逐）。

用法
----
    python eval_retrieval.py               # 全部用例
    python eval_retrieval.py --k 5         # 看 top-5
    python eval_retrieval.py --verbose     # 打印每条用例检索到了什么
    python eval_retrieval.py --json out.json
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

if __name__ == "__main__":
    # ⚠️ 必须在导入任何项目模块**之前**切到 .venv。
    # 双击运行时用的是系统全局 Python，而依赖只装在 .venv 里 ——
    # 放到文件末尾调用就太晚了，import app.diagnose 会先因找不到 dotenv 而崩掉。
    from app.launcher import reenter_venv

    reenter_venv()

from app.context import build_context  # noqa: E402
from app.diagnose import format_logs  # noqa: E402
from app.parser import parse_text  # noqa: E402
from app.retriever import (  # noqa: E402
    BACKENDS,
    DEFAULT_BACKEND,
    IndexNotBuiltError,
    current_backend,
    make_retriever,
)

CASES_PATH = ROOT / "data" / "eval_cases.json"

# 用例 id → 真因所属的 fault_group（与 data/fault_cases.jsonl 的标注体系一致）
# 标为 None 的用例没有明确的单一故障类型（故意信息不足），不参与检索评测。
EXPECTED_GROUP: dict[str, str | None] = {
    "res-01": "disk",          # 磁盘配额超限
    "res-02": "memory",        # JVM 堆溢出
    "res-03": "fd_thread",     # 文件句柄耗尽
    "res-04": "fd_thread",     # 线程池拒绝策略
    "dep-01": "db_conn",       # 数据库连接数上限
    "dep-02": "mq",            # Kafka 重平衡
    "dep-03": "mq",            # MQ broker 断开
    "dep-04": "upstream",      # 上游 503
    "bug-01": "code_bug",      # 数组越界
    "bug-02": "code_bug",      # 类型转换失败
    "bug-03": "concurrency",   # 多线程死锁
    "bug-04": "concurrency",   # 并发修改集合
    "cfg-01": "cert",          # 证书过期
    "cfg-02": "config",        # 端口占用
    "cfg-03": "config",        # YAML 语法错误
    "net-01": "net",           # DNS 解析失败
    "net-02": "net",           # 连接被重置
    "net-03": "net",           # 读超时
    "os-01": "permission",     # 文件权限不足
    "os-02": "data",           # 唯一键冲突
    "tricky-01": "db_avail",   # 数据库只读模式
    "tricky-02": "db_avail",   # 数据库死锁
    "tricky-03": "code_bug",   # 无限递归
    "tricky-04": "lifecycle",  # 应用上下文已关闭
    "vague-01": None,
    "vague-02": None,
    "vague-03": None,
    "vague-04": None,
    "multi-01": "memory",      # 真因：Pod 因节点内存压力被驱逐
    "multi-02": "db_conn",     # 真因：慢查询占满连接池
    "multi-03": "cert",        # 真因：证书自动续期失败
    "multi-04": "upstream",    # 真因：上游发版做了接口重构
    "multi-05": "disk",        # 真因：本地磁盘写满
    "multi-06": "db_avail",    # 真因：MySQL 主从切换
    "noise-01": "config",      # 真因：配置中心推送了未知取值
    "noise-02": "upstream",    # 真因：灰度和发版改了接口路径
    "noise-03": "file_lock",   # 真因：库存数据文件被外部进程锁定
    "noise-04": "file_lock",
}


def load_cases() -> list[dict]:
    return json.loads(CASES_PATH.read_text(encoding="utf-8"))["cases"]


def build_query(case: dict) -> str:
    """按真实链路构造查询：解析 → 上下文聚合 → 格式化。

    不用原始日志全文，是因为生产链路里检索发生在聚合之后 ——
    多日志故障的真因往往在聚合进来的那几条里，直接拿原始文本会漏掉。
    """
    logs = parse_text("\n".join(case["log_lines"]))
    context = build_context(logs)
    return format_logs(context.logs)


def main() -> int:
    ap = argparse.ArgumentParser(description="检索质量评测（不调用大模型）")
    ap.add_argument("--k", type=int, default=3, help="取前 k 条（默认 3）")
    ap.add_argument("--top", type=int, default=5, help="无论 --k 多少，都统计到 top-N 的命中（默认 5）")
    ap.add_argument("--verbose", action="store_true", help="打印每条用例的检索结果")
    ap.add_argument("--json", help="把逐条结果导出为 JSON")
    ap.add_argument(
        "--backend",
        choices=list(BACKENDS),
        default=None,
        help="检索后端：bm25=词法；dense=向量；hybrid=RRF 融合。"
             "不指定时读 .env 里的 LOGWISE_RETRIEVER_BACKEND（默认 bm25）",
    )
    ap.add_argument("--compare", action="store_true", help="依次跑全部后端并输出并排对比")
    args = ap.parse_args()

    # 双击运行时不会有任何参数 —— 那多半就是想看三后端对比，直接给上。
    # 否则双击只能跑到默认后端，用户会以为"向量那两项没做"。
    if len(sys.argv) == 1:
        args.compare = True

    if args.compare:
        return run_compare(args)

    try:
        retriever = make_retriever(args.backend)
    except IndexNotBuiltError as exc:
        print(f"❌ {exc}")
        return 1
    return run_one(args, retriever, args.backend or current_backend())


def run_one(args, retriever, backend: str) -> int:
    cases = load_cases()
    top_n = max(args.k, args.top)

    print("=" * 82)
    print(f"故障案例检索质量评测（后端: {backend}）")
    print("=" * 82)
    if backend == DEFAULT_BACKEND:
        print("  说明        : 纯本地计算，零成本")
    else:
        print("  说明        : 每条 query 需调一次 embedding 接口，"
              f"约 {(top_n and len(cases) or 0) * 0.0001:.4f} 元量级")
    print(f"  检索后端    : {backend}")
    print(f"  知识库规模  : {retriever.size()} 条")
    groups = defaultdict(int)
    for c in cases:
        g = EXPECTED_GROUP.get(c["id"])
        if g:
            groups[g] += 1
    print(f"  参与评测用例: {sum(groups.values())} / {len(cases)} 条"
          f"（{len(cases) - sum(groups.values())} 条因无明确故障类型被排除）")
    print(f"  涉及故障类别: {len(groups)} 种")
    print(f"  检索深度    : top-{args.k}（命中统计到 top-{top_n}）")
    print()

    records = []
    ranks: list[int | None] = []

    for case in cases:
        expected = EXPECTED_GROUP.get(case["id"])
        if not expected:
            continue

        query = build_query(case)
        hits = retriever.search(query, k=top_n)
        hit_groups = [h.fault_group for h in hits]

        rank = next((i + 1 for i, g in enumerate(hit_groups) if g == expected), None)
        ranks.append(rank)

        records.append({
            "id": case["id"],
            "expected_group": expected,
            "expected": case["expected"],
            "retrieved": [{"id": h.id, "group": h.fault_group, "score": h.score} for h in hits],
            "rank": rank,
            "top1_ok": rank == 1,
            "hit_at_k": rank is not None and rank <= args.k,
            "hit_at_top": rank is not None,
        })

        if args.verbose:
            mark = "OK " if rank == 1 else ("~  " if rank else "MISS")
            shown = "  ".join(f"{g}({h.id})" for g, h in zip(hit_groups, hits))
            print(f"  [{mark}] {case['id']:10s} 期望 {expected:10s} 检索: {shown}")

    total = len(records)
    if not total:
        print("没有可评测的用例")
        return 1

    def rate(key: str) -> float:
        return sum(r[key] for r in records) / total * 100

    hit_k = sum(r["hit_at_k"] for r in records)
    hit_all = sum(r["hit_at_top"] for r in records)
    top1 = sum(r["top1_ok"] for r in records)
    got = [r for r in ranks if r]
    mrr = statistics.mean(1 / r for r in got) if got else 0.0

    print("=" * 82)
    print("整体结果")
    print("=" * 82)
    print(f"  Recall@{args.k:<2}          : {hit_k:3d}/{total}  = {rate('hit_at_k'):5.1f}%   （top-{args.k} 里有同类案例）")
    print(f"  Recall@{top_n:<2}          : {hit_all:3d}/{total}  = {rate('hit_at_top'):5.1f}%")
    print(f"  Precision@1（top-1 即正确）: {top1:3d}/{total}  = {rate('top1_ok'):5.1f}%")
    print(f"  MRR                    : {mrr:.3f}            （首个正确案例的排名倒数，越接近 1 越好）")
    print(f"  完全未召回到同类       : {total - hit_all:3d}/{total}  = {(total - hit_all) / total * 100:5.1f}%")

    # 随机的对照基线：知识库里总共多少种 group，随机取一条命中的概率
    all_groups = {
        json.loads(x)["fault_group"]
        for x in (ROOT / "data" / "fault_cases.jsonl").read_text(encoding="utf-8").splitlines()
        if x.strip()
    }
    print()
    print(f"  对照：知识库共 {len(all_groups)} 种故障类别，"
          f"若随机返回一条，命中同类的概率约 {sum(groups.values()) / total / len(all_groups) * 100:.1f}%")

    print()
    print("=" * 82)
    print("按故障类别统计")
    print("=" * 82)
    by_g: dict[str, list] = defaultdict(list)
    for r in records:
        by_g[r["expected_group"]].append(r)
    for g, items in sorted(by_g.items(), key=lambda kv: (-len(kv[1]), kv[0])):
        ok = sum(i["hit_at_k"] for i in items)
        bar = "#" * round(ok / len(items) * 10) + "." * (10 - round(ok / len(items) * 10))
        top1_ok = sum(i["top1_ok"] for i in items)
        print(f"  {g:12s} {ok}/{len(items)}  {bar}  top1 {top1_ok}/{len(items)}")

    misses = [r for r in records if not r["hit_at_top"]]
    if misses:
        print()
        print("=" * 82)
        print("完全未召唤到同类案例的用例")
        print("=" * 82)
        for r in misses:
            print(f"  {r['id']:10s} 期望 {r['expected_group']:10s} 实际: "
                  + ", ".join(f"{h['group']}/{h['id']}" for h in r["retrieved"][:3]))
            print(f"             真因: {r['expected']}")

    if args.json:
        out = Path(args.json)
        out.write_text(json.dumps({
            "summary": {
                "kb_size": retriever.size(),
                "evaluated": total,
                "k": args.k,
                "recall_at_k": hit_k / total,
                "recall_at_top": hit_all / total,
                "precision_at_1": top1 / total,
                "mrr": mrr,
                "miss_no_recall": (total - hit_all) / total,
            },
            "records": records,
        }, ensure_ascii=False, indent=2), encoding="utf-8")
        print()
        print(f"  已导出: {out}")

    return 0


def _evaluate(retriever, cases: list[dict], k: int, top_n: int) -> dict:
    """跑一遍并返回汇总指标。供 --compare 复用。"""
    ranks: list[int | None] = []
    for case in cases:
        expected = EXPECTED_GROUP.get(case["id"])
        if not expected:
            continue
        hits = retriever.search(build_query(case), k=top_n)
        ranks.append(next((i + 1 for i, h in enumerate(hits) if h.fault_group == expected), None))

    total = len(ranks) or 1
    hit_k = sum(1 for r in ranks if r and r <= k)
    hit_all = sum(1 for r in ranks if r)
    top1 = sum(1 for r in ranks if r == 1)
    got = [r for r in ranks if r]
    return {
        "total": len(ranks),
        "recall@1": top1 / total,
        "recall@k": hit_k / total,
        "recall@5": hit_all / total,
        "mrr": statistics.mean(1 / r for r in got) if got else 0.0,
        "miss": sum(1 for r in ranks if not r),
    }


def run_compare(args) -> int:
    """依次跑全部后端并输出并排对比。

    这是回答「向量到底带来多少增量」最直接的方式 ——
    同一套用例、同一条 query，只有检索算法不同，差异就只来自算法本身。
    """
    available: list[tuple[str, object]] = []
    for backend in BACKENDS:
        try:
            available.append((backend, make_retriever(backend)))
        except IndexNotBuiltError as exc:
            print(f"  ⚠️  跳过 {backend}：{exc}".split("\n")[0])
        except Exception as exc:  # noqa: BLE001
            print(f"  ⚠️  跳过 {backend}：{type(exc).__name__}: {exc}")

    if not available:
        print("没有可用的检索后端")
        return 1

    cases = load_cases()
    top_n = max(args.k, args.top)

    print("=" * 82)
    print(f"检索后端对比（top-{args.k}，命中统计到 top-{top_n}）")
    print("=" * 82)

    results = []
    for backend, retriever in available:
        t0 = time.perf_counter()
        metrics = _evaluate(retriever, cases, args.k, top_n)
        metrics["elapsed_ms"] = (time.perf_counter() - t0) * 1000 / (metrics["total"] or 1)
        results.append((backend, metrics))

    header = f"  {'后端':<10}{'Precision@1':>13}{'Recall@' + str(args.k):>11}{'Recall@5':>11}{'MRR':>9}{'未召回':>9}{'每条耗时':>11}"
    print(header)
    print("  " + "-" * 76)
    for backend, m in results:
        print(f"  {backend:<10}{m['recall@1'] * 100:>12.1f}%{m['recall@k'] * 100:>10.1f}%"
              f"{m['recall@5'] * 100:>10.1f}%{m['mrr']:>9.3f}{m['miss']:>9}{m['elapsed_ms']:>9.1f}ms")

    if len(results) >= 2:
        base_name, base = results[0]
        print()
        print(f"  相对 {base_name} 的增量：")
        for backend, m in results[1:]:
            print(f"    {backend:<10} Precision@1 {m['recall@1'] - base['recall@1']:+.1%}   "
                  f"Recall@{args.k} {m['recall@k'] - base['recall@k']:+.1%}   "
                  f"MRR {m['mrr'] - base['mrr']:+.3f}   每条耗时 {m['elapsed_ms'] - base['elapsed_ms']:+.1f}ms")

    if args.json:
        Path(args.json).write_text(
            json.dumps({"compare": {b: m for b, m in results}}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(f"\n  已导出: {args.json}")
    return 0


if __name__ == "__main__":
    from app.launcher import pause_if_console

    exit_code = main()
    pause_if_console()
    raise SystemExit(exit_code)
