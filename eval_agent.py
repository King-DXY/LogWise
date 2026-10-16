"""Agent 模式评测：同一批评测用例，基线 vs Agent 对照。

用法
----
    python eval_agent.py                 # 全量：基线 + Agent 双跑（41 条 × 2）
    python eval_agent.py --limit 5       # 冒烟：只跑前 5 条
    python eval_agent.py --only-agent    # 只跑 Agent（复用已有的基线数据时）
    python eval_agent.py --json out.json # 逐条明细导出

回答三个问题（对应 docs/EXPERIMENTS.md 第 7.2 节的假设）
------------------------------------------------------
1. 常见故障上，模型会不会多调工具、多花钱？（预期：不调，成本≈基线）
2. 私有错误码（NS-1042 等）上，Agent 能否靠查知识库命中？
   （基线模型不可能认识编造的错误码 —— 这是「私有知识才有检索价值」的直接检验）
3. 强制收尾（轮次上限）触发频率 —— Agent 失控风险的量化。

计费口径与 eval_diagnosis.py 一致：三档计价、真实 usage 累加（Agent 多轮全部计入）。
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
    # 同 eval_diagnosis.py：双击运行时先切到 .venv，再 import 项目模块
    from app.launcher import reenter_venv

    reenter_venv()

from dotenv import load_dotenv  # noqa: E402

from app.agent import AGENT_PROMPT_VERSION, max_tool_rounds, run_agent  # noqa: E402
from app.diagnose import current_model, diagnose_multi_ex  # noqa: E402
from app.metrics import price_input, price_input_cached, price_output  # noqa: E402
from app.parser import parse_text  # noqa: E402
from eval_diagnosis import (  # noqa: E402
    CASES_PATH,
    keyword_hit,
    log_to_text_with_context,
    schema_ok,
)

load_dotenv(ROOT / ".env")

PRIVATE_CATEGORY = "私有错误码"


def usage_cost(usage) -> float:
    """三档计价 —— 与 /stats 同一套单价。"""
    return (
        usage.uncached_prompt_tokens / 1e6 * price_input()
        + usage.cached_tokens / 1e6 * price_input_cached()
        + usage.completion_tokens / 1e6 * price_output()
    )


def run_one(label: str, case: dict, logs) -> dict:
    """跑一条用例。label 决定走基线还是 Agent。"""
    t0 = time.perf_counter()
    error = ""
    result: dict = {}
    usage = None
    trace_summary = {"tool_calls": 0, "rounds": 0, "forced": False, "tools": []}

    try:
        if label == "agent":
            result, usage, trace = run_agent(logs)
            trace_summary = {
                "tool_calls": trace.tool_call_count,
                "rounds": len(trace.rounds),
                "forced": trace.forced_answer,
                "tools": [
                    c["name"]
                    for r in trace.rounds
                    for c in r.get("tool_calls", [])
                ],
            }
        else:  # baseline
            result, usage = diagnose_multi_ex(logs)
    except Exception as exc:  # 单条失败不中断整轮
        error = f"{type(exc).__name__}: {exc}"

    elapsed = (time.perf_counter() - t0) * 1000
    hit = keyword_hit(result, case.get("keywords", [])) if case.get("keywords") else None

    return {
        "id": case["id"],
        "category": case["category"],
        "is_private": case["category"] == PRIVATE_CATEGORY,
        "hit": hit,
        "schema_ok": schema_ok(result) if not error else False,
        "tool_calls": trace_summary["tool_calls"] if label == "agent" else 0,
        "rounds": trace_summary["rounds"] if label == "agent" else 1,
        "forced": trace_summary["forced"] if label == "agent" else False,
        "tools": trace_summary["tools"],
        "prompt_tokens": usage.prompt_tokens if usage else 0,
        "cached_tokens": usage.cached_tokens if usage else 0,
        "completion_tokens": usage.completion_tokens if usage else 0,
        "cost": round(usage_cost(usage), 6) if usage else 0.0,
        "duration_ms": round(elapsed, 1),
        "error": error,
    }


def summarize(records: list[dict]) -> dict:
    scored = [r for r in records if r["hit"] is not None]
    # tool_calls / rounds 字段只有 Agent 组才有意义，基线组恒为 0/1 —— 统一计算无副作用。
    return {
        "n": len(records),
        "hit_rate": (sum(1 for r in scored if r["hit"]) / len(scored)) if scored else None,
        "schema_ok_rate": sum(1 for r in records if r["schema_ok"]) / len(records) if records else 0,
        "avg_cost": statistics.mean(r["cost"] for r in records) if records else 0,
        "total_cost": sum(r["cost"] for r in records),
        "avg_duration_ms": statistics.mean(r["duration_ms"] for r in records) if records else 0,
        "agent": {
            "avg_tool_calls": statistics.mean(r["tool_calls"] for r in records) if records else 0,
            "avg_rounds": statistics.mean(r["rounds"] for r in records) if records else 0,
            "forced_count": sum(1 for r in records if r["forced"]),
            "zero_tool_rate": sum(1 for r in records if r["tool_calls"] == 0) / len(records) if records else 0,
        },
    }


def print_group_table(label: str, records: list[dict]) -> None:
    by_group: dict[str, list[dict]] = defaultdict(list)
    for r in records:
        by_group["私有错误码" if r["is_private"] else "通用故障"].append(r)

    print(f"\n[{label}] 分组对比")
    print(f"  {'组':6s} {'命中':>8s} {'平均工具调用':>10s} {'0 次调用占比':>10s} {'平均成本':>10s}")
    for group, recs in by_group.items():
        scored = [r for r in recs if r["hit"] is not None]
        hit = (sum(1 for r in scored if r["hit"]) / len(scored)) if scored else float("nan")
        zero = sum(1 for r in recs if r["tool_calls"] == 0) / len(recs)
        print(
            f"  {group:6s} {hit * 100:7.1f}% "
            f"{statistics.mean(r['tool_calls'] for r in recs):10.2f} "
            f"{zero * 100:9.1f}% "
            f"{statistics.mean(r['cost'] for r in recs) * 1000:9.4f}‰"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="LogWise Agent 模式对照评测")
    parser.add_argument("--limit", type=int, default=0, help="只跑前 N 条（冒烟用）")
    parser.add_argument("--only-agent", action="store_true", help="跳过基线，只跑 Agent")
    parser.add_argument(
        "--category",
        metavar="NAME",
        default=None,
        help="只跑指定类别的用例（如 --category 私有错误码），用于局部重跑",
    )
    parser.add_argument("--json", metavar="PATH", help="逐条明细导出为 JSON")
    args = parser.parse_args()

    dataset = json.loads(CASES_PATH.read_text(encoding="utf-8"))
    cases = dataset["cases"]
    if args.category:
        cases = [c for c in cases if c["category"] == args.category]
    if args.limit:
        cases = cases[: args.limit]

    print("=" * 78)
    print("LogWise Agent 模式对照评测")
    print("=" * 78)
    print(f"  模型         : {current_model()}")
    print(f"  Agent Prompt : {AGENT_PROMPT_VERSION}（工具轮次上限 {max_tool_rounds()}）")
    print(f"  用例数       : {len(cases)}（含 私有错误码 {sum(1 for c in cases if c['category'] == PRIVATE_CATEGORY)} 条）")
    print(f"  单价         : 缓存命中 {price_input_cached()} / 未命中 {price_input()} / 输出 {price_output()} 元每百万")
    print("=" * 78)

    out: dict = {"baseline": [], "agent": []}

    if not args.only_agent:
        print("\n---- 基线（固定 Few-shot，不检索）----")
        for i, case in enumerate(cases, 1):
            logs = parse_text("\n".join(case["log_lines"]))
            rec = run_one("baseline", case, logs)
            out["baseline"].append(rec)
            mark = "✅" if rec["hit"] else ("➖" if rec["hit"] is None else "❌")
            print(
                f"  [{i:2d}/{len(cases)}] {mark} {case['id']:12s} "
                f"{rec['prompt_tokens']:5d}→{rec['completion_tokens']:4d} tok  {rec['duration_ms']:6.0f}ms"
            )

    print("\n---- Agent 模式（模型自主决定是否查工具）----")
    for i, case in enumerate(cases, 1):
        logs = parse_text("\n".join(case["log_lines"]))
        rec = run_one("agent", case, logs)
        out["agent"].append(rec)
        mark = "✅" if rec["hit"] else ("➖" if rec["hit"] is None else "❌")
        tools = ",".join(dict.fromkeys(rec["tools"])) or "-"
        print(
            f"  [{i:2d}/{len(cases)}] {mark} {case['id']:12s} "
            f"工具×{rec['tool_calls']}（{tools}）  "
            f"{rec['prompt_tokens']:5d}→{rec['completion_tokens']:4d} tok  {rec['duration_ms']:6.0f}ms"
        )
        if rec["forced"]:
            print(f"          ⚠️ 轮次上限触发，强制收尾")

    # ---------------- 汇总 ---------------- #
    print("\n" + "=" * 78)
    print("汇总")
    print("=" * 78)
    for label in ("baseline", "agent"):
        if not out[label]:
            continue
        s = summarize(out[label])
        hit_text = f"{s['hit_rate'] * 100:.1f}%" if s["hit_rate"] is not None else " N/A"
        print(f"  [{label:8s}] 命中 {hit_text}  格式合规 {s['schema_ok_rate'] * 100:.1f}%  "
              f"平均成本 {s['avg_cost'] * 1000:.4f}‰  平均耗时 {s['avg_duration_ms']:.0f}ms")
        if label == "agent":
            a = s["agent"]
            print(f"              工具调用平均 {a['avg_tool_calls']:.2f} 次  "
                  f"0 次调用占比 {a['zero_tool_rate'] * 100:.1f}%  "
                  f"强制收尾 {a['forced_count']} 次")

    if out["baseline"] and out["agent"]:
        print_group_table("基线", out["baseline"])
        print_group_table("Agent", out["agent"])
        b, a = summarize(out["baseline"]), summarize(out["agent"])
        delta = (a["total_cost"] - b["total_cost"]) / b["total_cost"] * 100 if b["total_cost"] else 0
        print(f"\n  成本差：Agent 全程 {'+' if delta >= 0 else ''}{delta:.1f}%（基线 {b['total_cost']:.4f} 元 vs Agent {a['total_cost']:.4f} 元）")

    if args.json:
        Path(args.json).write_text(
            json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(f"\n明细已写入 {args.json}")


if __name__ == "__main__":
    main()
