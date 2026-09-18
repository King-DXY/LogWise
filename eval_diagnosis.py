"""诊断质量评测：用人工标注的评测集量化 Prompt 的准确率与成本。

用法
----
    python eval_diagnosis.py                    # 评测当前 Prompt
    python eval_diagnosis.py --from-git HEAD    # 评测某个 git 版本的 Prompt（做 A/B 对比）
    python eval_diagnosis.py --limit 5          # 只跑前 5 条（快速验证脚本本身）
    python eval_diagnosis.py --json out.json    # 顺便导出逐条明细

评测三个维度
------------
1. **错误类型命中率**：预测的 error_type / root_cause 是否包含标注的关键词。
   只考察有没有抓对故障类别，不苛求措辞一致。
2. **输出格式合规率**：返回的是否为合法 JSON，且包含全部必需字段。
3. **证据可溯源率**：evidence 里引用的日志片段能否在原始日志中找到。
   —— 这是对幻觉的客观度量：编造的"证据"不可能在日志里出现。

同时统计 token 用量、耗时与估算成本，用于回答"这次 Prompt 改动值不值"。
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv
from openai import OpenAI

from app import cache as cache_mod
from app.context import build_context, context_enabled, context_max_logs, context_window
from app.diagnose import _extract_usage, current_model, format_logs
from app.metrics import price_input, price_input_cached, price_output
from app.parser import parse_text
from app.prompts import FEW_SHOT_EXAMPLES, PROMPT_VERSION, SYSTEM_PROMPT

load_dotenv(ROOT / ".env")

# 两版 Prompt 都要求的核心字段 —— 用它来判定「格式是否合规」，
# 保证新旧版本对比时口径一致。
CORE_FIELDS = {
    "error_type",
    "root_cause",
    "impact",
    "solution_steps",
    "confidence",
}
# v3 新增字段，单独统计（旧版没有它，不能拿它当合规判据）
EVIDENCE_FIELD = "evidence"

CASES_PATH = ROOT / "data" / "eval_cases.json"
EVIDENCE_WINDOW = 12  # 证据片段匹配时的滑窗长度（容忍轻微改写）


# --------------------------------------------------------------------------- #
# Prompt 加载：支持从 git 历史版本取，用于 A/B 对比
# --------------------------------------------------------------------------- #
def load_prompts_from_git(ref: str) -> tuple[str, list, str]:
    try:
        src = subprocess.check_output(
            ["git", "show", f"{ref}:app/prompts.py"], cwd=str(ROOT), stderr=subprocess.PIPE
        ).decode("utf-8")
    except subprocess.CalledProcessError as exc:
        raise SystemExit(f"无法从 git 读取 {ref}:app/prompts.py —— {exc.stderr.decode('utf-8', 'replace')}")
    namespace: dict = {}
    exec(compile(src, f"prompts@{ref}", "exec"), namespace)  # noqa: S102
    return (
        namespace["SYSTEM_PROMPT"],
        namespace["FEW_SHOT_EXAMPLES"],
        namespace.get("PROMPT_VERSION", ref),
    )


def build_messages(log_text: str, system: str, few_shot: list) -> list[dict]:
    messages = [{"role": "system", "content": system}]
    for example in few_shot:
        messages.append({"role": "user", "content": example["log"]})
        messages.append({"role": "assistant", "content": example["diagnosis"]})
    messages.append({"role": "user", "content": log_text})
    return messages


def log_to_text_with_context(logs: list) -> tuple[str, int]:
    """用**生产环境同一套**上下文聚合逻辑构造输入。

    这一点很关键：如果评测脚本自己拼日志、而生产代码用另一套逻辑，
    那 A/B 对比就失去了意义。这里直接复用 ``app.context.build_context``，
    因此 ``LOGWISE_CONTEXT_ENABLED`` / ``_WINDOW`` / ``_MAX_LOGS`` 三个环境变量
    能同时控制评测与线上行为。

    返回 (给模型看的文本, 实际送进去的日志条数)。
    """
    context = build_context(logs)
    return format_logs(context.logs), context.size


# --------------------------------------------------------------------------- #
# 打分
# --------------------------------------------------------------------------- #
def keyword_hit(result: dict, keywords: list[str]) -> bool:
    haystack = f"{result.get('error_type', '')} {result.get('root_cause', '')}".lower()
    return any(kw.lower() in haystack for kw in keywords)


def evidence_grounded(snippet: str, normalized_log: str) -> bool:
    """判断一条 evidence 是否真的出自原始日志。

    先试整体匹配；不中再用滑窗，容忍模型对原文做的小幅改写（引号、省略号等）。
    """
    text = cache_mod.normalize(snippet)
    if not text:
        return False
    if text in normalized_log:
        return True
    if len(text) <= EVIDENCE_WINDOW:
        return False
    for i in range(0, len(text) - EVIDENCE_WINDOW + 1, 4):
        if text[i : i + EVIDENCE_WINDOW] in normalized_log:
            return True
    return False


def schema_ok(result: dict) -> bool:
    """核心字段是否齐全（新旧 Prompt 用同一套判据）。"""
    return isinstance(result, dict) and CORE_FIELDS.issubset(result.keys())


# --------------------------------------------------------------------------- #
# 主流程
# --------------------------------------------------------------------------- #
def main() -> None:
    parser = argparse.ArgumentParser(description="LogWise 诊断质量评测")
    parser.add_argument("--from-git", metavar="REF", help="从指定 git 版本加载 Prompt（如 HEAD）")
    parser.add_argument("--limit", type=int, default=0, help="只跑前 N 条")
    parser.add_argument("--json", metavar="PATH", help="把逐条明细导出为 JSON")
    args = parser.parse_args()

    if args.from_git:
        system, few_shot, version = load_prompts_from_git(args.from_git)
        label = f"{version}（来自 git:{args.from_git}）"
    else:
        system, few_shot, version = SYSTEM_PROMPT, FEW_SHOT_EXAMPLES, PROMPT_VERSION
        label = version

    dataset = json.loads(CASES_PATH.read_text(encoding="utf-8"))
    cases = dataset["cases"]
    if args.limit:
        cases = cases[: args.limit]

    client = OpenAI(api_key=os.getenv("DEEPSEEK_API_KEY"), base_url=os.getenv("DEEPSEEK_BASE_URL"))
    model = current_model()

    print("=" * 78)
    print("LogWise 诊断质量评测")
    print("=" * 78)
    print(f"  Prompt   : {label}  （System {len(system)} 字符 + {len(few_shot)} 个示例）")
    print(f"  模型     : {model}")
    print(f"  用例数   : {len(cases)}")
    if context_enabled():
        print(f"  上下文聚合: 开启（窗口 ±{context_window()} 条，上限 {context_max_logs()} 条）")
    else:
        print("  上下文聚合: 关闭（只送锚点一条 —— 旧行为）")
    print(f"  单价     : 缓存命中输入 {price_input_cached()} / 未命中输入 {price_input()} / 输出 {price_output()} 元每百万")
    print("=" * 78)

    records: list[dict] = []
    for i, case in enumerate(cases, 1):
        text = "\n".join(case["log_lines"])
        logs = parse_text(text)
        if not logs:
            print(f"  [{i:2d}/{len(cases)}] {case['id']} 解析失败，跳过")
            continue
        body, ctx_size = log_to_text_with_context(logs)

        t0 = time.perf_counter()
        raw = ""
        error = ""
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=build_messages(body, system, few_shot),
                temperature=0.2,
                response_format={"type": "json_object"},
            )
            raw = resp.choices[0].message.content or ""
            usage = _extract_usage(resp.usage)
        except Exception as exc:  # 单条失败不应中断整轮评测
            error = f"{type(exc).__name__}: {exc}"
            usage = _extract_usage(None)
        elapsed = (time.perf_counter() - t0) * 1000

        try:
            result = json.loads(raw)
        except json.JSONDecodeError:
            result = {}

        keywords = case.get("keywords", [])
        difficulty = case.get("difficulty", "clear")
        expect_low = case.get("expect_low_confidence", False)
        threshold = case.get("confidence_threshold", 0.7)

        ok_schema = schema_ok(result)
        # 命中判定只看有没有抓对故障类别，不依赖字段是否齐全 ——
        # 否则旧版 Prompt（没有 evidence 字段）会被整批判错，新旧对比就不公平了
        # keywords 为空的用例（信息不足类）不参与命中率统计
        hit = keyword_hit(result, keywords) if keywords else None
        has_evidence = EVIDENCE_FIELD in result
        confidence = result.get("confidence")
        conf_ok = isinstance(confidence, (int, float)) and not isinstance(confidence, bool)
        # 信息不足的用例：考察模型是否诚实给出低置信度，而不是硬编一个答案
        honest = (conf_ok and confidence <= threshold) if expect_low else None

        evidence = result.get("evidence") or []
        log_norm = cache_mod.normalize(text)
        grounded = [e for e in evidence if isinstance(e, str) and evidence_grounded(e, log_norm)]

        cost = (
            usage.uncached_prompt_tokens / 1e6 * price_input()
            + usage.cached_tokens / 1e6 * price_input_cached()
            + usage.completion_tokens / 1e6 * price_output()
        )

        records.append({
            "id": case["id"],
            "category": case["category"],
            "difficulty": difficulty,
            "expected": case["expected"],
            "predicted": result.get("error_type", ""),
            "confidence": confidence if conf_ok else None,
            "hit": hit,
            "honest": honest,
            "schema_ok": ok_schema,
            "has_evidence": has_evidence,
            "evidence_total": len(evidence),
            "evidence_grounded": len(grounded),
            "context_logs": ctx_size,
            "prompt_tokens": usage.prompt_tokens,
            "cached_tokens": usage.cached_tokens,
            "completion_tokens": usage.completion_tokens,
            "duration_ms": round(elapsed, 1),
            "cost": round(cost, 6),
            "error": error,
        })

        conf_text = f"{confidence:.2f}" if conf_ok else " N/A"
        if expect_low:
            mark = "✅" if honest else "❌"
            print(
                f"  [{i:2d}/{len(cases)}] {mark} {case['id']:9s} {case['category']:6s} "
                f"上下文 {ctx_size:2d}条  置信度 {conf_text}  证据 {len(grounded)}/{len(evidence)}  "
                f"{usage.prompt_tokens:5d}→{usage.completion_tokens:4d} tok  {elapsed:6.0f}ms"
            )
            if not honest:
                print(f"          期望置信度 ≤ {threshold}（信息不足时就该承认，不该硬猜）")
                print(f"          实际根因: {str(result.get('root_cause', ''))[:70]}")
        else:
            mark = "✅" if hit else "❌"
            print(
                f"  [{i:2d}/{len(cases)}] {mark} {case['id']:9s} {case['category']:6s} "
                f"上下文 {ctx_size:2d}条  置信度 {conf_text}  证据 {len(grounded)}/{len(evidence)}  "
                f"{usage.prompt_tokens:5d}→{usage.completion_tokens:4d} tok  {elapsed:6.0f}ms"
            )
            if not hit:
                print(f"          期望: {case['expected']}")
                print(f"          预测: {result.get('error_type', '(无)')}")
        if error:
            print(f"          错误: {error}")

    if not records:
        raise SystemExit("没有产生任何评测记录")

    # ---------------- 汇总 ---------------- #
    total = len(records)
    # 只有带标注关键词的用例才参与命中率统计（信息不足类用例不打这个分）
    scored = [r for r in records if r["hit"] is not None]
    hits = sum(r["hit"] for r in scored)
    schemas = sum(r["schema_ok"] for r in records)
    ev_total = sum(r["evidence_total"] for r in records)
    ev_ok = sum(r["evidence_grounded"] for r in records)
    avg_in = statistics.mean(r["prompt_tokens"] for r in records)
    avg_cached = statistics.mean(r["cached_tokens"] for r in records)
    avg_out = statistics.mean(r["completion_tokens"] for r in records)
    avg_ms = statistics.mean(r["duration_ms"] for r in records)
    avg_ctx = statistics.mean(r["context_logs"] for r in records)
    total_cost = sum(r["cost"] for r in records)

    # 置信度校准：清晰的用例应该给高置信度，信息不足的用例应该给低置信度。
    # 两者的差值就是「置信度有没有区分度」—— 一个只会给 0.9 的模型差值为 0。
    vague = [r for r in records if r["difficulty"] == "vague"]
    honest_n = sum(1 for r in vague if r["honest"])
    clear_conf = [r["confidence"] for r in records
                  if r["difficulty"] in ("clear", "tricky") and r["confidence"] is not None]
    vague_conf = [r["confidence"] for r in vague if r["confidence"] is not None]
    gap = None
    if clear_conf and vague_conf:
        gap = statistics.mean(clear_conf) - statistics.mean(vague_conf)

    print()
    print("=" * 78)
    print("分类统计")
    print("=" * 78)
    by_cat: dict[str, list[dict]] = defaultdict(list)
    for r in records:
        by_cat[r["category"]].append(r)
    for cat, items in sorted(by_cat.items(), key=lambda kv: -len(kv[1])):
        scored_items = [i for i in items if i["hit"] is not None]
        if not scored_items:
            # 信息不足类：不适用命中率，改看「是否诚实给出低置信度」
            honest_items = [i for i in items if i["honest"]]
            print(f"  {cat:8s} 诚实 {len(honest_items)}/{len(items)}  （考察信息不足时是否给低置信度）")
            continue
        cat_hits = sum(i["hit"] for i in scored_items)
        ratio = cat_hits / len(scored_items)
        bar = "█" * round(ratio * 10) + "░" * (10 - round(ratio * 10))
        print(f"  {cat:8s} {cat_hits}/{len(scored_items)}  {bar}  {ratio*100:5.1f}%")

    print()
    print("=" * 78)
    print("汇总")
    print("=" * 78)
    print(f"  用例数              : {total}  （其中信息不足类 {len(vague)} 条，不参与命中率）")
    if scored:
        print(f"  错误类型命中率      : {hits}/{len(scored)}  = {hits/len(scored)*100:.1f}%")
    print(f"  输出格式合规率      : {schemas}/{total}  = {schemas/total*100:.1f}%   （核心字段齐全）")
    ev_cases = sum(r["has_evidence"] for r in records)
    print(f"  含 evidence 字段    : {ev_cases}/{total}  = {ev_cases/total*100:.1f}%   （v3 新增字段）")
    if ev_total:
        grounded_rate = ev_ok / ev_total * 100
        print(f"  证据可溯源率        : {ev_ok}/{ev_total}  = {grounded_rate:.1f}%   （越低说明幻觉越多）")
    else:
        grounded_rate = 0.0
        print("  证据可溯源率        : 不适用（该版本 Prompt 不输出 evidence 字段）")
    if vague:
        print(f"  低置信度诚实率      : {honest_n}/{len(vague)}  = {honest_n/len(vague)*100:.1f}%   （信息不足时应给低置信度）")
    if gap is not None:
        print(f"  置信度区分度        : {gap:+.3f}   （清晰用例 {statistics.mean(clear_conf):.2f} − 信息不足 {statistics.mean(vague_conf):.2f}，越大越有区分力）")
    print(f"  平均输入 token      : {avg_in:.0f}  (其中命中缓存 {avg_cached:.0f}，{avg_cached/avg_in*100:.0f}%)")
    print(f"  平均输出 token      : {avg_out:.0f}")
    print(f"  平均上下文条数      : {avg_ctx:.1f} 条")
    print(f"  平均耗时            : {avg_ms:.0f} ms")
    print(f"  单次成本            : {total_cost/total:.6f} 元")
    print(f"  本轮总成本          : {total_cost:.4f} 元")
    print()
    hit_rate = hits / len(scored) * 100 if scored else 0.0
    honest_rate = honest_n / len(vague) * 100 if vague else 0.0
    print()
    print(
        f"SUMMARY|{label}|{total}|{hit_rate:.1f}|{schemas/total*100:.1f}|{grounded_rate:.1f}"
        f"|{honest_rate:.1f}|{gap if gap is not None else 'NA'}"
        f"|{avg_in:.0f}|{avg_out:.0f}|{total_cost/total:.6f}|{avg_ms:.0f}"
    )

    if args.json:
        out_path = Path(args.json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(
            json.dumps({"label": label, "summary": {
                "total": total,
                "scored_cases": len(scored),
                "hit_rate": hits / len(scored) if scored else None,
                "schema_rate": schemas / total,
                "evidence_field_rate": ev_cases / total,
                "evidence_grounded_rate": ev_ok / ev_total if ev_total else None,
                "low_confidence_honest_rate": honest_n / len(vague) if vague else None,
                "confidence_gap": gap,
                "avg_confidence_clear": statistics.mean(clear_conf) if clear_conf else None,
                "avg_confidence_vague": statistics.mean(vague_conf) if vague_conf else None,
                "avg_prompt_tokens": avg_in,
                "avg_completion_tokens": avg_out,
                "avg_duration_ms": avg_ms,
                "total_cost": total_cost,
            }, "records": records}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(f"  明细已导出: {args.json}")


if __name__ == "__main__":
    main()
