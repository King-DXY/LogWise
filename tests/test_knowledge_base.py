"""故障案例知识库的结构与隔离校验。

为什么值得测：RAG 的检索质量完全取决于这个库。
如果库本身格式不对、字段缺失，或者与评测集 / Few-shot 示例撞了内容，
那么「RAG 让准确率提升了多少」这个结论就是假的 —— 而且不会报错。

两类判据要区分清楚，否则要么误报、要么漏报：

- **不算泄漏**：JDK / 驱动库的标准堆栈帧与通用异常消息。
  例如 `at java.base/java.io.File.createNewFile(File.java:1035)`、
  `java.lang.OutOfMemoryError: Java heap space`、
  `Caused by: java.sql.SQLException: The MySQL server is running with the --read-only option`。
  这些是语言与中间件的固定输出，任何同类故障都会出现，不含业务上下文。

- **算泄漏**：业务签名相同 —— 即带自定义包名（com.example.）或中文业务描述的行完全一致。
  这说明知识库里躺着一段与评测用例同一个业务现场的日志，
  检索会直接命中它，模型等于抄答案。
"""
from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path

import pytest

from app.parser import parse_text

KB_PATH = Path(__file__).resolve().parent.parent / "data" / "fault_cases.jsonl"
DATASET_PATH = Path(__file__).resolve().parent.parent / "data" / "eval_cases.json"

DIAG_FIELDS = {
    "error_type",
    "root_cause",
    "evidence",
    "impact",
    "solution_steps",
    "confidence",
}
REQUIRED = ("id", "fault_group", "fault_type", "summary", "log", "diagnosis")
ABNORMAL_LEVELS = {"ERROR", "WARN", "CRITICAL", "FATAL"}


def _business_signature(text: str) -> set[str]:
    """提取业务签名：带自定义包名或中文的行，剔除标准库堆栈帧与通用异常消息。"""
    out: set[str] = set()
    for line in text.split("\n"):
        n = re.sub(r"\s+", " ", line).strip()
        if not n or n.lower().startswith("at "):
            continue
        if "com.example." in n or re.search(r"[\u4e00-\u9fa5]", n):
            out.add(n.lower())
    return out


@pytest.fixture(scope="module")
def cases() -> list[dict]:
    return [json.loads(x) for x in KB_PATH.read_text(encoding="utf-8").splitlines() if x.strip()]


@pytest.fixture(scope="module")
def eval_cases() -> list[dict]:
    return json.loads(DATASET_PATH.read_text(encoding="utf-8"))["cases"]


class TestKnowledgeBaseStructure:
    def test_file_exists_and_parses(self, cases: list[dict]):
        """每行必须是独立合法的 JSON —— 坏一行整个知识库就等于被截断了。"""
        assert KB_PATH.exists(), f"知识库缺失: {KB_PATH}"
        assert len(cases) >= 40, f"知识库只有 {len(cases)} 条，规模不足会让检索失去区分度"

    def test_ids_are_unique(self, cases: list[dict]):
        ids = [c["id"] for c in cases]
        dup = [i for i, n in Counter(ids).items() if n > 1]
        assert not dup, f"存在重复 id: {dup}"

    def test_required_fields_present(self, cases: list[dict]):
        for c in cases:
            missing = [k for k in REQUIRED if not c.get(k)]
            assert not missing, f"{c.get('id', '?')} 缺字段: {missing}"

    def test_diagnosis_matches_prompt_output_format(self, cases: list[dict]):
        """诊断字段必须与 Prompt 要求模型输出的结构一致，否则没法直接当 few-shot 用。"""
        for c in cases:
            d = c["diagnosis"]
            assert set(d) == DIAG_FIELDS, (
                f"{c['id']} 的 diagnosis 字段不符: 多={set(d) - DIAG_FIELDS} 少={DIAG_FIELDS - set(d)}"
            )
            assert isinstance(d["evidence"], list) and d["evidence"], f"{c['id']} 的 evidence 必须是非空列表"
            assert isinstance(d["solution_steps"], list) and d["solution_steps"], (
                f"{c['id']} 的 solution_steps 必须是非空列表"
            )
            conf = d["confidence"]
            assert isinstance(conf, (int, float)) and 0 <= conf <= 1, f"{c['id']} 的 confidence 非法: {conf}"

    def test_every_case_is_parsable_as_abnormal_log(self, cases: list[dict]):
        """案例日志必须能被解析器识别出异常级别，否则检索时提取不到特征。"""
        for c in cases:
            logs = parse_text(c["log"])
            assert logs, f"{c['id']} 解析出 0 条日志"
            assert ABNORMAL_LEVELS & {log.level for log in logs}, (
                f"{c['id']} 没有 ERROR/WARN 级别日志: {[log.level for log in logs]}"
            )

    def test_covers_enough_fault_groups(self, cases: list[dict]):
        """故障类别太少的话，检索随便返回一条都可能撞对，指标会虚高。"""
        groups = Counter(c["fault_group"] for c in cases)
        assert len(groups) >= 12, f"故障类别只有 {len(groups)} 种，不足以验证检索能力"
        thin = [g for g, n in groups.items() if n < 2]
        assert not thin, f"这些类别只有 1 条案例，检索时没有同类可选: {thin}"


class TestKnowledgeBaseIsolation:
    """知识库不得与评测集 / Few-shot 示例共享业务现场。"""

    def test_no_shared_business_signature_with_eval_cases(self, cases, eval_cases):
        for c in eval_cases:
            sig = _business_signature("\n".join(c["log_lines"]))
            if not sig:
                continue
            for kb in cases:
                shared = sig & _business_signature(kb["log"])
                assert not shared, (
                    f"知识库 {kb['id']} 与评测用例 {c['id']} 有相同的业务行：{sorted(shared)[0][:70]}。"
                    "检索时会直接命中它，等于抄答案，请换成不同的业务现场。"
                )

    def test_no_shared_business_signature_with_few_shot(self, cases):
        from app.prompts import FEW_SHOT_EXAMPLES

        ex_sig = set()
        for ex in FEW_SHOT_EXAMPLES:
            ex_sig |= _business_signature(ex["log"])
        for kb in cases:
            shared = _business_signature(kb["log"]) & ex_sig
            assert not shared, (
                f"知识库 {kb['id']} 与 Few-shot 示例有相同的业务行：{sorted(shared)[0][:70]}。"
                "这会造成重复注入，浪费 token。"
            )

    def test_timestamps_do_not_overlap_with_eval_cases(self, cases, eval_cases):
        """知识库定位为「历史故障」，时间应与评测集错开，避免看起来像同一批数据。"""
        def dates(lines: list[str]) -> set[str]:
            out = set()
            for line in lines:
                m = re.match(r"(\d{4}-\d{2}-\d{2})", line.strip())
                if m:
                    out.add(m.group(1))
            return out

        eval_dates = set()
        for c in eval_cases:
            eval_dates |= dates(c["log_lines"])
        kb_dates = set()
        for c in cases:
            kb_dates |= dates(c["log"].split("\n"))

        overlap = eval_dates & kb_dates
        assert not overlap, f"知识库与评测集有相同日期: {sorted(overlap)[:5]}"
