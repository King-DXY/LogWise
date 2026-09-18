"""Prompt 模板的结构校验。

为什么值得测：Few-shot 示例里的 JSON 是**手写的字符串**，一旦写错（少个逗号、
字段名拼错），程序照跑不误，但模型会学到一个错误的格式，质量静默下降。
所以这里把「示例必须是合法且完整的 JSON」固化成回归测试。
"""
from __future__ import annotations

import json

from app.prompts import FEW_SHOT_EXAMPLES, PROMPT_VERSION, SYSTEM_PROMPT

EXPECTED_FIELDS = {
    "error_type",
    "root_cause",
    "evidence",
    "impact",
    "solution_steps",
    "confidence",
}


class TestPromptVersion:
    def test_version_is_not_empty(self):
        assert isinstance(PROMPT_VERSION, str)
        assert PROMPT_VERSION.strip()

    def test_version_bumped_after_cache_feature(self):
        """缓存键依赖这个版本号，它必须是个真实存在的值（防止被误删）。"""
        assert PROMPT_VERSION.startswith("v")


class TestSystemPrompt:
    def test_declares_every_output_field(self):
        for field in EXPECTED_FIELDS:
            assert field in SYSTEM_PROMPT, f"System Prompt 里没有提到字段 {field}"

    def test_contains_anti_leak_instruction(self):
        """必须有「示例只是格式，别当成真实历史」的约束。

        实测过：不加这条时，模型会说出"这与 4 月 2 日的同类问题一致"这种编造的话。
        """
        assert "示例" in SYSTEM_PROMPT
        assert "无关" in SYSTEM_PROMPT or "不要引用" in SYSTEM_PROMPT

    def test_contains_confidence_scale(self):
        """置信度必须有取值标尺，否则模型会习惯性全给 0.9。"""
        assert "confidence 取值参考" in SYSTEM_PROMPT

    def test_allows_admitting_insufficient_information(self):
        assert "信息不足" in SYSTEM_PROMPT

    def test_mentions_root_cause_vs_symptom(self):
        assert "根因" in SYSTEM_PROMPT and "症状" in SYSTEM_PROMPT


class TestFewShotExamples:
    def test_has_multiple_examples(self):
        assert len(FEW_SHOT_EXAMPLES) >= 3, "示例太少，覆盖不了不同故障类型"

    def test_every_example_has_log_and_diagnosis(self):
        for i, ex in enumerate(FEW_SHOT_EXAMPLES, 1):
            assert set(ex.keys()) == {"log", "diagnosis"}, f"示例 {i} 结构不对"
            assert ex["log"].strip(), f"示例 {i} 的 log 为空"

    def test_diagnosis_is_valid_json(self):
        for i, ex in enumerate(FEW_SHOT_EXAMPLES, 1):
            try:
                json.loads(ex["diagnosis"])
            except json.JSONDecodeError as exc:
                raise AssertionError(f"示例 {i} 的 diagnosis 不是合法 JSON: {exc}") from exc

    def test_diagnosis_has_exact_field_set(self):
        for i, ex in enumerate(FEW_SHOT_EXAMPLES, 1):
            got = set(json.loads(ex["diagnosis"]).keys())
            assert got == EXPECTED_FIELDS, (
                f"示例 {i} 字段不匹配：多出 {got - EXPECTED_FIELDS}，缺少 {EXPECTED_FIELDS - got}"
            )

    def test_confidence_is_a_valid_ratio(self):
        for i, ex in enumerate(FEW_SHOT_EXAMPLES, 1):
            conf = json.loads(ex["diagnosis"])["confidence"]
            assert isinstance(conf, (int, float)) and not isinstance(conf, bool)
            assert 0.0 <= conf <= 1.0, f"示例 {i} 的 confidence 越界: {conf}"

    def test_evidence_is_non_empty_list_of_strings(self):
        for i, ex in enumerate(FEW_SHOT_EXAMPLES, 1):
            evidence = json.loads(ex["diagnosis"])["evidence"]
            assert isinstance(evidence, list) and evidence, f"示例 {i} 的 evidence 必须非空"
            assert all(isinstance(item, str) and item.strip() for item in evidence)

    def test_solution_steps_is_non_empty_list(self):
        for i, ex in enumerate(FEW_SHOT_EXAMPLES, 1):
            steps = json.loads(ex["diagnosis"])["solution_steps"]
            assert isinstance(steps, list) and len(steps) >= 2, f"示例 {i} 的排查步骤太少"

    def test_examples_cover_different_fault_categories(self):
        """示例之间不应重复同类故障，否则覆盖面会打折。"""
        types = [json.loads(ex["diagnosis"])["error_type"] for ex in FEW_SHOT_EXAMPLES]
        assert len(set(types)) == len(types), f"存在重复的错误类型: {types}"
