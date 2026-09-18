"""评测集自身的结构校验。

为什么值得测：评测集是「判断 Prompt 改得好不好」的唯一依据。
如果数据集本身格式不对（关键词语义错位、用例不能被解析器识别、
信息不足类忘了标注阈值），跑出来的分数就是假的，而且不会报错。
"""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import pytest

from app.parser import parse_text

DATASET = Path(__file__).resolve().parent.parent / "data" / "eval_cases.json"


@pytest.fixture(scope="module")
def dataset() -> dict:
    return json.loads(DATASET.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def cases(dataset: dict) -> list[dict]:
    return dataset["cases"]


class TestDatasetStructure:
    def test_dataset_file_exists(self):
        assert DATASET.exists(), f"评测集缺失: {DATASET}"

    def test_has_enough_cases(self, cases: list[dict]):
        assert len(cases) >= 20, "用例太少，结论不可靠"

    def test_ids_are_unique(self, cases: list[dict]):
        ids = [c["id"] for c in cases]
        assert len(set(ids)) == len(ids), f"存在重复 id: {[i for i, n in Counter(ids).items() if n > 1]}"

    def test_required_fields_present(self, cases: list[dict]):
        for c in cases:
            for field in ("id", "category", "expected", "keywords", "log_lines"):
                assert field in c, f"{c.get('id', '?')} 缺少字段 {field}"
            assert isinstance(c["log_lines"], list) and c["log_lines"], f"{c['id']} 的 log_lines 为空"

    def test_every_log_is_parsable(self, cases: list[dict]):
        for c in cases:
            logs = parse_text("\n".join(c["log_lines"]))
            assert logs, f"{c['id']} 的日志无法被解析器识别"
            assert any(l.level in ("ERROR", "CRITICAL", "FATAL") for l in logs), (
                f"{c['id']} 里没有 ERROR 级别日志，诊断接口会挑不到目标"
            )

    def test_covers_multiple_categories(self, cases: list[dict]):
        categories = {c["category"] for c in cases}
        assert len(categories) >= 5, f"类别覆盖不足: {categories}"


class TestScoringAnnotations:
    def test_keyword_cases_have_keywords(self, cases: list[dict]):
        """参与命中率统计的用例必须有非空关键词，否则 hit 恒为 False。"""
        for c in cases:
            if not c.get("expect_low_confidence"):
                assert c["keywords"], f"{c['id']} 没有标注关键词，无法判定命中"

    def test_vague_cases_have_no_keywords(self, cases: list[dict]):
        """信息不足类不应有命中关键词 —— 它考察的是置信度而非类别判断。"""
        for c in cases:
            if c.get("expect_low_confidence"):
                assert not c["keywords"], f"{c['id']} 是信息不足类，不该标注命中关键词"

    def test_vague_cases_declare_threshold(self, cases: list[dict]):
        for c in cases:
            if c.get("expect_low_confidence"):
                assert 0.0 < c.get("confidence_threshold", 0) <= 1.0, (
                    f"{c['id']} 缺少合理的 confidence_threshold"
                )

    def test_difficulty_values_are_known(self, cases: list[dict]):
        allowed = {"clear", "tricky", "vague", "multi", "noise"}
        for c in cases:
            got = c.get("difficulty", "clear")
            assert got in allowed, f"{c['id']} 的 difficulty 取值非法: {got}"

    def test_has_both_tricky_and_vague_cases(self, cases: list[dict]):
        """只有简单用例的评测集会饱和（全部满分），区分不出 Prompt 好坏。"""
        kinds = Counter(c.get("difficulty", "clear") for c in cases)
        assert kinds["tricky"] >= 3, "缺少「根因辨析」难题，评测会饱和"
        assert kinds["vague"] >= 3, "缺少「信息不足」用例，无法考察置信度校准"

    def test_has_multi_log_cases(self, cases: list[dict]):
        """多日志用例用于度量「只取第一条 ERROR」这个实现缺陷的严重程度。"""
        kinds = Counter(c.get("difficulty", "clear") for c in cases)
        assert kinds["multi"] >= 4, "缺少多日志用例，无法度量上下文缺失问题"

    def test_has_noise_cases(self, cases: list[dict]):
        """长日志噪音用例用于区分聚合参数（窗口 K / 上限 M）的优劣。

        没有这类用例时，几组参数会得到完全相同的分数 —— 参数扫描就失去了意义。
        """
        kinds = Counter(c.get("difficulty", "clear") for c in cases)
        assert kinds["noise"] >= 3, "缺少长日志噪音用例，参数扫描会失去区分度"

    def test_noise_cases_are_long_enough(self, cases: list[dict]):
        for c in cases:
            if c.get("difficulty") != "noise":
                continue
            logs = parse_text("\n".join(c["log_lines"]))
            assert len(logs) >= 8, f"{c['id']} 只有 {len(logs)} 条日志，构不成噪音场景"

    def test_root_cause_keywords_never_leak_into_anchor(self, cases: list[dict]):
        """多日志 / 噪音用例的关键设计：真因关键词不能出现在**锚点**里。

        锚点是模型一定会看到的那条日志。如果真因关键词泄漏进锚点，
        那么「只送锚点」的旧实现也能命中 —— 这条用例就测不出问题了。
        断言在这里是为了防止用例被写废（我确实踩过一次：把 `stock` 当关键词，
        而锚点的类名正好是 `StockClient`）。
        """
        for c in cases:
            if c.get("difficulty") not in ("multi", "noise"):
                continue
            logs = parse_text("\n".join(c["log_lines"]))
            anchor = next(
                (l for l in logs if l.level in ("ERROR", "CRITICAL", "FATAL")),
                logs[0],
            )
            blob = f"{anchor.message} {anchor.stack_trace}".lower()
            leaked = [kw for kw in c["keywords"] if kw.lower() in blob]
            assert not leaked, (
                f"{c['id']} 的关键词 {leaked} 出现在锚点中，用例失去测度能力"
            )

    def test_multi_log_cases_have_multiple_entries(self, cases: list[dict]):
        for c in cases:
            if c.get("difficulty") != "multi":
                continue
            logs = parse_text("\n".join(c["log_lines"]))
            assert len(logs) >= 3, f"{c['id']} 日志条数太少（{len(logs)}），不构成多日志场景"
            levels = {l.level for l in logs}
            assert levels - {"ERROR", "CRITICAL", "FATAL"}, (
                f"{c['id']} 全是 ERROR，真因应该是更低级别的日志"
            )

    def test_keywords_are_lowercase_safe_strings(self, cases: list[dict]):
        """匹配时会转小写比较，所以关键词本身写大小写都行，但不能是空串。"""
        for c in cases:
            for kw in c["keywords"]:
                assert isinstance(kw, str) and kw.strip(), f"{c['id']} 有空白关键词"


class TestDefaultParamsCoverRootCause:
    """把「参数扫描的结论」固化成回归测试。

    默认的窗口 / 上限是花了一轮评测（38 条用例、约 5 分钟、真金白银）才选出来的。
    如果有人把参数调小，这条测试会**立刻**失败，
    而不是等到下次跑评测才发现 —— 那代价高得多。

    对应关系：关闭聚合 10% → ±2/8 60% → ±4/12 80% → **±8/20 100%**。
    """

    def test_default_context_params_include_root_cause_everywhere(self, cases, monkeypatch):
        from app.context import build_context

        # 强制使用代码里的默认值，不受本机 .env 影响
        monkeypatch.setenv("LOGWISE_CONTEXT_ENABLED", "true")
        monkeypatch.delenv("LOGWISE_CONTEXT_WINDOW", raising=False)
        monkeypatch.delenv("LOGWISE_CONTEXT_MAX_LOGS", raising=False)

        checked = 0
        for c in cases:
            if c.get("difficulty") not in ("multi", "noise"):
                continue
            logs = parse_text("\n".join(c["log_lines"]))
            context = build_context(logs)
            selected = {id(log) for log in context.logs}

            truth = [
                log for log in logs
                if any(kw.lower() in f"{log.message} {log.stack_trace}".lower()
                       for kw in c["keywords"])
            ]
            assert truth, f"{c['id']} 里找不到含真因关键词的日志（用例标注有问题）"
            for log in truth:
                assert id(log) in selected, (
                    f"{c['id']} 的真因未被默认聚合参数纳入："
                    f"真因「{log.message[:40]}」不在选中的 {context.size} 条日志里。"
                    f"如果你调小了 LOGWISE_CONTEXT_WINDOW / _MAX_LOGS，请重跑参数扫描。"
                )
            checked += 1

        assert checked >= 8, f"只检查了 {checked} 条上下文类用例，覆盖不足"
