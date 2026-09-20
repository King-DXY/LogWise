"""诊断引擎的健壮性测试：输出结构校验、格式重试、超时配置。

为什么值得单独测
----------------
实测遇到过一次接口 502：模型没有返回 JSON，而是回了一句大白话
（"这段日志信息不足，我无法给出结构化诊断"），``json.loads`` 直接抛
``JSONDecodeError: Expecting value: line 1 column 1 (char 0)``，
这个异常原文被原样甩给了用户 —— 既没重试，也看不懂。

修复后的行为是：**解析 → 校验结构 → 不合格就把问题反馈给模型重试一次**，
仍然不合格才抛一个可读的 ``DiagnosisFormatError``。

另一半是超时：SDK 默认 read=600 秒 + 2 次重试 ≈ 最坏挂 30 分钟，
而实测一次诊断只要约 2 秒。必须显式收窄。
"""
from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from app import diagnose
from app.parser import ParsedLog

GOOD = {
    "error_type": "磁盘空间不足",
    "root_cause": "日志分区写满，写入被拒",
    "evidence": ["java.io.IOException: No space left on device"],
    "impact": "指标落盘失败，监控数据缺失",
    "solution_steps": ["df -h 定位分区", "清理历史日志"],
    "confidence": 0.9,
}
PROSE = "抱歉，这段日志信息不足，我无法给出结构化诊断。"


def _resp(content: str, prompt: int = 100, completion: int = 50, cached: int = 0):
    return SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=content))],
        usage=SimpleNamespace(
            prompt_tokens=prompt,
            completion_tokens=completion,
            prompt_tokens_details=SimpleNamespace(cached_tokens=cached),
        ),
    )


class FakeClient:
    """按预设脚本依次返回内容；内容用完后重复最后一条。"""

    def __init__(self, contents: list[str]):
        self.contents = contents
        self.calls = 0
        self.seen_messages: list[list[dict]] = []
        outer = self

        class _Completions:
            def create(self, **kwargs):
                outer.seen_messages.append(list(kwargs["messages"]))
                idx = min(outer.calls, len(outer.contents) - 1)
                outer.calls += 1
                return _resp(outer.contents[idx])

        self.chat = SimpleNamespace(completions=_Completions())


@pytest.fixture
def err_log() -> ParsedLog:
    return ParsedLog(
        timestamp="2026-09-20 10:00:00,000",
        level="ERROR",
        module="metrics-writer",
        message="指标落盘失败",
    )


@pytest.fixture
def use_fake(monkeypatch):
    def _install(contents: list[str]) -> FakeClient:
        fake = FakeClient(contents)
        monkeypatch.setattr(diagnose, "get_client", lambda: fake)
        return fake

    return _install


# --------------------------------------------------------------------------- #
# JSON 解析：尽量宽容，但失败必须抛
# --------------------------------------------------------------------------- #
class TestParseJson:
    def test_bare_object(self):
        assert diagnose._parse_json(json.dumps(GOOD)) == GOOD

    def test_code_fence_lowercase(self):
        raw = "```json\n" + json.dumps(GOOD) + "\n```"
        assert diagnose._parse_json(raw) == GOOD

    def test_code_fence_uppercase(self):
        """大小写要兼容 —— 模型经常写成 ```JSON。"""
        raw = "```JSON\n" + json.dumps(GOOD) + "\n```"
        assert diagnose._parse_json(raw) == GOOD

    def test_object_with_prose_prefix(self):
        raw = "分析如下：\n" + json.dumps(GOOD) + "\n以上。"
        assert diagnose._parse_json(raw) == GOOD

    def test_raises_on_prose(self):
        with pytest.raises(json.JSONDecodeError):
            diagnose._parse_json(PROSE)

    def test_raises_on_empty(self):
        with pytest.raises(json.JSONDecodeError):
            diagnose._parse_json("")

    def test_single_element_array_is_salvaged(self):
        """模型偶尔会把结果包成 ``[{...}]``。截取首尾大括号能救回来，这是有意为之。"""
        assert diagnose._parse_json('[{"error_type": "x"}]') == {"error_type": "x"}

    def test_multi_element_array_still_raises(self):
        """但多个对象拼在一起就不是一个合法对象了，救不回来也不能硬凑。"""
        with pytest.raises(json.JSONDecodeError):
            diagnose._parse_json('[{"a": 1}, {"b": 2}]')


# --------------------------------------------------------------------------- #
# 结构校验
# --------------------------------------------------------------------------- #
class TestSchemaProblems:
    def test_good_result_has_no_problem(self):
        assert diagnose.schema_problems(GOOD) == []

    def test_missing_fields_are_reported(self):
        problems = diagnose.schema_problems({"error_type": "x"})
        assert problems and "缺少字段" in problems[0]

    def test_non_dict_is_reported(self):
        assert diagnose.schema_problems([1, 2])
        assert diagnose.schema_problems(None)

    def test_wrong_list_types_are_reported(self):
        bad = {**GOOD, "solution_steps": "不是数组"}
        assert any("数组" in p for p in diagnose.schema_problems(bad))

    def test_confidence_out_of_range_is_reported(self):
        assert any("0~1" in p for p in diagnose.schema_problems({**GOOD, "confidence": 1.5}))
        assert any("数字" in p for p in diagnose.schema_problems({**GOOD, "confidence": "高"}))

    def test_bool_is_not_a_valid_confidence(self):
        """True 在 Python 里是 int —— 但当成 confidence 显然是错的。"""
        assert diagnose.schema_problems({**GOOD, "confidence": True})

    def test_evidence_is_optional(self):
        """evidence 是 v3 新增的，缺失可以接受（旧 Prompt / 旧缓存）。"""
        without = {k: v for k, v in GOOD.items() if k != "evidence"}
        assert diagnose.schema_problems(without) == []


# --------------------------------------------------------------------------- #
# 重试
# --------------------------------------------------------------------------- #
class TestFormatRetry:
    def test_recovers_from_prose(self, err_log, use_fake):
        fake = use_fake([PROSE, json.dumps(GOOD)])
        result, _ = diagnose.diagnose_multi_ex([err_log])
        assert result == GOOD
        assert fake.calls == 2, "第一次不合格，应该重试一次"

    def test_recovers_from_incomplete_json(self, err_log, use_fake):
        fake = use_fake(['{"error_type": 1}', json.dumps(GOOD)])
        result, _ = diagnose.diagnose_multi_ex([err_log])
        assert result == GOOD
        assert fake.calls == 2

    def test_no_retry_when_first_attempt_is_good(self, err_log, use_fake):
        fake = use_fake([json.dumps(GOOD)])
        diagnose.diagnose_multi_ex([err_log])
        assert fake.calls == 1, "合格就不该多花一次钱"

    def test_correction_prompt_names_the_problem(self, err_log, use_fake):
        fake = use_fake([PROSE, json.dumps(GOOD)])
        diagnose.diagnose_multi_ex([err_log])
        last = fake.seen_messages[1]
        assert last[-1]["role"] == "user"
        assert "不是合法 JSON" in last[-1]["content"]
        assert last[-2]["role"] == "assistant", "要把上一次的回复也带回去"

    def test_raises_readable_error_after_exhausting_retries(self, err_log, use_fake):
        use_fake([PROSE])  # 一直不合格
        with pytest.raises(diagnose.DiagnosisFormatError) as exc:
            diagnose.diagnose_multi_ex([err_log])
        msg = str(exc.value)
        assert "2 次" in msg            # 说清了试了几次
        assert "不是合法 JSON" in msg     # 说清了哪里不合格
        assert "无法给出结构化诊断" in msg  # 带上了模型的原话，便于判断

    def test_retries_zero_means_single_attempt(self, err_log, use_fake):
        fake = use_fake([PROSE])
        with pytest.raises(diagnose.DiagnosisFormatError):
            diagnose.diagnose_multi_ex([err_log], retries=0)
        assert fake.calls == 1

    def test_usage_is_summed_across_retries(self, err_log, use_fake):
        """重试花的钱也是真花的，不能只记最后一次。"""
        use_fake([PROSE, json.dumps(GOOD)])
        _, usage = diagnose.diagnose_multi_ex([err_log])
        assert usage.prompt_tokens == 200, "两次各 100，应该累加"
        assert usage.completion_tokens == 100


# --------------------------------------------------------------------------- #
# 超时与重试配置
# --------------------------------------------------------------------------- #
class TestClientOptions:
    def test_read_timeout_is_finite_and_sane(self, monkeypatch):
        monkeypatch.delenv("LOGWISE_LLM_TIMEOUT", raising=False)
        assert diagnose.read_timeout() == diagnose.DEFAULT_READ_TIMEOUT
        # SDK 默认是 600 秒，必须比它小得多才有意义
        assert diagnose.read_timeout() < 120

    def test_timeout_from_env(self, monkeypatch):
        monkeypatch.setenv("LOGWISE_LLM_TIMEOUT", "12.5")
        assert diagnose.read_timeout() == 12.5

    def test_invalid_timeout_falls_back(self, monkeypatch):
        monkeypatch.setenv("LOGWISE_LLM_TIMEOUT", "abc")
        assert diagnose.read_timeout() == diagnose.DEFAULT_READ_TIMEOUT
        monkeypatch.setenv("LOGWISE_LLM_TIMEOUT", "-5")
        assert diagnose.read_timeout() == diagnose.DEFAULT_READ_TIMEOUT

    def test_max_retries_bounded(self, monkeypatch):
        monkeypatch.setenv("LOGWISE_LLM_MAX_RETRIES", "99")
        assert diagnose.max_retries() == 5
        monkeypatch.setenv("LOGWISE_LLM_MAX_RETRIES", "-3")
        assert diagnose.max_retries() == 0

    def test_client_options_shape(self, monkeypatch):
        monkeypatch.setenv("LOGWISE_LLM_TIMEOUT", "30")
        monkeypatch.setenv("LOGWISE_LLM_CONNECT_TIMEOUT", "3")
        opts = diagnose.client_options()
        assert opts["max_retries"] == diagnose.max_retries()
        assert opts["timeout"].read == 30
        assert opts["timeout"].connect == 3
