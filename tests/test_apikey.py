"""API Key 占位符判定的测试。

这个文件的重点是**锁住「三处判定必须一致」**这个行为。

背景：判定逻辑原先散在 diagnose / embedding / run 三处，各自匹配不同的
中文字符串（"sk-在这里"、"在这里填你的智谱Key"）。实测把占位符换成另一种
写法就会被判成「已配置」，然后去调接口报一个和配置无关的难懂错误。
下面 ``TestConsistencyAcrossCallers`` 就是那次事故的回归。
"""

from __future__ import annotations

import pytest

from app import embedding
from app import diagnose
from app.apikey import is_placeholder, mask, read_api_key


class TestIsPlaceholder:
    @pytest.mark.parametrize("value", [
        None,
        "",
        "   ",
        "在这里填你的key",              # 裸中文占位符
        "sk-在这里填你的key",           # 旧写法
        "在这里填你的智谱Key",           # embedding 里的旧写法（.env.example 里其实不存在）
        "sk-xxx",
        "xxxxxxxx",
        "sk-your-key-here",
        "your-api-key",
        "sk-placeholder",
        "changeme",
        "<your-key>",
        "...",
    ])
    def test_detected_as_placeholder(self, value):
        assert is_placeholder(value) is True, f"{value!r} 应被判为占位符"

    @pytest.mark.parametrize("value", [
        "sk-" + "a1b2c3d4" * 5,          # 真实形态
        "c1d2e3f4" * 6,
        "sk-abc123",
        "abc",                            # 短，但看不出是占位符 → 不拦
        "0c9a8b7d6e5f4a3b2c1d0e9f8a7b6c5d",
    ])
    def test_real_looking_keys_pass(self, value):
        """宁可不拦，也不要把真 Key 拒掉 —— 误拒比漏拦更难排查。"""
        assert is_placeholder(value) is False, f"{value!r} 不该被判为占位符"

    def test_strips_whitespace_before_judging(self):
        assert is_placeholder("  sk-xxx  ") is True
        assert is_placeholder("  sk-abc123  ") is False


class TestReadApiKey:
    def test_missing_env_returns_none(self, monkeypatch):
        monkeypatch.delenv("TEST_API_KEY", raising=False)
        assert read_api_key("TEST_API_KEY") is None

    def test_placeholder_env_returns_none(self, monkeypatch):
        monkeypatch.setenv("TEST_API_KEY", "sk-your-key-here")
        assert read_api_key("TEST_API_KEY") is None

    def test_real_key_returned_stripped(self, monkeypatch):
        monkeypatch.setenv("TEST_API_KEY", "  sk-abc123def456  ")
        assert read_api_key("TEST_API_KEY") == "sk-abc123def456"


class TestMask:
    def test_never_exposes_full_key(self):
        key = "sk-" + "a1b2c3d4e5f6" * 3
        shown = mask(key)
        assert key not in shown
        assert shown.startswith("sk-")
        assert "****" in shown

    def test_short_key_all_starred(self):
        assert mask("abc") == "***"

    def test_short_key_does_not_leak_any_char(self):
        assert "a" not in mask("abcd")


class TestConsistencyAcrossCallers:
    """三处调用方必须用同一套判据 —— 这是那次「静默失效」事故的回归。"""

    # 这些值在修复前会被不同模块给出**互相矛盾**的结论
    PLACEHOLDERS = [
        "在这里填你的key",       # 旧代码：diagnose 放行（不以 sk-在这里 开头）
        "sk-在这里填你的key",
        "在这里填你的智谱Key",     # 旧代码：embedding 恰好挡对，但靠字符串巧合
        "sk-your-key-here",
        "",
        "   ",
    ]

    def test_diagnose_and_embedding_agree(self, monkeypatch):
        """同一个值，诊断侧和向量侧的结论必须一致。"""
        for value in self.PLACEHOLDERS:
            monkeypatch.setenv("DEEPSEEK_API_KEY", value)
            monkeypatch.setenv("EMBEDDING_API_KEY", value)

            try:
                diagnose._require_api_key()
                diagnose_says_configured = True
            except RuntimeError:
                diagnose_says_configured = False

            embedding_says_configured = embedding.embedding_configured()

            assert diagnose_says_configured == embedding_says_configured == False, (
                f"{value!r}: diagnose={diagnose_says_configured} "
                f"embedding={embedding_says_configured} —— 两侧判据不一致"
            )

    def test_chinese_variant_is_rejected(self, monkeypatch):
        """具体回归：这个变体曾经溜过 diagnose 的检查。"""
        monkeypatch.setenv("DEEPSEEK_API_KEY", "在这里填你的key")
        with pytest.raises(RuntimeError):
            diagnose._require_api_key()


class TestLauncherEnvCheck:
    """启动器读 .env 的那条路径 —— 用户最早遇到的提示就在这里。

    ``check_env_file()`` 把路径写死成 ``ROOT / ".env"``，所以测试里替换
    ``app.run.ROOT`` 指向临时目录，**不碰用户真实的 .env**。
    """

    @staticmethod
    def _check(monkeypatch, tmp_path, content: str | None):
        import app.run as run_mod

        (tmp_path / ".env").unlink(missing_ok=True)
        if content is not None:
            (tmp_path / ".env").write_text(content, encoding="utf-8")
        monkeypatch.setattr(run_mod, "ROOT", tmp_path)
        return run_mod.check_env_file()

    @pytest.mark.parametrize("value", [
        "sk-在这里填你的key",      # 旧写法
        "在这里填你的key",          # 变体 —— 修复前会溜过去
        "在那里-your-key",          # 另一种中文变体
        "sk-your-key-here",
        "",
        "   ",
    ])
    def test_placeholder_yields_none(self, monkeypatch, tmp_path, value):
        assert self._check(monkeypatch, tmp_path, f"DEEPSEEK_API_KEY={value}\n") is None

    def test_missing_env_file_yields_none(self, monkeypatch, tmp_path):
        assert self._check(monkeypatch, tmp_path, None) is None

    def test_real_key_is_returned(self, monkeypatch, tmp_path):
        key = "sk-" + "a" * 32
        assert self._check(monkeypatch, tmp_path, f"DEEPSEEK_API_KEY={key}\n") == key

    def test_short_but_valid_key_is_not_rejected(self, monkeypatch, tmp_path):
        """回归：旧代码用 ``len(key) < 20`` 猜长度，会把合法的短 Key 当成「没填」。

        长度不是可靠信号 —— 不同厂商的 Key 长度差别很大。现在只提醒、不拒绝。
        """
        short = "sk-shortkey12"
        assert self._check(monkeypatch, tmp_path, f"DEEPSEEK_API_KEY={short}\n") == short


class TestNoHardcodedChineseSentinel:
    """防止有人再把「匹配中文字符串」这种写法加回来。

    判据只看**代码里的字符串字面量**，不看注释和文档字符串 ——
    因为这个模块的 docstring 里就写着旧写法（作为「为什么改」的说明），
    全文件文本搜索会把说明文字当成违规，那是误报。
    """

    NEEDLES = ("sk-在这里", "在这里填你的")

    @staticmethod
    def _code_strings(path: Path) -> list[str]:
        """取出文件里所有「非 docstring」的字符串字面量。"""
        import ast

        tree = ast.parse(path.read_text(encoding="utf-8"))
        docstrings = set()
        for node in ast.walk(tree):
            if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
                body = getattr(node, "body", [])
                if (
                    body
                    and isinstance(body[0], ast.Expr)
                    and isinstance(body[0].value, ast.Constant)
                    and isinstance(body[0].value.value, str)
                ):
                    docstrings.add(id(body[0].value))
        return [
            n.value for n in ast.walk(tree)
            if isinstance(n, ast.Constant) and isinstance(n.value, str) and id(n) not in docstrings
        ]

    def test_no_chinese_sentinel_in_code(self):
        from pathlib import Path

        root = Path(__file__).resolve().parent.parent
        offenders = []
        for path in sorted((root / "app").glob("*.py")):
            if path.name == "apikey.py":
                continue  # 判定逻辑集中在这里，其他地方不该再出现
            for text in self._code_strings(path):
                for needle in self.NEEDLES:
                    if needle in text:
                        offenders.append(f"{path.name}: {text[:40]!r}")
        assert not offenders, (
            "判定占位符的字符串常量出现在代码里，改一处漏一处会静默失效："
            f" {offenders}（应统一用 app.apikey.is_placeholder）"
        )

    def test_the_checker_itself_works(self, tmp_path):
        """确认这条绊线真的会响 —— 否则它只是一句空话。"""
        from pathlib import Path as _P

        fake = _P(tmp_path) / "fake.py"
        fake.write_text(
            '"""文档字符串里出现 sk-在这里 不算违规。"""\n'
            'KEY = "sk-在这里"\n',
            encoding="utf-8",
        )
        found = [t for t in self._code_strings(fake) for n in self.NEEDLES if n in t]
        assert found, "绊线失效：连真实代码里的占位符都查不出来"
        # 同时确认 docstring 被正确排除
        assert not any("不算违规" in t for t in self._code_strings(fake))

