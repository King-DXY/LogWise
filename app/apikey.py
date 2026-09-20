"""API Key 的读取与「占位符」判定。

为什么单独成一个模块
--------------------
这段判定原先散在三个地方，而且各自匹配**不同的中文字符串**：

    app/diagnose.py    key.startswith("sk-在这里")
    app/embedding.py   key != "在这里填你的智谱Key"   ← 这个字符串 .env.example 里并不存在
    app/run.py         key.startswith("sk-在这里") or len(key) < 20

问题有两个：

1. **改一处忘另一处就静默失效。** 实测：把占位符换成另一种写法
   （如 ``在这里填你的key``）就会被判成「已配置」，然后去调接口、
   报一个和配置毫无关系的难懂错误。
2. ``len(key) < 20`` 会**误伤合法的短 Key** —— 长度不是可靠信号，
   不同厂商的 Key 长度差别很大。

现在的判据
----------
**含非 ASCII 字符即为占位符。** 真实 API Key 只用 ASCII 可打印字符，
所以这条规则**不需要维护任何字符串常量**，将来把占位符和注释改成英文也不会失效。
另外再挡掉几种常见的英文占位符写法（``sk-xxx`` / ``sk-your-key`` / ``<your-key>``）。
"""

from __future__ import annotations

import os
import re

# 只挡「明显是占位符」的写法。宁可不挡，也不要把真 Key 拒掉 ——
# 真 Key 顶多让接口返回 401，而误拒一个合法 Key 会让用户完全摸不着头脑。
_PLACEHOLDER_PATTERNS = (
    re.compile(r"^(sk-)?x{3,}$", re.I),                          # sk-xxx / xxxxxxxx
    re.compile(r"^(sk-)?your[-_].+$", re.I),                     # sk-your-key-here
    re.compile(r"^(sk-)?(placeholder|changeme|change[-_]me|todo)$", re.I),
    re.compile(r"^<.+>$"),                                       # <your-key>
    re.compile(r"^\.{3,}$"),                                     # ...
)


def is_placeholder(value: str | None) -> bool:
    """这个配置值是否属于「还没填」的状态。"""
    if value is None:
        return True
    text = value.strip()
    if not text:
        return True
    if not text.isascii():
        # 中文（或任何非 ASCII）占位符，例如「sk-在这里填你的key」。
        # 真实 Key 不会出现非 ASCII 字符。
        return True
    return any(p.match(text) for p in _PLACEHOLDER_PATTERNS)


def read_api_key(env_name: str) -> str | None:
    """读环境变量里的 Key；缺失、为空、或是占位符都返回 ``None``。"""
    return None if is_placeholder(os.getenv(env_name)) else os.getenv(env_name, "").strip()


def mask(key: str, head: int = 6, tail: int = 4) -> str:
    """给日志用的脱敏展示，绝不打印完整 Key。"""
    if len(key) <= head + tail:
        return "*" * len(key)
    return f"{key[:head]}****{key[-tail:]}"
