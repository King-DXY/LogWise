"""测试共用的小工具。

文件名不以 test_ 开头，pytest 不会把它当测试用例收集 —— 只作为共享模块。
"""
from __future__ import annotations

import math
import re
import zlib


def fake_vector(text: str, dim: int = 64) -> list[float]:
    """确定性假向量：用 hashing trick，含相同词的文本会得到相似向量。

    用 ``crc32`` 而不是内置 ``hash()`` —— 后者对字符串是随机化的
    （受 PYTHONHASHSEED 影响），同一文本在不同进程里会得到不同向量，
    这会让"检索结果是否稳定"这类测试变得随机。

    这样造出来的假向量保留了"词面重叠 → 向量接近"的性质，
    足以验证检索链路是否正确，又不需要联网或花任何钱。
    """
    vec = [0.0] * dim
    for tok in re.findall(r"[a-z0-9\u4e00-\u9fa5]+", text.lower()):
        vec[zlib.crc32(tok.encode()) % dim] += 1.0
    norm = math.sqrt(sum(x * x for x in vec)) or 1.0
    return [x / norm for x in vec]
