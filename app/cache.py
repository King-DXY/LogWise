"""诊断结果缓存：同一段日志不重复花钱。

缓存键的组成
------------
``sha256(归一化日志文本 + 模型名 + Prompt 版本)``

为什么要带上后两项：
- **模型名**：换模型后结果会变，不带上就会返回旧模型算出的结论。
- **Prompt 版本**：改了 Prompt 同理。这是缓存类功能最常见的坑 ——
  改了提示词却忘了让缓存失效，结果线上跑的还是旧逻辑的结果。

为什么要归一化日志文本：
- 用户从各处复制日志时，行尾空格、连续空行、缩进差异很常见。归一化后
  「同一段日志」才能稳定命中同一份缓存。
"""
from __future__ import annotations

import hashlib
import os
import re
import threading
import time
from collections import OrderedDict

DEFAULT_TTL = 3600.0   # 秒
DEFAULT_SIZE = 500     # 最多缓存多少条

_WHITESPACE = re.compile(r"[ \t\u3000]+")


def _flag(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() not in ("0", "false", "no", "off", "")


def cache_enabled() -> bool:
    return _flag("LOGWISE_CACHE_ENABLED", True)


def _ttl() -> float:
    try:
        return float(os.getenv("LOGWISE_CACHE_TTL", DEFAULT_TTL))
    except ValueError:
        return DEFAULT_TTL


def _max_size() -> int:
    try:
        return max(1, int(os.getenv("LOGWISE_CACHE_SIZE", DEFAULT_SIZE)))
    except ValueError:
        return DEFAULT_SIZE


def normalize(text: str) -> str:
    """归一化日志文本，用于生成缓存键。

    做三件事：
    1. 统一换行符（CRLF / CR → LF）
    2. 逐行折叠行内空白（多个空格/制表符 → 一个）并去掉行首行尾空白
    3. 丢弃空行

    注意：**这只影响缓存键的计算，发给模型的仍然是原始文本**。
    所以这里可以放心做得激进一些 —— 用户从不同地方复制同一段日志时，
    行尾空格、缩进差异、多出来的空行都不应该导致缓存失效。
    """
    unified = text.replace("\r\n", "\n").replace("\r", "\n")
    lines = []
    for raw_line in unified.split("\n"):
        line = _WHITESPACE.sub(" ", raw_line).strip()
        if line:
            lines.append(line)
    return "\n".join(lines)


def make_key(
    text: str,
    model: str = "",
    prompt_version: str = "",
    context_signature: str = "",
    rag_signature: str = "",
) -> str:
    """生成缓存键。

    ``context_signature`` 是上下文聚合参数的签名（是否启用 / 窗口大小 / 条数上限）。
    ``rag_signature`` 是检索配置与检索结果的指纹（模式 / top-k / 命中的案例 id）。

    为什么要带上它们：改参数之后，同样的输入会产生不同的上下文包与不同的参考案例，
    进而产生不同的诊断结论。不纳入缓存键的话，调参会以为「参数改了没效果」——
    实际是命中了旧缓存。这是缓存类功能很容易踩的坑，
    而且**在 RAG 上更隐蔽**：检索结果本身会随知识库变化，
    所以指纹里放的是「检索到了哪几条」而不是「配置是什么」。
    """
    payload = "\x00".join([model, prompt_version, context_signature, rag_signature, normalize(text)])
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


class DiagnosisCache:
    """带 TTL 与 LRU 淘汰的线程安全内存缓存。"""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._data: OrderedDict[str, tuple[float, str]] = OrderedDict()
        self._hits = 0
        self._misses = 0
        self._evictions = 0

    def get(self, key: str) -> str | None:
        # 开关必须在这里生效 —— 之前只有 snapshot() 读了它，
        # 于是 LOGWISE_CACHE_ENABLED=false 只是"显示成关闭"，实际照样缓存，
        # 等于一个不起作用的配置项。
        if not cache_enabled():
            return None
        now = time.time()
        with self._lock:
            item = self._data.get(key)
            if item is None:
                self._misses += 1
                return None

            expire_at, value = item
            if expire_at <= now:
                # 过期即视为未命中，顺手清掉
                self._data.pop(key, None)
                self._misses += 1
                return None

            self._data.move_to_end(key)  # LRU：命中的挪到队尾
            self._hits += 1
            return value

    def set(self, key: str, value: str) -> None:
        if not cache_enabled():
            return
        now = time.time()
        with self._lock:
            self._data[key] = (now + _ttl(), value)
            self._data.move_to_end(key)
            while len(self._data) > _max_size():
                self._data.popitem(last=False)
                self._evictions += 1

    def clear(self) -> None:
        """只清缓存数据，**保留**命中/未命中计数。

        想连统计一起清（做对照实验）请用 ``reset()``。
        """
        with self._lock:
            self._data.clear()

    def reset(self) -> None:
        """清空缓存数据 + 命中/未命中/淘汰计数。

        ``/stats/reset`` 用的是这个 —— 它的返回消息说的是「统计与缓存已清零」，
        而原来只调了 ``clear()``，缓存那几项计数其实没清，做对照实验时会看到
        上一轮的 hits/misses 混在里面。
        """
        with self._lock:
            self._data.clear()
            self._hits = 0
            self._misses = 0
            self._evictions = 0

    def snapshot(self) -> dict:
        with self._lock:
            total = self._hits + self._misses
            return {
                "enabled": cache_enabled(),
                "size": len(self._data),
                "max_size": _max_size(),
                "ttl_seconds": _ttl(),
                "hits": self._hits,
                "misses": self._misses,
                "evictions": self._evictions,
                "hit_rate": round(self._hits / total, 4) if total else 0.0,
            }


# 全局单例：整个进程共用一个缓存
cache = DiagnosisCache()


def chunk_text(text: str, size: int = 48) -> list[str]:
    """把缓存命中的结果切成小块，让 SSE 前端仍然有逐字出现的效果。

    直接一次性返回整段会让界面「啪」地全亮，和后端真实流式的观感不一致，
    演示时容易让人以为流式失效了。
    """
    return [text[i : i + size] for i in range(0, len(text), size)] or [""]
