"""接口防护：API Key 鉴权 + 按来源 IP 的令牌桶限流。

设计取舍
--------
**鉴权默认「本机放行」**
    - 未配置 ``LOGWISE_API_KEY`` 时，鉴权整体关闭 —— 保证本地双击启动的演示页
      不用改一行就能继续用。
    - 配置了 Key 之后，来自回环地址（127.0.0.1 / ::1）的请求仍可豁免，
      由 ``LOGWISE_TRUST_LOCALHOST`` 控制（默认 true）。
    - 效果：本机演示不受影响，而部署到 K8s 后**外部流量必须带 Key**。
      注意容器里外部请求的源 IP 是 Pod/节点 IP，不会是 127.0.0.1，所以不会被误放行。

**限流对所有人生效**（包括本机）
    因为限流的目的就是控制成本，本地调用同样在花钱。

**全部用进程内存实现**
    单实例够用。多副本部署时应换成 Redis（见 README「已知坑」）。
"""
from __future__ import annotations

import hmac
import os
import threading
import time
from dataclasses import dataclass

from fastapi import HTTPException, Request

DEFAULT_RATE_LIMIT = 20      # 每个窗口允许的请求数
DEFAULT_RATE_WINDOW = 60.0   # 窗口长度（秒）
LOCAL_HOSTS = {"127.0.0.1", "::1", "localhost"}


# --------------------------------------------------------------------------- #
# 配置读取
# --------------------------------------------------------------------------- #
def _flag(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() not in ("0", "false", "no", "off", "")


def configured_api_key() -> str:
    return os.getenv("LOGWISE_API_KEY", "").strip()


def auth_enabled() -> bool:
    return bool(configured_api_key())


def rate_limit_enabled() -> bool:
    try:
        return int(os.getenv("LOGWISE_RATE_LIMIT", DEFAULT_RATE_LIMIT)) > 0
    except ValueError:
        return True


def rate_capacity() -> int:
    try:
        return max(1, int(os.getenv("LOGWISE_RATE_LIMIT", DEFAULT_RATE_LIMIT)))
    except ValueError:
        return DEFAULT_RATE_LIMIT


def rate_window() -> float:
    try:
        return max(1.0, float(os.getenv("LOGWISE_RATE_WINDOW", DEFAULT_RATE_WINDOW)))
    except ValueError:
        return DEFAULT_RATE_WINDOW


def client_ip(request: Request) -> str:
    """取来源 IP。

    只有在显式开启 ``LOGWISE_TRUST_PROXY_HEADERS`` 时才采信 X-Forwarded-For，
    否则它可以被客户端随意伪造，等于把限流和本机豁免都变成摆设。
    """
    if _flag("LOGWISE_TRUST_PROXY_HEADERS", False):
        forwarded = request.headers.get("x-forwarded-for", "")
        if forwarded:
            return forwarded.split(",")[0].strip()
    return request.client.host if request.client else "unknown"


def is_localhost(request: Request) -> bool:
    return client_ip(request) in LOCAL_HOSTS


# --------------------------------------------------------------------------- #
# 鉴权
# --------------------------------------------------------------------------- #
async def require_api_key(request: Request) -> None:
    """FastAPI 依赖：校验 X-API-Key 请求头。"""
    if not auth_enabled():
        return
    if _flag("LOGWISE_TRUST_LOCALHOST", True) and is_localhost(request):
        return

    provided = request.headers.get("x-api-key", "")
    # 用固定时间比较，避免通过响应时间差逐字节猜出 Key
    if not provided or not hmac.compare_digest(provided, configured_api_key()):
        raise HTTPException(
            status_code=401,
            detail="缺少或无效的 API Key（请在请求头带上 X-API-Key）",
            headers={"WWW-Authenticate": "X-API-Key"},
        )


# --------------------------------------------------------------------------- #
# 限流（令牌桶）
# --------------------------------------------------------------------------- #
@dataclass
class _Bucket:
    tokens: float
    updated: float
    last_seen: float


class TokenBucketLimiter:
    """按 key 分桶的令牌桶限流器。

    令牌桶相对「固定窗口计数」的好处：允许短时突发，但长期速率被严格限制，
    不会出现「窗口边界瞬间放行两倍请求」的问题。
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._buckets: dict[str, _Bucket] = {}

    def _params(self) -> tuple[int, float]:
        capacity = rate_capacity()
        window = rate_window()
        return capacity, capacity / window  # 每秒补充的令牌数

    def allow(self, key: str) -> tuple[bool, float]:
        """返回 (是否放行, 建议等待秒数)。"""
        capacity, refill = self._params()
        now = time.monotonic()

        with self._lock:
            self._evict(now)
            bucket = self._buckets.get(key)
            if bucket is None:
                bucket = _Bucket(tokens=float(capacity), updated=now, last_seen=now)
                self._buckets[key] = bucket

            elapsed = max(0.0, now - bucket.updated)
            bucket.tokens = min(float(capacity), bucket.tokens + elapsed * refill)
            bucket.updated = now
            bucket.last_seen = now

            if bucket.tokens >= 1.0:
                bucket.tokens -= 1.0
                return True, 0.0

            need = (1.0 - bucket.tokens) / refill if refill > 0 else rate_window()
            return False, need

    def _evict(self, now: float) -> None:
        """清掉长时间不活跃的桶，避免内存无限增长。调用方需持有锁。"""
        ttl = rate_window() * 4
        if len(self._buckets) < 64:
            return
        for key in [k for k, b in self._buckets.items() if now - b.last_seen > ttl]:
            self._buckets.pop(key, None)

    def snapshot(self) -> dict:
        with self._lock:
            return {"active_clients": len(self._buckets)}


limiter = TokenBucketLimiter()


async def rate_limit(request: Request) -> None:
    """FastAPI 依赖：按来源 IP 限流。"""
    if not rate_limit_enabled():
        return

    ip = client_ip(request)
    allowed, retry_after = limiter.allow(ip)
    if not allowed:
        raise HTTPException(
            status_code=429,
            detail=(
                f"请求过于频繁：每 {int(rate_window())} 秒最多 {rate_capacity()} 次。"
                f"请等待约 {int(retry_after) + 1} 秒后重试。"
            ),
            headers={"Retry-After": str(int(retry_after) + 1)},
        )


# --------------------------------------------------------------------------- #
# 供启动时打印的安全状态摘要
# --------------------------------------------------------------------------- #
def startup_notice() -> str:
    if not auth_enabled():
        return (
            "[提示] 未配置 LOGWISE_API_KEY，接口鉴权已关闭（适合本机演示）。\n"
            "       若要部署到公网，请在 .env 里设置 LOGWISE_API_KEY。"
        )
    trust = "开启（本机请求免 Key）" if _flag("LOGWISE_TRUST_LOCALHOST", True) else "关闭（所有请求都要 Key）"
    return (
        f"[安全] 接口鉴权已开启；本机豁免：{trust}\n"
        f"       限流：每 {int(rate_window())} 秒最多 {rate_capacity()} 次 / 每 IP"
    )
