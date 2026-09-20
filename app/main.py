"""LogWise - LLM 智能日志诊断助手

阶段 4：在原有解析 / 诊断 / SSE 之上，补上运行时度量与接口防护。

防护策略（实现见 app/guard.py、app/cache.py）
---------------------------------------------
- **鉴权**：配置 LOGWISE_API_KEY 后生效；本机回环地址可豁免（保证本地演示不受影响）
- **限流**：按来源 IP 令牌桶，对所有业务接口生效，超限返回 429
- **缓存**：同一段日志在 TTL 内不重复调用模型；缓存键包含模型名与 Prompt 版本

度量：每次请求都会记录耗时、token 用量与估算成本，聚合结果见 ``GET /stats``。
"""
from __future__ import annotations

import json
import time
from pathlib import Path

from fastapi import Depends, FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from starlette.concurrency import run_in_threadpool

from app import cache as cache_mod
from app import context as context_mod
from app import guard, metrics, rag
from app.context import LogContext, build_context, describe
from app.diagnose import (
    PROMPT_VERSION,
    Usage,
    current_model,
    diagnose_multi_ex,
    format_logs,
    prompt_char_size,
    stream_diagnosis_multi,
)
from app.parser import ParsedLog, parse_text

BASE_DIR = Path(__file__).resolve().parent.parent

app = FastAPI(
    title="LogWise",
    description="LLM 智能日志诊断助手",
    version="0.4.0",
)

# 业务接口统一挂上「鉴权 + 限流」两个依赖。
# /health 故意不加：K8s 的 readiness/liveness 探针必须能无条件访问。
PROTECTED = [Depends(guard.require_api_key), Depends(guard.rate_limit)]

SSE_DONE = "data: [DONE]\n\n"
NO_LOG_MSG = "未解析到任何日志"

# /context 接口最多展示多少条日志。
# 只影响「前端展示」，不影响实际送给模型的日志（那由 app.context 决定）。
# 实测 2 万条日志的响应体约 4MB、前端要渲染 2 万行 DOM，浏览器会卡；
# 而 500 行既能覆盖绝大多数实际粘贴量，渲染也毫无压力。
MAX_CONTEXT_DISPLAY = 500


class ParseRequest(BaseModel):
    text: str


class DiagnoseRequest(BaseModel):
    text: str


class TextRequest(BaseModel):
    """通用文本请求体。

    存在的理由：把日志放在 URL query string 里有长度上限
    （实测约 32K 字符，再长会被 ASGI 服务器直接以 400 拒绝），
    而日志往往很长。POST 版接口用 JSON body 传，没有这个问题。
    """

    text: str


# --------------------------------------------------------------------------- #
# 基础接口
# --------------------------------------------------------------------------- #
# 接口契约版本。改了路由签名或响应结构就 +1 —— 前端拿它做「服务端是不是旧版」的握手。
# 用 /health 而不是某个业务接口来做这件事，是因为它不计度量、也不受限流，
# 前端每次刷新页面探测一次不会污染 /stats 的请求数，也不会吃掉限流额度。
API_REVISION = "2"


@app.get("/health")
def health():
    """健康检查（K8s 探针使用）。故意不加鉴权、不计度量、不受限流。"""
    return {"status": "ok", "api": API_REVISION}


@app.get("/stats", dependencies=PROTECTED)
def stats():
    """运行时度量：请求量、耗时、token 用量、估算成本、缓存命中率。"""
    return {
        "model": current_model(),
        "prompt_version": PROMPT_VERSION,
        "context": {
            "enabled": context_mod.context_enabled(),
            "window": context_mod.context_window(),
            "max_logs": context_mod.context_max_logs(),
            "note": "决定把哪些日志送给模型；参数已纳入缓存键",
        },
        "rag": rag.describe(),
        "metrics": metrics.store.summary(),
        "cache": cache_mod.cache.snapshot(),
        "rate_limit": guard.limiter.snapshot(),
    }


@app.get("/context", dependencies=PROTECTED)
def context_preview(text: str):
    """预览上下文聚合结果（GET 版）。

    **不调用大模型，因此不产生任何费用** —— 前端用它把「聚合过程」显示出来，
    否则用户只能看到一份诊断 JSON，完全感知不到上下文聚合在起作用。

    ⚠️ 日志较长时请用 POST 版，query string 有长度上限。
    """
    started = time.perf_counter()
    payload = _context_payload(text)
    # 也要计入 /stats 的请求数 —— 否则"请求总数"漏掉这一类，口径不完整。
    _record("/context", started)  # llm_call 默认 False
    return payload


@app.post("/context", dependencies=PROTECTED)
def context_preview_post(req: TextRequest):
    """预览上下文聚合结果（POST 版，无长度限制）。"""
    started = time.perf_counter()
    payload = _context_payload(req.text)
    _record("/context", started)
    return payload


def _context_payload(text: str) -> dict:
    logs = parse_text(text)
    if not logs:
        return {"error": NO_LOG_MSG}

    ctx = build_context(logs)
    selected_ids = {id(log) for log in ctx.logs}

    # 决定「展示」哪些日志。
    # 为什么不无脑全返回：实测 2 万条日志的响应体接近 4MB，前端要渲染同样多的 DOM 行，
    # 浏览器会明显卡顿甚至卡死。而用户真正关心的只有「锚点附近」和「被选中的」那些。
    # 注意这只影响展示，不影响实际送给模型的日志（那由 build_context 决定）。
    display_limited = len(logs) > MAX_CONTEXT_DISPLAY
    if display_limited:
        half = MAX_CONTEXT_DISPLAY // 2
        lo = max(0, min(ctx.anchor_index - half, len(logs) - MAX_CONTEXT_DISPLAY))
        hi = lo + MAX_CONTEXT_DISPLAY
        keep = set(range(lo, hi)) | {
            i for i, log in enumerate(logs) if id(log) in selected_ids
        }
        display_indices = sorted(keep)
    else:
        display_indices = list(range(len(logs)))

    return {
        "total_parsed": ctx.total_parsed,
        "selected": ctx.size,
        "anchor_index": ctx.anchor_index,
        "truncated": ctx.truncated,
        "enabled": ctx.enabled,
        "window": context_mod.context_window() if ctx.enabled else 0,
        "max_logs": context_mod.context_max_logs() if ctx.enabled else 1,
        "display_limited": display_limited,
        "displayed": len(display_indices),
        "logs": [
            {
                "index": i,
                "level": logs[i].level,
                "module": logs[i].module,
                "message": logs[i].message,
                "is_anchor": i == ctx.anchor_index,
                "selected": id(logs[i]) in selected_ids,
            }
            for i in display_indices
        ],
    }


@app.post("/stats/reset", dependencies=PROTECTED)
def stats_reset():
    """清零统计。

    用途：做对照实验 —— 清零后跑一轮，看到的数字就是这一轮的真实数据，
    不会被之前测试请求的累计值干扰。
    """
    metrics.store.reset()
    cache_mod.cache.reset()
    return {"status": "ok", "message": "统计与缓存已清零"}


# --------------------------------------------------------------------------- #
# 日志解析（不调用模型，不产生费用）
# --------------------------------------------------------------------------- #
@app.post("/parse", dependencies=PROTECTED)
def parse(req: ParseRequest):
    """解析日志，返回结构化字段列表。"""
    started = time.perf_counter()
    logs = parse_text(req.text)
    metrics.store.record(
        metrics.CallRecord(
            endpoint="/parse",
            duration_ms=(time.perf_counter() - started) * 1000,
        )
    )
    return {"logs": [log.__dict__ for log in logs]}


# --------------------------------------------------------------------------- #
# 诊断（一次性返回）
# --------------------------------------------------------------------------- #
@app.post("/diagnose", dependencies=PROTECTED)
def diagnose(req: DiagnoseRequest):
    """解析 + 诊断，一次性返回 JSON。命中缓存时不调用模型。"""
    started = time.perf_counter()

    logs = parse_text(req.text)
    if not logs:
        _record("/diagnose", started, error=NO_LOG_MSG)
        return {"error": NO_LOG_MSG}

    target = _pick_target(logs)
    context = build_context(logs)
    key = _cache_key(req.text, context)

    cached_text = cache_mod.cache.get(key)
    if cached_text is not None:
        _record("/diagnose", started, cache_hit=True)
        return {
            "log": target.__dict__,
            "context": describe(context),
            "diagnosis": json.loads(cached_text),
            "cached": True,
        }

    error = ""
    result: dict = {}
    usage = Usage()
    try:
        result, usage = diagnose_multi_ex(context.logs)
    except Exception as exc:  # 模型侧任何异常都不该让接口 500 裸奔
        error = f"{type(exc).__name__}: {exc}"

    # llm_call=True：不管成功还是抛异常，这次是真的打到模型了。
    # 但失败不产生 token，所以 cost 为 0，不会被计入 billable_calls。
    _record("/diagnose", started, usage=usage, error=error, llm_call=True)

    if error:
        raise HTTPException(status_code=502, detail=f"模型调用失败：{error}")

    # 只缓存模型输出，日志结构本身由解析器决定，没必要一起存
    cache_mod.cache.set(key, json.dumps(result, ensure_ascii=False))
    return {
        "log": target.__dict__,
        "context": describe(context),
        "diagnosis": result,
        "cached": False,
    }


# --------------------------------------------------------------------------- #
# 诊断（SSE 流式）
# --------------------------------------------------------------------------- #
@app.get("/diagnose/stream", dependencies=PROTECTED)
async def diagnose_stream_get(text: str):
    """SSE 流式诊断（GET 版）。

    ⚠️ 日志较长时请改用 POST 版：query string 有长度上限
    （实测约 32K 字符，再长会被 ASGI 服务器以 400 拒绝）。
    """
    return await run_in_threadpool(_stream_response, text)


@app.post("/diagnose/stream", dependencies=PROTECTED)
async def diagnose_stream_post(req: TextRequest):
    """SSE 流式诊断（POST 版）。日志放在 JSON body 里，没有长度限制。"""
    return await run_in_threadpool(_stream_response, req.text)


def _stream_response(text: str) -> StreamingResponse:
    """SSE 流式诊断：逐 token 返回诊断 JSON 文本。

    ⚠️ 这个函数是**同步且可能阻塞**的：``_cache_key`` 为了算检索结果指纹会跑一次
    检索，而后端是 dense/hybrid 时会发起 embedding 网络调用。
    所以两个 async 端点都用 ``run_in_threadpool`` 调它 —— 直接在事件循环里调用
    会阻塞整个服务，表现和当初踩过的「SSE 攒到最后一次性输出」是同一个症状。
    """
    started = time.perf_counter()

    logs = parse_text(text)
    if not logs:
        _record("/diagnose/stream", started, error=NO_LOG_MSG)
        return StreamingResponse(
            iter([_sse(json.dumps({"error": NO_LOG_MSG}, ensure_ascii=False)), SSE_DONE]),
            media_type="text/event-stream",
        )

    target = _pick_target(logs)
    context = build_context(logs)
    key = _cache_key(text, context)

    # ---- 缓存命中：把完整结果切块吐出去，保持"逐字出现"的观感 ----
    cached_text = cache_mod.cache.get(key)
    if cached_text is not None:
        _record("/diagnose/stream", started, cache_hit=True)

        async def replay():
            for piece in cache_mod.chunk_text(cached_text):
                yield _sse(piece)
            yield SSE_DONE

        return StreamingResponse(replay(), media_type="text/event-stream")

    # ---- 未命中：真正调用模型 ----
    usage = Usage()

    async def gen():
        collected: list[str] = []
        error = ""
        estimated = False
        try:
            async for delta in stream_diagnosis_multi(context.logs, usage):
                collected.append(delta)
                yield _sse(delta)
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
            yield _sse(f"[诊断出错] {error}")
        finally:
            # 放在 finally：即使客户端中途断开，也要留下度量记录
            full_text = "".join(collected)
            if not error and full_text:
                cache_mod.cache.set(key, _canonical(full_text))
            if not usage.total and full_text:
                # DeepSeek 实测会返回真实 usage，这里只是保险
                estimated = True
                usage.completion_tokens = metrics.estimate_tokens(full_text)
                usage.prompt_tokens = metrics.estimate_tokens_from_chars(prompt_char_size(context.logs))
            _record(
                "/diagnose/stream",
                started,
                usage=usage,
                error=error,
                tokens_estimated=estimated,
                llm_call=True,  # 走到这里就说明真的发起过流式调用了
            )
        yield SSE_DONE

    return StreamingResponse(gen(), media_type="text/event-stream")


# --------------------------------------------------------------------------- #
# 内部工具
# --------------------------------------------------------------------------- #
def _pick_target(logs: list[ParsedLog]) -> ParsedLog:
    """返回锚点日志（兼容旧调用）。

    新代码请用 ``app.context.build_context`` —— 它会以锚点为中心
    选出一组相关日志，而不是只取一条。
    """
    return logs[context_mod.select_anchor_index(logs)]


def _cache_key(text: str, context: LogContext | None = None) -> str:
    """生成缓存键。

    带上模型名、Prompt 版本、**上下文聚合参数**，以及**检索结果指纹**：
    换模型、改 Prompt、调聚合窗口、切换 RAG 模式或检索到不同案例，
    任何一项变化都会让旧缓存自动失效。

    检索结果指纹放的是「这次检索到了哪几条案例」而不是「配置是什么」——
    因为知识库变了之后，同样的配置也会检索出不同案例。
    """
    rag_signature = ""
    if context is not None:
        cases = rag.retrieve(format_logs(context.logs))
        rag_signature = rag.signature(retrieved_ids=[c.id for c in cases])

    return cache_mod.make_key(
        text,
        model=current_model(),
        prompt_version=PROMPT_VERSION,
        context_signature=context.cache_signature() if context is not None else "",
        rag_signature=rag_signature,
    )


def _canonical(text: str) -> str:
    """把流式拼出来的文本规范化后再入缓存。

    这样「同步接口写入的缓存」和「流式接口写入的缓存」格式一致，
    两个接口可以互相复用彼此的缓存条目。
    """
    try:
        return json.dumps(json.loads(text), ensure_ascii=False)
    except (json.JSONDecodeError, TypeError):
        return text


def _sse(payload: str) -> str:
    return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"


def _record(
    endpoint: str,
    started: float,
    usage: Usage | None = None,
    cache_hit: bool = False,
    error: str = "",
    tokens_estimated: bool = False,
    llm_call: bool = False,
) -> None:
    """写一条度量记录。

    ``llm_call`` 由调用方显式传入，**不要按 endpoint 前缀推断** ——
    缓存命中和"没解析到日志"走的都是 /diagnose，但它们一次模型都没调。
    之前用 ``endpoint.startswith("/diagnose")`` 判断，结果 llm_calls 实际是
    "diagnose 请求数"，指标名与含义不符。
    """
    usage = usage or Usage()
    metrics.store.record(
        metrics.CallRecord(
            endpoint=endpoint,
            duration_ms=(time.perf_counter() - started) * 1000,
            prompt_tokens=usage.prompt_tokens,
            cached_prompt_tokens=usage.cached_tokens,
            completion_tokens=usage.completion_tokens,
            cache_hit=cache_hit,
            llm_call=llm_call,
            tokens_estimated=tokens_estimated,
            error=error,
        )
    )


# 静态演示页（必须挂在最后，否则会拦截上面的 API 路由）
app.mount("/", StaticFiles(directory=str(BASE_DIR / "static"), html=True), name="static")
