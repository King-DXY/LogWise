"""LogWise - LLM 智能日志诊断助手

阶段 3：API 接口 + SSE 流式输出 + 演示页。
"""
import json
import os

from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from app.diagnose import build_messages, get_async_client, diagnose as app_diagnose
from app.parser import parse_text

app = FastAPI(
    title="LogWise",
    description="LLM 智能日志诊断助手",
    version="0.3.0",
)


class ParseRequest(BaseModel):
    text: str


class DiagnoseRequest(BaseModel):
    text: str


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/parse")
def parse(req: ParseRequest):
    """解析日志，返回结构化字段列表。"""
    logs = parse_text(req.text)
    return {"logs": [log.__dict__ for log in logs]}


@app.post("/diagnose")
def diagnose(req: DiagnoseRequest):
    """解析 + 诊断，一次性返回结果。"""
    logs = parse_text(req.text)
    if not logs:
        return {"error": "未解析到任何日志"}
    target = _pick_target(logs)
    return {"log": target.__dict__, "diagnosis": app_diagnose(target)}


@app.get("/diagnose/stream")
async def diagnose_stream(text: str):
    """SSE 流式诊断：逐 token 返回诊断 JSON 文本。"""
    logs = parse_text(text)
    if not logs:
        return StreamingResponse(
            iter(['data: {"error": "未解析到任何日志"}\n\n']),
            media_type="text/event-stream",
        )
    target = _pick_target(logs)

    # 用异步客户端 + async for：同步客户端在 async 生成器里迭代会阻塞事件循环，
    # 导致 uvicorn 无法逐块发送（表现为"一次性输出"）。
    resp = await get_async_client().chat.completions.create(
        model=os.getenv("DEEPSEEK_MODEL", "deepseek-chat"),
        messages=build_messages(target),
        temperature=0.2,
        stream=True,
    )

    async def gen():
        async for chunk in resp:
            delta = chunk.choices[0].delta.content
            if delta:
                yield f"data: {json.dumps(delta, ensure_ascii=False)}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(gen(), media_type="text/event-stream")


def _pick_target(logs):
    """优先选 ERROR 级别日志诊断，没有则用第一条。"""
    return next((log for log in logs if log.level in ("ERROR", "CRITICAL", "FATAL")), logs[0])


# 静态演示页（挂载在最后，避免拦截上面的 API 路由）
app.mount("/", StaticFiles(directory="static", html=True), name="static")
