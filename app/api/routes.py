"""HTTP API 层：把外部请求转换为 Agent Harness 调用。

这个文件只处理 FastAPI 路由、SSE 编码、请求/响应适配；真正的 Agent 循环在
``game_agent/agent.py``，数据库和附件存储在 ``storage/``。
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from pathlib import Path
from uuid import uuid4

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import FileResponse, Response, StreamingResponse
from langchain_core.messages import HumanMessage

from app.game_agent.models import (
    AttachmentRef,
    ChatRequest,
    ChatResponse,
    SessionRenameRequest,
)
from app.game_agent.stream import (
    AgentStreamEvent,
    ModelCompleted,
    ModelStarted,
    TextDelta,
    ToolCompleted,
    ToolStarted,
    TurnCompleted,
    TurnError,
)
from app.runtime.runmanager import RunAlreadyActive, RunCancelledEvent, RunSnapshotEvent


router = APIRouter()
WEB_DIR = Path(__file__).parents[1] / "web"


def sse(event: str, data: dict) -> str:
    """把一个业务事件编码为 SSE 文本块。"""
    payload = json.dumps(data, ensure_ascii=False, default=str)
    return f"event: {event}\ndata: {payload}\n\n"


async def prepare_turn(store, req: ChatRequest) -> HumanMessage:
    """保存用户消息和附件，返回 Agent 本轮需要的 HumanMessage。

    持久消息只保存文本和图片引用；图片原文已在 record_user_message() 中写入 MinIO。
    模型调用前，Harness 会再把最新消息里的 type="image" 引用临时转成 image_url。
    """
    stored_attachments = await store.record_user_message(
        req.session_id,
        req.question,
        req.attachments,
    )
    refs = [AttachmentRef.model_validate(item) for item in stored_attachments]
    if not refs:
        return HumanMessage(content=req.question, id=str(uuid4()))

    content = []
    if req.question:
        content.append({"type": "text", "text": req.question})
    content.extend({"type": "image", "attachment": ref.model_dump()} for ref in refs)
    return HumanMessage(content=content, id=str(uuid4()))


def event_to_sse(
    event: AgentStreamEvent | RunSnapshotEvent | RunCancelledEvent,
) -> str | None:
    """把 Harness 内部事件映射成前端稳定消费的 SSE 事件。"""
    if isinstance(event, RunSnapshotEvent):
        return sse("run_snapshot", event.snapshot)
    if isinstance(event, RunCancelledEvent):
        return sse("cancelled", {"answer": event.answer})
    if isinstance(event, TextDelta):
        return sse("token", {"content": event.content})
    if isinstance(event, ModelStarted):
        return sse("model_started", {})
    if isinstance(event, ModelCompleted):
        return sse("model_completed", {"elapsed_ms": event.elapsed_ms})
    if isinstance(event, ToolStarted):
        return sse("tool_started", {"name": event.name})
    if isinstance(event, ToolCompleted):
        return sse("tool_completed", {"name": event.name})
    if isinstance(event, TurnCompleted):
        return sse("final", {"answer": event.answer})
    if isinstance(event, TurnError):
        return sse("error", {"detail": event.detail})
    return None


@router.get("/", include_in_schema=False)
async def chat_page():
    return FileResponse(WEB_DIR / "index.html")


@router.get("/favicon.svg", include_in_schema=False)
async def favicon_svg():
    return FileResponse(WEB_DIR / "favicon.svg", media_type="image/svg+xml")


@router.get("/favicon.ico", include_in_schema=False)
async def favicon_ico():
    return FileResponse(WEB_DIR / "favicon.svg", media_type="image/svg+xml")


@router.post("/ai/chat", response_model=ChatResponse)
async def chat(request: Request, req: ChatRequest) -> ChatResponse:
    """非流式接口：启动同一种后台 Run，等待其终态后返回。"""
    store = request.app.state.session_store
    try:
        run = await request.app.state.run_manager.start(
            req.session_id,
            lambda: prepare_turn(store, req),
        )
    except RunAlreadyActive as exc:
        raise HTTPException(
            status_code=409,
            detail={"message": str(exc), "active_run_id": exc.run_id},
        ) from exc
    result = await request.app.state.run_manager.wait(run["run_id"])
    if not result or result["status"] != "completed":
        raise HTTPException(status_code=500, detail=(result or {}).get("error", "任务执行失败"))
    return ChatResponse(answer=result["final_answer"] or "")


@router.post("/ai/chat/stream")
async def stream_chat(request: Request, req: ChatRequest):
    """启动后台 Run；当前 SSE 仅订阅事件，断开不会取消 Run。"""

    store = request.app.state.session_store
    manager = request.app.state.run_manager
    try:
        run = await manager.start(
            req.session_id,
            lambda: prepare_turn(store, req),
        )
    except RunAlreadyActive as exc:
        raise HTTPException(
            status_code=409,
            detail={"message": str(exc), "active_run_id": exc.run_id},
        ) from exc

    async def events() -> AsyncIterator[str]:
        async for event in manager.subscribe(run["run_id"]):
            chunk = event_to_sse(event)
            if chunk:
                yield chunk

    return StreamingResponse(
        events(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@router.get("/ai/sessions")
async def list_sessions(request: Request):
    return {"sessions": await request.app.state.session_store.list_sessions()}


@router.get("/ai/runs/{run_id}")
async def inspect_run(request: Request, run_id: str):
    run = await request.app.state.run_manager.get(run_id)
    if not run:
        raise HTTPException(status_code=404, detail="run not found")
    return run


@router.post("/ai/runs/{run_id}/cancel")
async def cancel_run(request: Request, run_id: str):
    """停止一个后台 Run；重复停止终态 Run 时直接返回其当前状态。"""
    run = await request.app.state.run_manager.cancel(run_id)
    if not run:
        raise HTTPException(status_code=404, detail="Run 不存在")
    return run


@router.get("/ai/sessions/{session_id}/active-run")
async def active_session_run(request: Request, session_id: str):
    """返回最近一次 Run；终态短暂保留，用于处理刷新与完成的竞态。"""
    return {"run": await request.app.state.run_manager.get_latest(session_id)}


@router.get("/ai/sessions/{session_id}/messages")
async def session_messages(request: Request, session_id: str):
    store = request.app.state.session_store
    messages = await store.get_messages(session_id)
    if messages:
        return {"session_id": session_id, "messages": messages}

    state = await store.load_state(session_id)
    if not state.get("active_messages"):
        raise HTTPException(status_code=404, detail="session not found")

    fallback = []
    for message in state["active_messages"]:
        if message.type == "human":
            fallback.append({"role": "user", "content": message.content})
        elif message.type == "ai" and message.content:
            fallback.append({"role": "assistant", "content": message.content})
    return {"session_id": session_id, "messages": fallback}


@router.get("/ai/sessions/{session_id}/attachments/{attachment_id}")
async def attachment_content(request: Request, session_id: str, attachment_id: str):
    attachment = await request.app.state.session_store.get_attachment(attachment_id, session_id)
    if not attachment:
        raise HTTPException(status_code=404, detail="attachment not found")
    return Response(
        content=bytes(attachment["content"]),
        media_type=attachment["mime_type"],
        headers={"Cache-Control": "private, max-age=31536000, immutable"},
    )


@router.patch("/ai/sessions/{session_id}")
async def rename_session(request: Request, session_id: str, req: SessionRenameRequest):
    title = " ".join(req.title.split())
    if not title:
        raise HTTPException(status_code=422, detail="title cannot be blank")
    if not await request.app.state.session_store.rename_session(session_id, title):
        raise HTTPException(status_code=404, detail="session not found")
    return {"session_id": session_id, "title": title}


@router.delete("/ai/sessions/{session_id}", status_code=204)
async def delete_session(request: Request, session_id: str):
    run = await request.app.state.run_manager.get_latest(session_id)
    if run and run["status"] not in {"completed", "failed", "cancelled"}:
        raise HTTPException(status_code=409, detail="当前会话仍有任务正在执行")
    deleted = await request.app.state.session_store.delete_session(session_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="session not found")
    await request.app.state.run_manager.forget_session(session_id)


@router.get("/ai/sessions/{session_id}/state")
async def inspect_session(request: Request, session_id: str):
    state = await request.app.state.session_store.load_state(session_id)
    if not state.get("active_messages"):
        raise HTTPException(status_code=404, detail="session not found")
    state = dict(state)
    state["active_messages"] = [
        {
            "id": message.id,
            "type": message.type,
            "name": getattr(message, "name", None),
            "content": message.content,
            "tool_calls": getattr(message, "tool_calls", []),
        }
        for message in state.get("active_messages", [])
    ]
    return state


@router.get("/ai/health")
async def health(request: Request):
    return {
        "status": "ok",
        "architecture": "single-agent-budgeted-harness",
        "persistence": "postgresql",
        "tools": ["read_skill", "web_search"],
        "skills": [item.name for item in request.app.state.harness.skill_registry.catalog()],
    }
