"""FastAPI 应用入口。

main.py 只负责三件事：
1. 读取环境变量；
2. 初始化共享运行时依赖；
3. 创建 FastAPI app 并挂载 API 路由。

HTTP 路由和 SSE 协议转换在 ``app/api/routes.py``，Agent 主循环在
``app/game_agent/agent.py``。
"""

from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager

from dotenv import load_dotenv
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.api.routes import router
from app.game_agent.agent import build_harness
from app.runtime.runmanager import RunManager
from app.storage.attachmentstore import MinioAttachmentStore
from app.storage.sessionstore import SessionStore


load_dotenv()

DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "postgresql://gamescope:gamescope@127.0.0.1:5433/gamescope?sslmode=disable",
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用启动/关闭生命周期：连接存储并构建可复用 Agent Runtime。"""
    attachment_store = MinioAttachmentStore.from_env()
    session_store = SessionStore(DATABASE_URL, attachment_store=attachment_store)

    await attachment_store.setup()
    await session_store.setup()

    app.state.session_store = session_store
    app.state.harness = build_harness(
        session_store,
        attachment_loader=session_store.get_attachment,
    )
    app.state.run_manager = RunManager(app.state.harness, session_store)
    logging.info("Game_Rover Agent Harness 已启动")

    try:
        yield
    finally:
        await app.state.run_manager.close()
        await session_store.close()


app = FastAPI(
    title="Game_Rover Agent Harness API",
    version="6.0.0",
    description="带持久会话、Tool Loop、上下文预算和滚动摘要的游戏资讯 Agent",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.exception_handler(Exception)
async def generic_exception_handler(_: Request, exc: Exception):
    """兜底异常处理；流式接口内部仍会优先返回 SSE error。"""
    logging.exception("未处理异常")
    return JSONResponse(status_code=500, content={"detail": str(exc)})

app.include_router(router)
