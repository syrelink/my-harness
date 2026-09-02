"""单进程 Agent Run 生命周期管理。

RunManager 持有后台任务，因此 HTTP 或 SSE 连接断开不会取消 Agent 本轮执行。
当前实现只解决页面刷新恢复，不负责进程重启恢复或分布式执行；最终回答仍通过
SessionStore 持久化。

AgentRun：记录一次回答现在执行到哪里。

asyncio.Task：真正负责在后台运行Agent。

Queue：把Agent事件临时传给SSE连接。

SessionStore：永久保存最终聊天消息。
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import AsyncIterator, Awaitable, Callable
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Literal, TypeAlias
from uuid import uuid4

from app.game_agent.stream import (
    AgentStreamEvent,
    ModelStarted,
    TextDelta,
    ToolCompleted,
    ToolStarted,
    TurnCompleted,
    TurnError,
)


RunStatus: TypeAlias = Literal["queued", "running", "completed", "failed", "cancelled"]
TERMINAL_STATUSES = frozenset({"completed", "failed", "cancelled"})


class RunAlreadyActive(RuntimeError):
    """同一会话已存在未结束 Run 时抛出。"""

    def __init__(self, run_id: str):
        self.run_id = run_id
        super().__init__(f"当前会话已有任务正在执行：{run_id}")


class RunSnapshotEvent:
    """订阅建立后的首个快照，避免客户端不知道 Run 当前执行到了哪里。"""

    def __init__(self, snapshot: dict[str, Any]) -> None:
        self.snapshot = snapshot


class RunCancelledEvent:
    """Run 被用户停止；answer 保存停止前已经生成的文字。"""

    def __init__(self, answer: str) -> None:
        self.answer = answer


RunEvent: TypeAlias = AgentStreamEvent | RunSnapshotEvent | RunCancelledEvent
MessageFactory: TypeAlias = Callable[[], Awaitable[Any]]


@dataclass(slots=True)
class AgentRun:
    run_id: str
    session_id: str
    status: RunStatus = "queued"
    phase: str = "queued"
    partial_answer: str = ""
    final_answer: str | None = None
    current_tool: str | None = None
    error: str | None = None
    created_at: datetime = field(default_factory=lambda: datetime.now().astimezone())
    finished_at: datetime | None = None
    subscribers: set[asyncio.Queue[RunEvent]] = field(default_factory=set, repr=False)
    lock: asyncio.Lock = field(default_factory=asyncio.Lock, repr=False)

    def snapshot(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "session_id": self.session_id,
            "status": self.status,
            "phase": self.phase,
            "partial_answer": self.partial_answer,
            "final_answer": self.final_answer,
            "current_tool": self.current_tool,
            "error": self.error,
            "created_at": self.created_at.isoformat(),
            "finished_at": self.finished_at.isoformat() if self.finished_at else None,
        }


class RunManager:
    """持有后台 Agent Turn，并提供状态查询和事件订阅。

    每个会话同一时间只允许一个未结束 Run。最近一次终态 Run 会暂存在内存中，
    这样页面刷新恰好撞上任务完成时，仍能查询终态并重新加载持久化聊天记录。
    """

    def __init__(self, harness, session_store):
        self.harness = harness
        self.session_store = session_store
        self._runs: dict[str, AgentRun] = {}
        self._tasks: dict[str, asyncio.Task[None]] = {}
        self._latest_by_session: dict[str, str] = {}
        self._lock = asyncio.Lock()

    async def start(self, session_id: str, message_factory: MessageFactory) -> dict[str, Any]:
        """占用会话并启动一个不依赖 HTTP 连接的后台 Agent Turn。"""
        async with self._lock:
            previous_id = self._latest_by_session.get(session_id)
            previous = self._runs.get(previous_id) if previous_id else None
            if previous and previous.status not in TERMINAL_STATUSES:
                raise RunAlreadyActive(previous.run_id)
            if previous:
                # 页面刷新只需恢复最近一次 Run，不长期保留旧的内存状态。
                self._runs.pop(previous.run_id, None)

            run = AgentRun(run_id=f"run-{uuid4()}", session_id=session_id)
            self._runs[run.run_id] = run
            self._latest_by_session[session_id] = run.run_id
            task = asyncio.create_task(
                self._execute(run, message_factory),
                name=f"agent-run:{run.run_id}",
            )
            self._tasks[run.run_id] = task
            task.add_done_callback(lambda _: self._tasks.pop(run.run_id, None))
            return run.snapshot()

    async def get(self, run_id: str) -> dict[str, Any] | None:
        run = self._runs.get(run_id)
        if not run:
            return None
        async with run.lock:
            return run.snapshot()

    async def get_latest(self, session_id: str) -> dict[str, Any] | None:
        run_id = self._latest_by_session.get(session_id)
        return await self.get(run_id) if run_id else None

    async def forget_session(self, session_id: str) -> None:
        """持久化会话删除后，清理对应的终态 Run。"""
        async with self._lock:
            run_id = self._latest_by_session.get(session_id)
            run = self._runs.get(run_id) if run_id else None
            if run and run.status not in TERMINAL_STATUSES:
                raise RunAlreadyActive(run.run_id)
            self._latest_by_session.pop(session_id, None)
            if run_id:
                self._runs.pop(run_id, None)

    async def subscribe(self, run_id: str) -> AsyncIterator[RunEvent]:
        """先返回权威快照，再持续返回实时事件，直到 Run 进入终态。"""
        run = self._runs.get(run_id)
        if not run:
            raise KeyError(run_id)
        queue: asyncio.Queue[RunEvent] = asyncio.Queue()
        async with run.lock:
            run.subscribers.add(queue)
            snapshot = run.snapshot()

        yield RunSnapshotEvent(snapshot)
        if snapshot["status"] in TERMINAL_STATUSES:
            async with run.lock:
                run.subscribers.discard(queue)
            return

        try:
            while True:
                event = await queue.get()
                yield event
                if isinstance(event, (TurnCompleted, TurnError, RunCancelledEvent)):
                    return
        finally:
            async with run.lock:
                run.subscribers.discard(queue)

    async def wait(self, run_id: str) -> dict[str, Any] | None:
        task = self._tasks.get(run_id)
        if task:
            await task
        return await self.get(run_id)

    async def cancel(self, run_id: str) -> dict[str, Any] | None:
        """取消后台任务并等待状态进入终态；Run 不存在时返回 None。"""
        run = self._runs.get(run_id)
        if not run:
            return None
        task = self._tasks.get(run_id)
        if task and not task.done():
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
        return await self.get(run_id)

    async def close(self) -> None:
        """应用关闭时取消当前进程内尚未完成的 Run。"""
        tasks = [task for task in self._tasks.values() if not task.done()]
        for task in tasks:
            task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)

    async def _execute(self, run: AgentRun, message_factory: MessageFactory) -> None:
        completed = False
        try:
            await self._update(run, status="running", phase="preparing")
            user_message = await message_factory()
            async for event in self.harness.stream_turn(run.session_id, user_message):
                if isinstance(event, TextDelta):
                    await self._append_text(run, event.content)
                elif isinstance(event, ModelStarted):
                    await self._update(run, phase="model_call", current_tool=None)
                elif isinstance(event, ToolStarted):
                    await self._update(run, phase="tool_call", current_tool=event.name)
                elif isinstance(event, ToolCompleted):
                    await self._update(run, phase="model_call", current_tool=None)
                elif isinstance(event, TurnError):
                    await self._fail(run, event.detail)
                    await self._publish(run, event)
                    return
                elif isinstance(event, TurnCompleted):
                    # 先持久化再标记完成，确保前端看到 completed 后一定能读到最终回答。
                    await self.session_store.record_assistant_message(run.session_id, event.answer)
                    await self._update(
                        run,
                        status="completed",
                        phase="completed",
                        partial_answer=event.answer,
                        final_answer=event.answer,
                        current_tool=None,
                        finished_at=datetime.now().astimezone(),
                    )
                    completed = True
                await self._publish(run, event)

            if not completed:
                raise RuntimeError("Agent事件流结束，但没有返回最终回答")
        except asyncio.CancelledError:
            partial_answer = run.partial_answer.strip()
            if partial_answer:
                try:
                    await self.session_store.record_assistant_message(
                        run.session_id,
                        partial_answer,
                    )
                except Exception:
                    # 停止操作必须优先完成；部分回答持久化失败只记录日志。
                    logging.exception("停止 Run 时保存部分回答失败：%s", run.run_id)
            await self._update(
                run,
                status="cancelled",
                phase="cancelled",
                error=None,
                finished_at=datetime.now().astimezone(),
            )
            await self._publish(run, RunCancelledEvent(partial_answer))
            raise
        except Exception as exc:
            logging.exception("Agent Run执行失败：%s", run.run_id)
            await self._fail(run, str(exc))
            await self._publish(run, TurnError(str(exc)))

    async def _append_text(self, run: AgentRun, content: str) -> None:
        async with run.lock:
            run.partial_answer += content

    async def _fail(self, run: AgentRun, detail: str) -> None:
        await self._update(
            run,
            status="failed",
            phase="failed",
            error=detail,
            current_tool=None,
            finished_at=datetime.now().astimezone(),
        )

    async def _update(self, run: AgentRun, **changes: Any) -> None:
        async with run.lock:
            for name, value in changes.items():
                setattr(run, name, value)

    async def _publish(self, run: AgentRun, event: RunEvent) -> None:
        async with run.lock:
            subscribers = tuple(run.subscribers)
        for queue in subscribers:
            queue.put_nowait(event)
