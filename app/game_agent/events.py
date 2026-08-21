"""Agent 运行时事件；同一份事件可供 SSE、日志或测试消费。"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Awaitable, Callable, Literal
from uuid import uuid4

from pydantic import BaseModel, Field


AgentEventType = Literal[
    "turn.started", "turn.completed",
    "context.measured", "compaction.completed",
    "model.started", "model.token", "model.completed",
    "tool.started", "tool.completed",
]


class AgentEvent(BaseModel):
    """一轮执行中的有序事件。payload 只放该事件真正需要的数据。"""

    event_id: str = Field(default_factory=lambda: str(uuid4()))
    event_type: AgentEventType
    session_id: str
    turn_id: str
    sequence: int
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    payload: dict[str, Any] = Field(default_factory=dict)


EventSink = Callable[[AgentEvent], Awaitable[None]]


class EventEmitter:
    """为单个 Turn 补齐标识和递增序号，并把事件交给可选消费者。"""

    def __init__(self, session_id: str, sink: EventSink | None = None):
        self.session_id = session_id
        self.turn_id = str(uuid4())
        self.sink = sink
        self.sequence = 0

    async def emit(self, event_type: AgentEventType, **payload: Any) -> None:
        self.sequence += 1
        if self.sink is None:
            return
        await self.sink(AgentEvent(
            event_type=event_type,
            session_id=self.session_id,
            turn_id=self.turn_id,
            sequence=self.sequence,
            payload=payload,
        ))
