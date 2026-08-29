"""Agent 对调用方公开的最小流式协议。

这些事件描述的是产品输出，不承担日志和观测职责；Langfuse 仍单独记录内部调用链。
"""

from dataclasses import dataclass, field
from typing import Literal, TypeAlias


@dataclass(frozen=True, slots=True)
class TextDelta:
    """模型刚生成的一段可展示文字。"""

    content: str
    type: Literal["text_delta"] = field(default="text_delta", init=False)


@dataclass(frozen=True, slots=True)
class ToolStarted:
    """模型请求调用某个工具；只用于展示状态，不暴露 tool_call 参数。"""

    name: str
    type: Literal["tool_started"] = field(default="tool_started", init=False)


@dataclass(frozen=True, slots=True)
class ToolCompleted:
    """某个工具执行完成；工具结果会写回模型上下文，不直接推给前端。"""

    name: str
    elapsed_ms: int | None = None
    type: Literal["tool_completed"] = field(default="tool_completed", init=False)


@dataclass(frozen=True, slots=True)
class ModelStarted:
    """开始一次 LLM 调用；用于前端展示“正在思考/正在整理回答”。"""

    type: Literal["model_started"] = field(default="model_started", init=False)


@dataclass(frozen=True, slots=True)
class ModelCompleted:
    """一次 LLM 调用结束；用于展示本次模型调用耗时。"""

    elapsed_ms: int
    type: Literal["model_completed"] = field(default="model_completed", init=False)


@dataclass(frozen=True, slots=True)
class TurnCompleted:
    """整个工具循环结束后的最终回答。"""

    answer: str
    type: Literal["turn_completed"] = field(default="turn_completed", init=False)


@dataclass(frozen=True, slots=True)
class TurnError:
    """整轮 Agent 执行失败。"""

    detail: str
    type: Literal["turn_error"] = field(default="turn_error", init=False)


AgentStreamEvent: TypeAlias = (
    TextDelta
    | ToolStarted
    | ToolCompleted
    | ModelStarted
    | ModelCompleted
    | TurnCompleted
    | TurnError
)
