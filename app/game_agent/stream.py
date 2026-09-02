"""Agent 对调用方公开的最小流式协议。

这些事件描述的是产品输出，不承担日志和观测职责；Langfuse 仍单独记录内部调用链。
"""

from typing import Literal, TypeAlias


class TextDelta:
    """模型刚生成的一段可展示文字。"""

    type: Literal["text_delta"] = "text_delta"

    def __init__(self, content: str) -> None:
        self.content = content


class ToolStarted:
    """模型请求调用某个工具；只用于展示状态，不暴露 tool_call 参数。"""

    type: Literal["tool_started"] = "tool_started"

    def __init__(self, name: str) -> None:
        self.name = name


class ToolCompleted:
    """某个工具执行完成；工具结果会写回模型上下文，不直接推给前端。"""

    type: Literal["tool_completed"] = "tool_completed"

    def __init__(self, name: str) -> None:
        self.name = name


class ModelStarted:
    """开始一次 LLM 调用；用于前端展示“正在思考/正在整理回答”。"""

    type: Literal["model_started"] = "model_started"


class ModelCompleted:
    """一次 LLM 调用结束；用于展示本次模型调用耗时。"""

    type: Literal["model_completed"] = "model_completed"

    def __init__(self, elapsed_ms: int) -> None:
        self.elapsed_ms = elapsed_ms


class TurnCompleted:
    """整个工具循环结束后的最终回答。"""

    type: Literal["turn_completed"] = "turn_completed"

    def __init__(self, answer: str) -> None:
        self.answer = answer


class TurnError:
    """整轮 Agent 执行失败。"""

    type: Literal["turn_error"] = "turn_error"

    def __init__(self, detail: str) -> None:
        self.detail = detail


AgentStreamEvent: TypeAlias = (
    TextDelta
    | ToolStarted
    | ToolCompleted
    | ModelStarted
    | ModelCompleted
    | TurnCompleted
    | TurnError
)
