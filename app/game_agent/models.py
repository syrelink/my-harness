"""GameRover 的共享数据协议。

这里不执行业务逻辑，只定义接口请求响应、工具轨迹和结构化模型。
已移除 LangGraph 的 State 定义：会话状态现在是普通 dict，由 SessionStore 持久化。
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, model_validator


class ContextSummary(BaseModel):
    """较早历史的通用任务存档"""

    primary_request_and_intent: list[str] = Field(default_factory=list)
    key_concepts: list[str] = Field(default_factory=list)
    completed_work: list[str] = Field(default_factory=list)
    errors_and_recoveries: list[str] = Field(default_factory=list)
    pending_tasks: list[str] = Field(default_factory=list)
    current_work: list[str] = Field(default_factory=list)
    next_step: str = ""
    critical_context: list[str] = Field(default_factory=list)
    important_tool_results: list[str] = Field(default_factory=list)
    referenced_artifacts: list[str] = Field(default_factory=list)


class ContextMetrics(BaseModel):
    """一次上下文预算检查的可观测指标，供 Harness 面板展示。"""

    context_window_tokens: int = 0
    trigger_ratio: float = 0.8
    trigger_tokens: int = 0
    retain_ratio: float = 0.16
    recent_budget_tokens: int = 0
    summary_budget_tokens: int = 0
    summary_tokens: int = 0
    model_input_tokens: int = 0
    model_input_source: Literal["estimated", "api_usage"] = "estimated"
    messages_before: int = 0
    messages_after: int = 0
    compacted_messages: int = 0
    tokens_before_compaction: int = 0
    tokens_after_compaction: int = 0
    reduced_tokens: int = 0
    converged: bool = False
    summary_version: int = 0


class TurnTokenUsage(BaseModel):
    """当前一轮内全部主模型调用的 Token 汇总。"""

    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    model_calls: int = 0
    estimated_calls: int = 0


class ToolTrace(BaseModel):
    """一次 Tool Call 的审计记录。"""

    name: str
    arguments: dict = Field(default_factory=dict)
    status: Literal["success", "error"]
    preview: str
    latency_ms: int
    error_type: str | None = None
    truncated: bool = False


class AttachmentRef(BaseModel):
    """State 中保存的轻量图片引用，不包含原始二进制或 Base64。"""

    attachment_id: str = Field(min_length=1)
    name: str = Field(min_length=1, max_length=255)
    mime_type: str = Field(pattern=r"^image/")
    size: int = Field(ge=0, le=10 * 1024 * 1024)


class AttachmentInput(BaseModel):
    """前端上传到聊天接口的单个附件。"""

    name: str = Field(min_length=1, max_length=255)
    mime_type: str = Field(min_length=1, max_length=100)
    size: int = Field(ge=0, le=10 * 1024 * 1024)
    data_url: str = Field(min_length=1)


class ChatRequest(BaseModel):
    """开始一轮 Agent 执行所需的输入。"""

    question: str = ""
    session_id: str = "default"
    attachments: list[AttachmentInput] = Field(default_factory=list, max_length=5)
    force_compaction: bool = False

    @model_validator(mode="after")
    def require_content(self):
        if not self.question.strip() and not self.attachments:
            raise ValueError("question or attachment is required")
        if any(not item.mime_type.startswith("image/") for item in self.attachments):
            raise ValueError("当前阶段只支持图片附件")
        if sum(item.size for item in self.attachments) > 20 * 1024 * 1024:
            raise ValueError("total attachment size cannot exceed 20MB")
        return self


class ChatResponse(BaseModel):
    """一轮执行完成后返回给前端的答案和 Harness 状态摘要。"""

    answer: str
    tool_trace: list[ToolTrace]
    context_metrics: ContextMetrics
    token_usage: TurnTokenUsage
    context_summary: ContextSummary
    compacted: bool


class SessionRenameRequest(BaseModel):
    """历史会话重命名接口的输入。"""

    title: str = Field(min_length=1, max_length=100)
