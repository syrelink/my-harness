"""Agent、API 和存储层共享的数据格式。"""

from __future__ import annotations

from pydantic import BaseModel, Field, model_validator


class ContextSummary(BaseModel):
    """摘要模型生成的结构化历史记忆。"""

    # 用户当前想实现什么。
    goals: list[str] = Field(default_factory=list)
    # 后续继续任务必须知道的事实。
    key_facts: list[str] = Field(default_factory=list)
    # 用户约束、技术边界和已经确认的方案。
    constraints_and_decisions: list[str] = Field(default_factory=list)
    # 已完成、正在处理和仍待处理的工作。
    completed_work: list[str] = Field(default_factory=list)
    current_work: list[str] = Field(default_factory=list)
    pending_tasks: list[str] = Field(default_factory=list)
    # 后续仍会用到的工具结论、错误恢复方式或附件引用。
    important_results: list[str] = Field(default_factory=list)


class AttachmentRef(BaseModel):
    """持久消息中的图片引用；原图保存在 MinIO。"""

    attachment_id: str = Field(min_length=1)
    name: str = Field(min_length=1, max_length=255)
    mime_type: str = Field(pattern=r"^image/")
    size: int = Field(ge=0, le=10 * 1024 * 1024)


class AttachmentInput(BaseModel):
    """聊天接口接收的一张图片；保存到 MinIO 后转换为 AttachmentRef。"""

    name: str = Field(min_length=1, max_length=255)
    mime_type: str = Field(pattern=r"^image/", max_length=100)
    size: int = Field(ge=0, le=10 * 1024 * 1024)
    data_url: str = Field(min_length=1)


class ChatRequest(BaseModel):
    """一次聊天请求。"""

    question: str = ""
    session_id: str = Field(min_length=1, max_length=200)
    attachments: list[AttachmentInput] = Field(default_factory=list, max_length=5)

    @model_validator(mode="after")
    def validate_request(self):
        """校验无法由单个 Field 表达的请求级规则。"""
        if not self.question.strip() and not self.attachments:
            raise ValueError("文字和附件不能同时为空")
        if sum(item.size for item in self.attachments) > 20 * 1024 * 1024:
            raise ValueError("附件总大小不能超过 20MB")
        return self


class ChatResponse(BaseModel):
    """一次聊天请求的最终结果。"""

    answer: str


class SessionRenameRequest(BaseModel):
    """历史会话重命名接口的输入。"""

    title: str = Field(min_length=1, max_length=100)
