"""GameRover 的最小滚动上下文压缩。

规则只有三条：模型调用前估算输入；达到 80% 后保留最近 16% 的完整 Turn；
新摘要由“旧摘要 + 本次过期消息”生成并替换旧摘要。
"""

from __future__ import annotations

import asyncio
import json
import os
from dataclasses import dataclass

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AnyMessage, HumanMessage, SystemMessage
from langchain_core.messages.utils import count_tokens_approximately

from app.game_agent.models import ContextMetrics, ContextSummary
from app.game_agent.prompts import COMPACTION_PROMPT


@dataclass(frozen=True)
class ContextBudget:
    """上下文窗口及压缩、保留比例。"""

    context_window_tokens: int
    trigger_ratio: float = 0.8
    retain_ratio: float = 0.16
    summary_tokens: int = 8192
    protocol_overhead_tokens: int = 2500

    def __post_init__(self) -> None:
        if self.context_window_tokens <= 0:
            raise ValueError("context_window_tokens 必须大于 0")
        if not 0 < self.retain_ratio < self.trigger_ratio < 1:
            raise ValueError("必须满足 0 < retain_ratio < trigger_ratio < 1")

    @property
    def trigger_tokens(self) -> int:
        return int(self.context_window_tokens * self.trigger_ratio)

    @property
    def recent_tokens(self) -> int:
        return int(self.context_window_tokens * self.retain_ratio)

    @classmethod
    def from_env(cls) -> "ContextBudget":
        return cls(
            context_window_tokens=int(os.getenv("GAME_EFFECTIVE_CONTEXT_TOKENS", "65536")),
            trigger_ratio=float(os.getenv("GAME_CONTEXT_TRIGGER_RATIO", "0.8")),
            retain_ratio=float(os.getenv("GAME_CONTEXT_RETAIN_RATIO", "0.16")),
            summary_tokens=int(os.getenv("GAME_SUMMARY_BUDGET_TOKENS", "8192")),
            protocol_overhead_tokens=int(os.getenv("GAME_PROMPT_OVERHEAD_TOKENS", "2500")),
        )


def estimate_tokens(messages: list[AnyMessage]) -> int:
    """估算实际消息列表的文本 Token；图片真实用量以供应商 Usage 为准。"""
    normalized = []
    for message in messages:
        if not isinstance(message.content, list):
            normalized.append(message)
            continue
        text = "\n".join(
            str(block.get("text", ""))
            for block in message.content
            if isinstance(block, dict) and block.get("type") in {"text", "input_text"}
        )
        normalized.append(message.model_copy(update={"content": text}))
    return count_tokens_approximately(normalized) if normalized else 0


def split_into_complete_turns(messages: list[AnyMessage]) -> list[list[AnyMessage]]:
    """按 HumanMessage 开始位置切成完整 Turn。"""
    turns: list[list[AnyMessage]] = []
    for message in messages:
        if message.type == "human" or not turns:
            turns.append([])
        turns[-1].append(message)
    return turns


def split_by_recent_budget(
    messages: list[AnyMessage],
    recent_budget: int,
) -> tuple[list[AnyMessage], list[AnyMessage]]:
    """从后向前保留完整 Turn，近期原文总量严格不超过预算。"""
    turns = split_into_complete_turns(messages)
    if len(turns) <= 1:
        return [], messages

    keep_count = 0
    used_tokens = 0
    for turn in reversed(turns):
        turn_tokens = estimate_tokens(turn)
        if used_tokens + turn_tokens > recent_budget:
            if keep_count == 0:
                raise RuntimeError("最新完整 Turn 已超过 16% 近期原文预算，无法安全压缩")
            break
        keep_count += 1
        used_tokens += turn_tokens

    expired_turns = turns[:-keep_count]
    recent_turns = turns[-keep_count:]
    return (
        [message for turn in expired_turns for message in turn],
        [message for turn in recent_turns for message in turn],
    )


def context_summary_from_state(state: dict) -> ContextSummary:
    return ContextSummary.model_validate(state.get("context_summary") or {})


class ContextManager:
    """压缩历史并构造每次真正发送给模型的消息。"""

    def __init__(self, summary_model: BaseChatModel, budget: ContextBudget | None = None):
        self.summary_model = summary_model
        self.budget = budget or ContextBudget.from_env()
        self.summary_timeout_seconds = float(os.getenv("GAME_SUMMARY_TIMEOUT_SECONDS", "12"))

    def build_model_context(self, state: dict, system_prompt: str) -> list[AnyMessage]:
        """构造临时输入：System Prompt + ContextSummary + 近期原文。"""
        context: list[AnyMessage] = [SystemMessage(content=system_prompt)]
        summary = context_summary_from_state(state)
        if any(summary.model_dump().values()):
            context.append(SystemMessage(
                content="【ContextSummary：较早历史的结构化任务摘要】\n"
                + json.dumps(summary.model_dump(), ensure_ascii=False)
            ))
        context.extend(state.get("active_messages", []))
        return context

    def estimate_model_context(self, state: dict, system_prompt: str) -> int:
        """估算当前真实消息输入，并为 Tool Schema 等协议结构保留固定预算。"""
        return (
            estimate_tokens(self.build_model_context(state, system_prompt))
            + self.budget.protocol_overhead_tokens
        )

    async def compact(
        self,
        state: dict,
        system_prompt: str,
        force: bool = False,
    ) -> dict:
        """达到阈值后滚动替换摘要；摘要失败直接向上抛错。"""
        messages = state.get("active_messages", [])
        tokens_before = self.estimate_model_context(state, system_prompt)
        base_metrics = ContextMetrics(
            context_window_tokens=self.budget.context_window_tokens,
            trigger_ratio=self.budget.trigger_ratio,
            trigger_tokens=self.budget.trigger_tokens,
            retain_ratio=self.budget.retain_ratio,
            recent_budget_tokens=self.budget.recent_tokens,
            summary_budget_tokens=self.budget.summary_tokens,
            model_input_tokens=tokens_before,
            messages_before=len(messages),
            messages_after=len(messages),
            summary_version=int(state.get("summary_version", 0)),
        )
        if not force and tokens_before < self.budget.trigger_tokens:
            return {
                "context_metrics": base_metrics.model_dump(),
                "compacted": False,
                "compaction_attempted": False,
            }

        expired, recent = split_by_recent_budget(messages, self.budget.recent_tokens)
        if not expired:
            raise RuntimeError("上下文已超过压缩阈值，但没有可安全压缩的旧完整 Turn")

        summary = await self._summarize(
            context_summary_from_state(state),
            expired,
        )
        next_version = int(state.get("summary_version", 0)) + 1
        candidate = {
            **state,
            "active_messages": recent,
            "context_summary": summary.model_dump(),
            "summary_version": next_version,
        }
        tokens_after = self.estimate_model_context(candidate, system_prompt)
        metrics = base_metrics.model_copy(update={
            "summary_tokens": estimate_tokens([
                SystemMessage(content=json.dumps(summary.model_dump(), ensure_ascii=False))
            ]),
            "model_input_tokens": tokens_after,
            "messages_after": len(recent),
            "compacted_messages": len(expired),
            "tokens_before_compaction": tokens_before,
            "tokens_after_compaction": tokens_after,
            "reduced_tokens": max(0, tokens_before - tokens_after),
            "converged": tokens_after < tokens_before,
            "summary_version": next_version,
        })
        return {
            "active_messages": recent,
            "context_summary": summary.model_dump(),
            "context_metrics": metrics.model_dump(),
            "compacted": True,
            "compaction_attempted": True,
            "compaction_count": int(state.get("compaction_count", 0)) + 1,
            "summary_version": next_version,
        }

    async def _summarize(
        self,
        old_summary: ContextSummary,
        expired_messages: list[AnyMessage],
    ) -> ContextSummary:
        """滚动摘要：NewSummary = Compress(OldSummary + ExpiredMessages)。"""
        instruction = HumanMessage(content=(
            f"{COMPACTION_PROMPT}\n\n"
            f"Existing ContextSummary："
            f"{json.dumps(old_summary.model_dump(), ensure_ascii=False)}\n"
            f"Summary Token Budget：{self.budget.summary_tokens}"
        ))
        # LangChain 负责提交 Pydantic Schema 并校验返回值；失败会直接抛错。
        structured_model = self.summary_model.bind(
            max_tokens=self.budget.summary_tokens
        ).with_structured_output(ContextSummary)
        return await asyncio.wait_for(
            structured_model.ainvoke([*expired_messages, instruction]),
            timeout=self.summary_timeout_seconds,
        )
