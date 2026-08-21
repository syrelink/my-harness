"""最小冒烟测试：覆盖无 LangGraph 版本中不依赖模型与数据库的纯逻辑。"""

from __future__ import annotations

from types import MethodType

import pytest
from langchain_core.messages import AIMessage, AIMessageChunk, HumanMessage

from app.attachment_store import attachment_object_key, attachment_prefix
from app.game_agent.agent import GameAgent
from app.game_agent.events import EventEmitter
from app.game_agent.memory import (
    ContextBudget,
    ContextManager,
    context_summary_from_state,
    split_by_recent_budget,
    split_into_complete_turns,
)
from app.game_agent.models import ContextSummary
from app.game_agent.search import plain_text
from app.game_agent.skills import SkillRegistry
from app.game_agent.tools import read_skill_file


def test_skill_registry_catalog_and_load():
    registry = SkillRegistry.default()
    catalog = registry.catalog()
    names = {item.name for item in catalog}
    assert {"game-news", "gameplay-guide", "game-build-advisor"} <= names
    for item in catalog:
        assert item.description  # 目录层必须有一句话描述
    doc = registry.load("game-news")
    assert doc.name == "game-news"
    assert "工作流" in doc.content


def test_skill_tool_returns_content():
    result = read_skill_file.invoke({"name": "game-news", "path": "SKILL.md"})
    assert '"status": "loaded"' in result
    assert "game-news" in result


def test_read_skill_file_rejects_unknown_reference():
    result = read_skill_file.invoke({"name": "game-news", "path": "references/nope.md"})
    assert '"error"' in result


def test_plain_text_strips_html():
    assert plain_text("<b>hello</b>   world") == "hello world"


def test_attachment_object_key_is_deterministic_and_session_scoped():
    attachment_id = "123e4567-e89b-12d3-a456-426614174000"
    key = attachment_object_key("session-a", attachment_id)
    assert key == attachment_object_key("session-a", attachment_id)
    assert key.startswith(attachment_prefix("session-a"))
    assert key != attachment_object_key("session-b", attachment_id)


def test_attachment_object_key_rejects_non_uuid():
    with pytest.raises(ValueError):
        attachment_object_key("session-a", "../../another-object")


def test_split_into_complete_turns():
    messages = [
        HumanMessage(content="q1"),
        AIMessage(content="a1"),
        HumanMessage(content="q2"),
        AIMessage(content="a2"),
    ]
    turns = split_into_complete_turns(messages)
    assert len(turns) == 2
    assert len(turns[0]) == 2


def test_split_by_recent_budget_keeps_recent():
    messages = []
    for i in range(10):
        messages.append(HumanMessage(content=f"q{i}"))
        messages.append(AIMessage(content=f"a{i}"))
    expired, recent = split_by_recent_budget(messages, recent_budget=50)
    assert len(recent) >= 2
    assert len(expired) + len(recent) == len(messages)


def test_context_summary_from_state_empty():
    assert context_summary_from_state({}) is not None


def test_model_context_estimate_includes_system_summary_and_messages():
    manager = ContextManager(None, ContextBudget(
        context_window_tokens=1_000,
        trigger_ratio=0.8,
        summary_tokens=20,
        retain_ratio=0.2,
        protocol_overhead_tokens=10,
    ))
    state = {
        "context_summary": {"critical_context": ["用户使用 Python"]},
        "active_messages": [HumanMessage(content="继续实现")],
    }
    messages_only = manager.build_model_context(state, "system")
    assert manager.estimate_model_context(state, "system") >= 10
    assert len(messages_only) == 3


def test_recent_turn_must_fit_within_recent_budget():
    messages = [
        HumanMessage(content="old"),
        AIMessage(content="old"),
        HumanMessage(content="recent " * 100),
        AIMessage(content="recent " * 100),
    ]
    with pytest.raises(RuntimeError, match="最新完整 Turn"):
        split_by_recent_budget(messages, recent_budget=10)


@pytest.mark.asyncio
async def test_agent_events_have_stable_turn_and_sequence():
    captured = []

    async def sink(event):
        captured.append(event)

    emitter = EventEmitter("session-1", sink)
    await emitter.emit("turn.started")
    await emitter.emit("model.started")
    assert [event.sequence for event in captured] == [1, 2]
    assert captured[0].turn_id == captured[1].turn_id


@pytest.mark.asyncio
async def test_force_compaction_keeps_recent_turn_and_updates_summary():
    manager = ContextManager(None, ContextBudget(
        context_window_tokens=1_000,
        trigger_ratio=0.8,
        summary_tokens=100,
        retain_ratio=0.1,
    ))

    async def fake_summarize(_self, old_summary, expired_messages):
        assert isinstance(old_summary, ContextSummary)
        assert old_summary.critical_context == ["更早一版摘要"]
        assert [message.id for message in expired_messages] == ["h1", "a1"]
        return ContextSummary(critical_context=["old question 已处理"])

    manager._summarize = MethodType(fake_summarize, manager)
    state = {
        "context_summary": {"critical_context": ["更早一版摘要"]},
        "active_messages": [
            HumanMessage(content="old question " * 10, id="h1"),
            AIMessage(content="old answer " * 10, id="a1"),
            HumanMessage(content="recent question " * 10, id="h2"),
            AIMessage(content="recent answer " * 10, id="a2"),
        ]
    }
    result = await manager.compact(state, "system", force=True)
    assert [message.id for message in result["active_messages"]] == ["h2", "a2"]
    assert result["context_summary"]["critical_context"] == ["old question 已处理"]
    assert result["summary_version"] == 1


@pytest.mark.asyncio
async def test_summary_failure_is_raised_without_mutating_state():
    manager = ContextManager(None, ContextBudget(
        context_window_tokens=1_000,
        trigger_ratio=0.8,
        summary_tokens=100,
        retain_ratio=0.1,
    ))

    async def failed_summary(_self, _old_summary, _expired_messages):
        raise RuntimeError("summary model unavailable")

    manager._summarize = MethodType(failed_summary, manager)
    original_messages = [
        HumanMessage(content="old question " * 10, id="h1"),
        AIMessage(content="old answer " * 10, id="a1"),
        HumanMessage(content="recent question " * 10, id="h2"),
        AIMessage(content="recent answer " * 10, id="a2"),
    ]
    state = {"active_messages": original_messages}
    with pytest.raises(RuntimeError, match="summary model unavailable"):
        await manager.compact(state, "system", force=True)
    assert state == {"active_messages": original_messages}


@pytest.mark.asyncio
async def test_agent_loop_emits_events_and_persists_only_active_context():
    class FakeModel:
        def bind_tools(self, _tools):
            return self

        async def astream(self, _messages, config=None):
            yield AIMessageChunk(content="ok")

    class FakeStore:
        def __init__(self):
            self.saved = None

        async def load_state(self, _session_id):
            return {}

        async def save_state(self, _session_id, state):
            self.saved = state

    store = FakeStore()
    manager = ContextManager(None, ContextBudget(
        context_window_tokens=10_000,
        trigger_ratio=0.8,
        summary_tokens=500,
        retain_ratio=0.1,
    ))
    agent = GameAgent(FakeModel(), [], manager, SkillRegistry.default(), store)
    agent._call_model = MethodType(GameAgent._call_model.__wrapped__, agent)
    captured = []

    async def sink(event):
        captured.append(event.event_type)

    # 绕过 Langfuse 装饰器，只验证 Harness 业务循环，测试不会访问外部观测服务。
    result = await agent.run_turn.__wrapped__(
        agent,
        "session-1",
        HumanMessage(content="hello", id="h1"),
        [],
        on_event=sink,
    )
    assert result["answer"] == "ok"
    assert "active_messages" in store.saved and "messages" not in store.saved
    assert captured == [
        "turn.started",
        "context.measured",
        "model.started",
        "model.token",
        "model.completed",
        "turn.completed",
    ]
