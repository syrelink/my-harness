"""用 Mock 依赖验证 Agent → Tool → Agent 的完整闭环。"""

import pytest
from langchain.tools import tool
from langchain_core.messages import AIMessageChunk, HumanMessage, ToolMessage

from app.game_agent.agent import AgentHarness
from app.game_agent.stream import (
    ModelCompleted,
    ModelStarted,
    TextDelta,
    ToolCompleted,
    ToolStarted,
    TurnCompleted,
)
from app.runtime.toolruntime import ToolRegistry


@tool
async def mock_game_search(query: str) -> str:
    """返回固定游戏资讯，不访问真实网络。"""

    return f"{query}：Mock 搜索结果"


class MockModel:
    """第一次请求工具，第二次读取 ToolMessage 并生成最终回答。"""

    def __init__(self) -> None:
        self.calls: list[list] = []

    def bind_tools(self, tools):
        self.bound_tools = tools
        return self

    async def astream(self, messages, config=None):
        self.calls.append(list(messages))

        if len(self.calls) == 1:
            yield AIMessageChunk(
                content="",
                tool_call_chunks=[{
                    "name": "mock_game_search",
                    "args": '{"query":"原神最新版本"}',
                    "id": "call-1",
                    "index": 0,
                }],
            )
            return

        tool_message = next(
            message for message in messages if isinstance(message, ToolMessage)
        )
        yield AIMessageChunk(content=f"根据工具结果：{tool_message.content}")


class MockContextManager:
    """测试时不触发上下文压缩，只保留 Agent Loop 必需接口。"""

    async def compact(self, state, system_prompt):
        return {"compacted": False}

    def build_model_context(self, state, system_prompt):
        return list(state.get("active_messages", []))


class MockSkillRegistry:
    def catalog_prompt(self) -> str:
        return "无测试 Skill"


class MockSessionStore:
    """使用内存代替 PostgreSQL，方便检查最终保存的消息状态。"""

    def __init__(self) -> None:
        self.state: dict = {}

    async def load_state(self, session_id: str) -> dict:
        return dict(self.state)

    async def save_state(self, session_id: str, state: dict) -> None:
        self.state = dict(state)


@pytest.mark.asyncio
async def test_agent_uses_tool_result_in_second_model_call():
    model = MockModel()
    store = MockSessionStore()
    registry = ToolRegistry()
    registry.register(mock_game_search, execution_mode="parallel")
    harness = AgentHarness(
        model=model,
        tool_registry=registry,
        context_manager=MockContextManager(),
        skill_registry=MockSkillRegistry(),
        session_store=store,
    )

    events = [
        event
        async for event in harness.stream_turn(
            "session-1",
            HumanMessage(content="原神最近更新了什么？"),
        )
    ]

    # 模型调用了两次：第一次决定使用工具，第二次读取工具结果并回答。
    assert len(model.calls) == 2
    assert any(isinstance(message, ToolMessage) for message in model.calls[1])

    # 前端所需的模型、工具和文本事件都按完整流程产生。
    assert [type(event) for event in events] == [
        ModelStarted,
        ModelCompleted,
        ToolStarted,
        ToolCompleted,
        ModelStarted,
        TextDelta,
        ModelCompleted,
        TurnCompleted,
    ]
    assert events[-1].answer == "根据工具结果：原神最新版本：Mock 搜索结果"

    # 最终状态包含 HumanMessage、模型工具调用、ToolMessage 和最终回答。
    assert len(store.state["active_messages"]) == 4

