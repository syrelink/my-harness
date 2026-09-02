import asyncio
import json

import pytest
from langchain.tools import tool

from app.runtime.toolruntime import ToolRegistry


@tool
async def echo_tool(value: str) -> str:
    """原样返回输入。"""

    return value


@tool
async def slow_tool() -> str:
    """用于验证工具超时。"""

    await asyncio.sleep(0.05)
    return "too late"


@tool
async def broken_tool() -> str:
    """用于验证工具内部异常会被统一降级。"""

    raise RuntimeError("Mock 工具执行失败")


def test_registry_rejects_duplicate_tool_names():
    registry = ToolRegistry()
    registry.register(echo_tool)

    with pytest.raises(ValueError, match="工具已注册"):
        registry.register(echo_tool)


@pytest.mark.asyncio
async def test_tool_execution_applies_spec_and_returns_result():
    registry = ToolRegistry()
    registry.register(
        echo_tool,
        execution_mode="parallel",
        idempotent=True,
        max_result_chars=3,
    )
    result = await registry.execute(
        {"id": "call-1", "name": "echo_tool", "args": {"value": "hello"}},
        step=1,
    )

    assert result.status == "success"
    assert result.name == "echo_tool"
    assert result.content.startswith("hel\n[结果已截断")


@pytest.mark.asyncio
async def test_tool_timeout_becomes_error_message():
    registry = ToolRegistry()
    registry.register(slow_tool, timeout_seconds=0.001)
    result = await registry.execute(
        {"id": "call-2", "name": "slow_tool", "args": {}},
        step=1,
    )

    payload = json.loads(result.content)
    assert result.status == "error"
    assert payload["error_code"] == "tool_timeout"


@pytest.mark.asyncio
async def test_unknown_tool_becomes_structured_error():
    registry = ToolRegistry()

    result = await registry.execute(
        {"id": "call-3", "name": "missing_tool", "args": {}},
        step=1,
    )

    payload = json.loads(result.content)
    assert result.status == "error"
    assert payload["error_code"] == "unknown_tool"
    assert payload["tool"] == "missing_tool"


@pytest.mark.asyncio
async def test_invalid_arguments_have_stable_error_code():
    registry = ToolRegistry()
    registry.register(echo_tool)

    result = await registry.execute(
        {"id": "call-4", "name": "echo_tool", "args": {}},
        step=1,
    )

    payload = json.loads(result.content)
    assert result.status == "error"
    assert payload["error_code"] == "invalid_args"


@pytest.mark.asyncio
async def test_tool_exception_has_stable_error_code():
    registry = ToolRegistry()
    registry.register(broken_tool)

    result = await registry.execute(
        {"id": "call-5", "name": "broken_tool", "args": {}},
        step=1,
    )

    payload = json.loads(result.content)
    assert result.status == "error"
    assert payload == {
        "error_code": "execution_failed",
        "message": "Mock 工具执行失败",
        "tool": "broken_tool",
    }


@pytest.mark.asyncio
async def test_mixed_parallel_and_serial_calls_are_grouped_in_order():
    events: list[str] = []

    @tool
    async def parallel_one() -> str:
        """第一个并行测试工具。"""
        events.append("parallel_one:start")
        await asyncio.sleep(0.02)
        events.append("parallel_one:end")
        return "one"

    @tool
    async def parallel_two() -> str:
        """第二个并行测试工具。"""
        events.append("parallel_two:start")
        await asyncio.sleep(0.01)
        events.append("parallel_two:end")
        return "two"

    @tool
    async def serial_middle() -> str:
        """位于两个并发组之间的串行工具。"""
        events.append("serial:start")
        await asyncio.sleep(0)
        events.append("serial:end")
        return "serial"

    registry = ToolRegistry()
    registry.register(parallel_one, execution_mode="parallel")
    registry.register(parallel_two, execution_mode="parallel")
    registry.register(serial_middle, execution_mode="serial")

    results = await registry.execute_many(
        [
            {"id": "call-1", "name": "parallel_one", "args": {}},
            {"id": "call-2", "name": "parallel_two", "args": {}},
            {"id": "call-3", "name": "serial_middle", "args": {}},
            {"id": "call-4", "name": "parallel_one", "args": {}},
        ],
        step=1,
        max_parallel_tools=2,
    )

    assert [result.content for result in results] == ["one", "two", "serial", "one"]
    assert events.index("parallel_two:start") < events.index("parallel_one:end")
    assert events.index("serial:start") > events.index("parallel_one:end")
    assert events.index("serial:start") > events.index("parallel_two:end")
    assert events.index("parallel_one:start", 1) > events.index("serial:end")
