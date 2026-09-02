import asyncio

import pytest

from app.game_agent.stream import ModelStarted, TextDelta, TurnCompleted
from app.runtime.runmanager import RunAlreadyActive, RunManager, RunSnapshotEvent


async def make_message():
    return "user-message"


class RecordingStore:
    def __init__(self):
        self.assistant_messages = []

    async def record_assistant_message(self, session_id, answer):
        self.assistant_messages.append((session_id, answer))


class ControlledHarness:
    def __init__(self):
        self.started = asyncio.Event()
        self.release = asyncio.Event()

    async def stream_turn(self, session_id, message):
        assert message == "user-message"
        self.started.set()
        yield ModelStarted()
        await self.release.wait()
        yield TextDelta("final")
        yield TurnCompleted("final answer")


@pytest.mark.asyncio
async def test_disconnect_does_not_cancel_background_run():
    harness = ControlledHarness()
    store = RecordingStore()
    manager = RunManager(harness, store)
    run = await manager.start("session-1", make_message)

    subscription = manager.subscribe(run["run_id"])
    first = await anext(subscription)
    assert isinstance(first, RunSnapshotEvent)
    await subscription.aclose()  # 模拟浏览器刷新导致 SSE 连接关闭。

    await harness.started.wait()
    harness.release.set()
    result = await manager.wait(run["run_id"])

    assert result["status"] == "completed"
    assert result["final_answer"] == "final answer"
    assert store.assistant_messages == [("session-1", "final answer")]


@pytest.mark.asyncio
async def test_one_session_rejects_a_second_active_run():
    harness = ControlledHarness()
    manager = RunManager(harness, RecordingStore())
    first = await manager.start("session-1", make_message)
    await harness.started.wait()

    with pytest.raises(RunAlreadyActive) as error:
        await manager.start("session-1", make_message)
    assert error.value.run_id == first["run_id"]

    harness.release.set()
    await manager.wait(first["run_id"])
    second = await manager.start("session-1", make_message)
    assert second["run_id"] != first["run_id"]
    harness.release.set()
    await manager.wait(second["run_id"])


@pytest.mark.asyncio
async def test_failure_becomes_queryable_run_state():
    class FailingHarness:
        async def stream_turn(self, session_id, message):
            yield ModelStarted()
            raise RuntimeError("provider unavailable")

    manager = RunManager(FailingHarness(), RecordingStore())
    run = await manager.start("session-1", make_message)
    result = await manager.wait(run["run_id"])

    assert result["status"] == "failed"
    assert result["phase"] == "failed"
    assert result["error"] == "provider unavailable"


@pytest.mark.asyncio
async def test_latest_run_exposes_running_and_terminal_status():
    harness = ControlledHarness()
    manager = RunManager(harness, RecordingStore())
    run = await manager.start("session-1", make_message)
    await harness.started.wait()

    running = await manager.get_latest("session-1")
    assert running["run_id"] == run["run_id"]
    assert running["status"] == "running"

    harness.release.set()
    await manager.wait(run["run_id"])
    completed = await manager.get_latest("session-1")
    assert completed["status"] == "completed"


@pytest.mark.asyncio
async def test_cancel_stops_background_run_and_keeps_partial_answer():
    class StreamingHarness:
        def __init__(self):
            self.partial_sent = asyncio.Event()
            self.release = asyncio.Event()

        async def stream_turn(self, session_id, message):
            yield TextDelta("已经生成的内容")
            self.partial_sent.set()
            await self.release.wait()
            yield TurnCompleted("不应生成到这里")

    harness = StreamingHarness()
    store = RecordingStore()
    manager = RunManager(harness, store)
    run = await manager.start("session-1", make_message)
    await harness.partial_sent.wait()

    cancelled = await manager.cancel(run["run_id"])

    assert cancelled["status"] == "cancelled"
    assert cancelled["partial_answer"] == "已经生成的内容"
    assert cancelled["final_answer"] is None
    assert store.assistant_messages == [("session-1", "已经生成的内容")]
