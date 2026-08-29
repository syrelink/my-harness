"""GameRover 的无 LangGraph Agent 循环。

一轮请求的主路径是显式的 while 循环：

    压缩测压 → Agent →（可选 工具执行 → 压缩测压 → Agent）→ END。

持久化遵循 DSH / Codex 的「状态与日志分离」原则：
- 会话级字段（active_messages / context_summary / turn_count 等）持久化；
- 当前图片和运行事件只在本轮存在，不进入持久状态。

可观测性由 Langfuse 提供（@observe 生成 Trace/Span，CallbackHandler 采集 LLM 调用）。
"""

from __future__ import annotations

import asyncio
import json
import os
import time
from collections.abc import AsyncIterator
from uuid import uuid4

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, ToolMessage
from langchain_openai import ChatOpenAI

from app.game_agent.multimodal import AttachmentLoader, hydrate_current_images
from app.game_agent.memory import (
    ContextBudget,
    ContextManager,
)
from app.game_agent.observability import langfuse_handler, observe
from app.game_agent.prompts import build_agent_system_prompt
from app.game_agent.skills import SkillRegistry
from app.game_agent.stream import (
    AgentStreamEvent,
    ModelCompleted,
    ModelStarted,
    TextDelta,
    ToolCompleted,
    ToolStarted,
    TurnCompleted,
)
from app.game_agent.tools import AGENT_TOOLS, skill_registry


def create_model(prefix: str = "GAME_ASSISTANT") -> ChatOpenAI:
    """按环境变量创建一个兼容 OpenAI 协议的聊天模型客户端。

    ``prefix`` 让同一项目可以给不同用途的模型配置不同变量。例如默认读取
    ``GAME_ASSISTANT_MODEL``；没有配置时再退回 ``DEEPSEEK_MODEL``。这里只创建客户端，
    不会立即向模型服务发送请求。
    """
    return ChatOpenAI(
        model=os.getenv(f"{prefix}_MODEL") or os.getenv("DEEPSEEK_MODEL", "deepseek-chat"),
        api_key=os.getenv(f"{prefix}_API_KEY") or os.getenv("DEEPSEEK_API_KEY"),
        base_url=os.getenv(f"{prefix}_BASE_URL") or os.getenv("DEEPSEEK_BASE_URL"),
        temperature=0.2,
    )


class AgentHarness:
    """单 Agent Harness：普通 dict 状态 + while 循环 + 手动工具执行。"""

    def __init__(
        self,
        model: BaseChatModel,
        tools: list,
        context_manager: ContextManager,
        skill_registry: SkillRegistry,
        session_store,
        attachment_loader: AttachmentLoader | None = None,
    ):
        """注入 Agent 运行所需依赖，并提前准备模型工具协议。

        - ``model``：真正负责推理和生成的聊天模型；
        - ``tools``：模型可以请求调用的工具列表；
        - ``context_manager``：Token 测压、历史压缩和模型上下文组装；
        - ``skill_registry``：启动时扫描到的 Skill 目录；
        - ``session_store``：PostgreSQL 会话状态；
        - ``attachment_loader``：按引用从 MinIO 临时读取当前图片。
        """
        self.model = model
        # 字典让工具执行阶段可以用模型返回的 name 快速找到对应工具。
        self.tools = {tool.name: tool for tool in tools}
        # 调度策略属于宿主，不写进模型 Tool Schema。未知或未声明的工具默认串行。
        self.parallel_tools = {
            tool.name
            for tool in tools
            if (tool.metadata or {}).get("execution_mode") == "parallel"
        }
        # bind_tools 只把工具 Schema 告诉模型；此时不会执行任何工具。
        self.model_with_tools = model.bind_tools(tools)
        self.context_manager = context_manager
        self.skill_registry = skill_registry
        self.session_store = session_store
        self.attachment_loader = attachment_loader
        self.agent_system_prompt = build_agent_system_prompt(skill_registry.catalog_prompt())
        self.default_timeout = float(os.getenv("GAME_TOOL_TIMEOUT_SECONDS", "35"))
        self.max_parallel_tools = max(1, int(os.getenv("GAME_MAX_PARALLEL_TOOLS", "4")))

    # ---------- 会话状态 ----------
    async def load_state(self, session_id: str) -> dict:
        """从 PostgreSQL 读取某个会话的跨轮状态。

        返回的普通 dict 主要包含 active_messages、ContextSummary 和版本计数。首次
        对话不存在记录时，SessionStore 返回空字典。
        """
        return await self.session_store.load_state(session_id)

    async def save_state(self, session_id: str, state: dict) -> None:
        """把一轮完成后的跨轮状态写回 PostgreSQL。"""
        await self.session_store.save_state(session_id, state)

    # ---------- 一轮主循环 ----------
    # Langfuse 当前 SDK 支持异步生成器：Observation 会覆盖从第一次迭代到流结束的整轮。
    # 关闭自动输出采集，避免把每个 TextDelta 重复写入 Trace；模型输出由 Callback 记录。
    @observe(name="agent_turn", as_type="agent", capture_output=False)
    async def stream_turn(
        self,
        session_id: str,
        user_message,
    ) -> AsyncIterator[AgentStreamEvent]:
        """执行完整 Agent Turn，并直接产出调用方可消费的异步事件流。

        整体流程：

            读取 PostgreSQL State
                      ↓
                追加 HumanMessage
                      ↓
            ┌────── Agent Loop ──────┐
            │ 模型调用前检查上下文   │
            │          ↓             │
            │ 必要时滚动压缩历史     │
            │          ↓             │
            │ 流式调用模型           │
            │          ↓             │
            │ 有 tool_calls？         │
            │  ├─ 否 → 退出循环      │
            │  └─ 是 → 执行工具      │
            │             ↓          │
            │        追加 ToolMessage│
            │             ↓          │
            │        回到循环顶部    │
            └────────────────────────┘
                      ↓
                保存最新 State
                      ↓
              yield TurnCompleted

        参数：
        - ``session_id``：数据库会话分区键；
        - ``user_message``：本轮只含文字和结构化图片引用的 HumanMessage；

        while 循环的退出条件是模型不再返回 tool_calls；跨轮 state 在返回前写入 PostgreSQL。
        """
        # —— 持久层：从 DB 读会话状态（只含跨轮字段） ——
        state = await self.load_state(session_id)
        state["active_messages"] = list(state.get("active_messages", [])) + [user_message]
        state["turn_count"] = int(state.get("turn_count", 0)) + 1

        last_response = AIMessage(content="")

        while True:
            # 一次循环代表一次“模型决策步”：测压 → 调模型 →（可能）执行工具。
            # 每次调用模型前测压，超阈值才压缩。
            compaction_result = await self.context_manager.compact(
                state,
                self.agent_system_prompt,
            )
            # compact() 返回本次检查结果，不是完整 state：达到阈值时包含新的近期消息和
            # ContextSummary；没有达到阈值时不修改 state。
            messages = compaction_result.get("active_messages")
            if messages is not None:
                state["active_messages"] = messages
            # 只把需要跨轮恢复的字段写回 state。
            for key in ("context_summary", "compaction_count", "summary_version"):
                if key in compaction_result:
                    state[key] = compaction_result[key]
            # Provider Stream → Agent Stream：合并底层消息分片，同时向调用方输出文字事件。
            response = None
            model_started_at = time.perf_counter()
            yield ModelStarted()
            async for chunk in self._stream_model(state, session_id):
                response = chunk if response is None else response + chunk
                if isinstance(chunk.content, str) and chunk.content:
                    yield TextDelta(chunk.content)
            yield ModelCompleted(int((time.perf_counter() - model_started_at) * 1000))
            response = response or AIMessage(content="")
            if not response.id:
                response.id = str(uuid4())
            last_response = response
            state["active_messages"] = list(state["active_messages"]) + [response]

            if not response.tool_calls:
                # 普通文本回答表示任务结束；有 tool_calls 则继续执行工具并进入下一圈。
                break

            # ToolExecution：手动执行本批工具。工具调用 JSON 不直接给前端，只发稳定状态事件。
            for call in response.tool_calls:
                yield ToolStarted(call.get("name", "unknown"))
            tool_started_at = time.perf_counter()
            tool_messages = await self._execute_tools(response)
            tool_elapsed_ms = int((time.perf_counter() - tool_started_at) * 1000)
            for call in response.tool_calls:
                yield ToolCompleted(call.get("name", "unknown"), elapsed_ms=tool_elapsed_ms)
            state["active_messages"] = list(state["active_messages"]) + tool_messages

        await self.save_state(session_id, state)
        yield TurnCompleted(last_response.text)

    # ---------- 模型调用 ----------
    # 这是 agent_turn 的子 Observation；其中的 LangChain CallbackHandler 还会再创建一个
    # Generation 子记录，所以可以同时看业务步骤耗时和真正的模型 Usage/TTFT。
    @observe(name="model_call", capture_output=False)
    async def _stream_model(
        self,
        state: dict,
        session_id: str,
    ) -> AsyncIterator:
        """组装临时模型上下文，并原样产出 LangChain 消息分片。

        ``model_context`` 是临时视图：System Prompt + ContextSummary + 近期消息。当前 Turn
        有图片时会从最新 HumanMessage 的 image block 读取引用，并临时转成 image_url，
        但 Base64 不会写回持久 state。

        这里是 Provider Stream 的边界；上层 ``stream_turn`` 再把它转换为 AgentStreamEvent。
        """
        model_context = self.context_manager.build_model_context(state, self.agent_system_prompt)
        model_context = await hydrate_current_images(
            model_context,
            session_id=session_id,
            loader=self.attachment_loader,
        )
        # CallbackHandler 监听 LangChain 模型事件，在 model_call 下创建 Generation，并采集
        # 模型输入输出、Token、耗时和错误。未配置 Langfuse 时 config 为 None。
        config = {"callbacks": [langfuse_handler]} if langfuse_handler else None
        async for chunk in self.model_with_tools.astream(model_context, config=config):
            yield chunk

    # ---------- 工具执行 ----------
    # 当前 Observation 代表“一批工具调用”。如果未来需要按工具聚合指标，可以进一步把
    # 单次工具调用抽成一个带 @observe(as_type="tool") 的函数。
    @observe(name="tool_execution")
    async def _execute_tools(self, response: AIMessage):
        """并行执行一批只读工具，并按模型原始调用顺序返回 ToolMessage。

        ToolMessage 会放回消息历史供下一次模型读取。单个工具具有超时和异常降级，
        失败会转成结构化 ToolMessage，而不是直接打断整轮。只要本批包含未声明为
        parallel 的工具，就保守地整批串行；当前 read_skill/web_search 都是只读工具。
        """
        calls = response.tool_calls
        can_run_in_parallel = all(
            call.get("name") in self.parallel_tools for call in calls
        )
        if not can_run_in_parallel:
            return [await self._execute_one_tool(call) for call in calls]

        semaphore = asyncio.Semaphore(self.max_parallel_tools)

        async def execute_limited(call: dict) -> ToolMessage:
            async with semaphore:
                return await self._execute_one_tool(call)

        # gather 允许执行乱序完成，但返回值严格保持 calls 的原始顺序。
        return await asyncio.gather(*(execute_limited(call) for call in calls))

    async def _execute_one_tool(self, call: dict) -> ToolMessage:
        """执行一个工具调用，并把成功、超时或异常统一转换为 ToolMessage。"""
        name = call.get("name", "unknown")
        args = call.get("args", {})
        tool = self.tools.get(name)
        if tool is None:
            content = json.dumps({
                "error": f"未知工具：{name}",
                "error_type": "unknown_tool",
                "tool": name,
            }, ensure_ascii=False)
            status = "error"
        else:
            try:
                result = await asyncio.wait_for(
                    tool.ainvoke(args), timeout=self.default_timeout
                )
                content = result if isinstance(result, str) else json.dumps(
                    result, ensure_ascii=False, default=str
                )
                status = "success"
            except asyncio.TimeoutError:
                content = json.dumps({
                    "error": f"{name} 执行超时",
                    "error_type": "tool_timeout",
                    "tool": name,
                }, ensure_ascii=False)
                status = "error"
            except Exception as exc:
                content = json.dumps({
                    "error": str(exc),
                    "error_type": type(exc).__name__,
                    "tool": name,
                }, ensure_ascii=False)
                status = "error"

        return ToolMessage(
            content=content,
            tool_call_id=call.get("id", ""),
            name=name,
            status="error" if status == "error" else "success",
            id=str(uuid4()),
        )

def build_harness(session_store, attachment_loader: AttachmentLoader | None = None) -> AgentHarness:
    """在 FastAPI 启动时组装并返回一个可复用的 Agent Harness 实例。

    这是项目的 Composition Root：集中创建模型、读取上下文预算、创建 ContextManager，
    再注入全局 Tool/Skill Registry、数据库和附件加载器。请求处理函数只调用组装好的
    Harness，不需要了解这些对象如何构造。
    """
    model = create_model()
    budget = ContextBudget.from_env()
    context_manager = ContextManager(model, budget)
    return AgentHarness(
        model=model,
        tools=AGENT_TOOLS,
        context_manager=context_manager,
        skill_registry=skill_registry,
        session_store=session_store,
        attachment_loader=attachment_loader,
    )
