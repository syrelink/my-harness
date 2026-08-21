"""GameRover 的无 LangGraph Agent 循环。

一轮请求的主路径是显式的 while 循环：

    压缩测压 → Agent →（可选 工具执行 → 压缩测压 → Agent）→ END。

持久化遵循 DSH / Codex 的「状态与日志分离」原则：
- 会话级字段（active_messages / context_summary / turn_count 等）持久化；
- 本轮临时字段（当前图片、token 用量、工具轨迹、本轮指标）只作为局部变量，
  每轮重新声明、天然归零，绝不进入持久状态。

可观测性由 Langfuse 提供（@observe 生成 Trace/Span，CallbackHandler 采集 LLM 调用）。
"""

from __future__ import annotations

import asyncio
import json
import os
import time
from uuid import uuid4

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, ToolMessage
from langchain_openai import ChatOpenAI

from app.game_agent.multimodal import AttachmentLoader, hydrate_current_images
from app.game_agent.events import EventEmitter, EventSink
from app.game_agent.memory import (
    ContextBudget,
    ContextManager,
    context_summary_from_state,
)
from app.game_agent.models import ToolTrace, TurnTokenUsage
from app.game_agent.observability import langfuse_handler, observe
from app.game_agent.prompts import build_agent_system_prompt
from app.game_agent.skills import SkillRegistry
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


def truncate_tool_payload(content: str, token_budget: int) -> tuple[str, bool]:
    """把工具结果裁到预算内，返回 ``(裁剪后内容, 是否发生裁剪)``。

    工具正文最终会成为 ToolMessage，再进入下一次模型输入。如果网页结果过长，会同时
    增加 Token、延迟和费用。这里用“约 4 字符 = 1 Token”换算粗略字符上限；这是保护
    上下文的快速估算，并不冒充供应商精确 Tokenizer。

    裁剪优先级：错误 JSON 只保留关键错误字段；搜索 JSON 优先减少 results；其他文本
    最后使用字符串截断，并附加 ``[truncated]`` 标记。
    """
    rough_limit = max(32, token_budget * 4)
    if len(content) <= rough_limit:
        return content, False
    try:
        payload = json.loads(content)
        if payload.get("error"):
            for message_limit in (240, 120, 60):
                compact = json.dumps({
                    "error": str(payload.get("error", ""))[:message_limit],
                    "error_type": payload.get("error_type", "tool_error"),
                    "tool": payload.get("tool", "unknown"),
                }, ensure_ascii=False)
                if len(compact) <= rough_limit:
                    return compact, True
            return json.dumps({"error": "tool_error"}, ensure_ascii=False), True

        results = payload.get("results", [])
        if isinstance(results, list):
            payload["results"] = results[:3]
            compact = json.dumps(payload, ensure_ascii=False)
            if len(compact) <= rough_limit:
                return compact, True
    except (json.JSONDecodeError, AttributeError, TypeError):
        pass

    suffix = "\n[truncated]"
    return content[: max(0, rough_limit - len(suffix))] + suffix, True


class GameAgent:
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
        # bind_tools 只把工具 Schema 告诉模型；此时不会执行任何工具。
        self.model_with_tools = model.bind_tools(tools)
        self.context_manager = context_manager
        self.budget = context_manager.budget
        self.skill_registry = skill_registry
        self.session_store = session_store
        self.attachment_loader = attachment_loader
        self.agent_system_prompt = build_agent_system_prompt(skill_registry.catalog_prompt())
        self.default_timeout = float(os.getenv("GAME_TOOL_TIMEOUT_SECONDS", "35"))
        self.tool_result_tokens = int(os.getenv("GAME_TOOL_RESULT_BUDGET_TOKENS", "2500"))

    # ---------- 会话状态 ----------
    async def load_state(self, session_id: str) -> dict:
        """从 PostgreSQL 读取某个会话的跨轮状态。

        返回的普通 dict 主要包含 active_messages、ContextSummary 和版本计数。首次
        对话不存在记录时，SessionStore 返回空字典。
        """
        return await self.session_store.load_state(session_id)

    async def save_state(self, session_id: str, state: dict) -> None:
        """把一轮完成后的跨轮状态写回 PostgreSQL。

        ToolTrace、本轮 Token Usage 等只存在于 ``run_turn`` 局部变量里，不会传给这里，
        因而不会污染下一轮 Agent 上下文。
        """
        await self.session_store.save_state(session_id, state)

    async def force_compact(self, session_id: str) -> dict:
        """由管理接口强制压缩一个已有会话，并持久化压缩结果。

        它不调用主模型回答用户，只运行 ContextManager 的压缩流程。``compact`` 返回的是
        State 增量，因此这里把新的消息列表和四个跨轮字段显式合回原 state。
        """
        state = await self.load_state(session_id)
        if not state.get("active_messages"):
            return {"compacted": False, "reason": "session not found"}
        update = await self.context_manager.compact(
            state,
            self.agent_system_prompt,
            force=True,
        )
        messages = update.pop("active_messages", None)
        if messages is not None:
            state["active_messages"] = messages
        for key in ("context_summary", "compaction_count", "summary_version"):
            if key in update:
                state[key] = update[key]
        await self.save_state(session_id, state)
        return update

    # ---------- 一轮主循环 ----------
    # 装饰器在不修改业务流程的前提下，把整个 run_turn 包装为名为 agent_turn 的
    # Langfuse 根 Observation。大致等价于 run_turn = observe(name="agent_turn")(run_turn)。
    @observe(name="agent_turn")
    async def run_turn(
        self,
        session_id: str,
        user_message,
        current_attachments: list,
        force_compaction: bool = False,
        on_event: EventSink | None = None,
    ) -> dict:
        """执行从一个用户输入到最终回答的完整 Agent Turn。

        参数：
        - ``session_id``：数据库会话分区键；
        - ``user_message``：本轮只含文字/附件引用的 HumanMessage；
        - ``current_attachments``：只在本轮临时 Hydrate 的图片引用；
        - ``force_compaction``：是否在首次模型调用前强制压缩；
        - ``on_event``：可选事件消费者，SSE、日志和测试可复用同一协议。

        while 循环的退出条件是模型不再返回 tool_calls。返回值包含最终答案以及本轮临时
        指标；真正需要跨轮恢复的 state 会在返回前单独写入 PostgreSQL。
        """
        events = EventEmitter(session_id, on_event)
        await events.emit("turn.started")

        # —— 持久层：从 DB 读会话状态（只含跨轮字段） ——
        state = await self.load_state(session_id)
        state["active_messages"] = list(state.get("active_messages", [])) + [user_message]
        state["turn_count"] = int(state.get("turn_count", 0)) + 1
        started = time.perf_counter()

        # —— 临时层：本轮局部变量，每轮重新声明、天然归零，不进入持久状态 ——
        turn_token_usage = TurnTokenUsage()
        tool_trace: list[dict] = []
        tool_rounds = 0
        compacted = False
        context_metrics: dict = {}
        last_response = AIMessage(content="")

        while True:
            # 一次循环代表一次“模型决策步”：测压 → 调模型 →（可能）执行工具。
            # 每次调用模型前测压，超阈值才压缩。
            update = await self.context_manager.compact(
                state,
                self.agent_system_prompt,
                force=bool(force_compaction) and tool_rounds == 0,
            )
            messages = update.pop("active_messages", None)
            if messages is not None:
                state["active_messages"] = messages
            # 只把跨轮字段写回 state；compacted / context_metrics 等是本轮观测结果。
            # 等临时字段留在本轮局部变量，不进入持久状态。
            for key in ("context_summary", "compaction_count", "summary_version"):
                if key in update:
                    state[key] = update[key]
            compacted = compacted or bool(update.get("compacted", False))
            measured = update.get("context_metrics", {})
            attempted = bool(update.get("compaction_attempted", False))
            await events.emit(
                "context.measured",
                estimated_input_tokens=(
                    measured.get("tokens_before_compaction")
                    or measured.get("model_input_tokens", 0)
                ),
                trigger_tokens=measured.get("trigger_tokens", 0),
                will_compact=attempted,
            )
            if attempted:
                await events.emit(
                    "compaction.completed",
                    compacted=bool(update.get("compacted", False)),
                    tokens_before=measured.get("tokens_before_compaction", 0),
                    tokens_after=measured.get("tokens_after_compaction", 0),
                    converged=measured.get("converged", False),
                )

            # Agent：调模型（流式），返回 AIMessage 与 usage。
            await events.emit("model.started")
            response, usage = await self._call_model(state, current_attachments, session_id, events)
            await events.emit("model.completed", **usage)
            last_response = response
            state["active_messages"] = list(state["active_messages"]) + [response]
            turn_token_usage = self._accumulate(turn_token_usage, usage)
            context_metrics = measured

            if not response.tool_calls:
                # 普通文本回答表示任务结束；有 tool_calls 则继续执行工具并进入下一圈。
                break

            # ToolExecution：手动执行本批工具。
            tool_messages, traces = await self._execute_tools(response, events)
            state["active_messages"] = list(state["active_messages"]) + tool_messages
            tool_trace.extend([t.model_dump() for t in traces])
            tool_rounds += 1

        elapsed_ms = int((time.perf_counter() - started) * 1000)
        # state 中此刻只含跨轮字段；本轮 token / trace / 指标等临时数据不进持久状态。
        await self.save_state(session_id, state)
        await events.emit("turn.completed", elapsed_ms=elapsed_ms)

        return {
            "answer": last_response.text,
            "tool_trace": tool_trace,
            "context_metrics": context_metrics,
            "token_usage": turn_token_usage,
            "context_summary": context_summary_from_state(state),
            "compacted": compacted,
            "elapsed_ms": elapsed_ms,
        }

    # ---------- 模型调用 ----------
    # 这是 agent_turn 的子 Observation；其中的 LangChain CallbackHandler 还会再创建一个
    # Generation 子记录，所以可以同时看业务步骤耗时和真正的模型 Usage/TTFT。
    @observe(name="model_call")
    async def _call_model(
        self,
        state: dict,
        current_attachments: list,
        session_id: str,
        events: EventEmitter,
    ):
        """组装一次模型请求，流式收集分片并返回 ``(AIMessage, usage)``。

        ``model_context`` 是临时视图：System Prompt + ContextSummary + 近期消息。当前 Turn
        有图片时会从 MinIO 读取并临时加入这个列表，但 Base64 不会写回持久 state。
        """
        model_context = self.context_manager.build_model_context(state, self.agent_system_prompt)
        model_context = await hydrate_current_images(
            model_context,
            current_attachments,
            session_id=session_id,
            loader=self.attachment_loader,
        )
        response = None
        # CallbackHandler 监听 LangChain 模型事件，在 model_call 下创建 Generation，并采集
        # 模型输入输出、Token、耗时和错误。未配置 Langfuse 时 config 为 None。
        config = {"callbacks": [langfuse_handler]} if langfuse_handler else None
        async for chunk in self.model_with_tools.astream(model_context, config=config):
            # LangChain 的 AIMessageChunk 支持相加，最终合成为一个完整 AIMessage。
            response = chunk if response is None else response + chunk
            if isinstance(chunk.content, str) and chunk.content:
                # 文本一到达就推给 SSE；Tool Call 参数分片不会当作回答文本发送。
                await events.emit("model.token", content=chunk.content)
        response = response or AIMessage(content="")
        if not response.id:
            response.id = str(uuid4())
        usage = getattr(response, "usage_metadata", None) or {}
        return response, usage

    # ---------- 工具执行 ----------
    # 当前 Observation 代表“一批工具调用”。如果未来需要按工具聚合指标，可以进一步把
    # 单次工具调用抽成一个带 @observe(as_type="tool") 的函数。
    @observe(name="tool_execution")
    async def _execute_tools(self, response: AIMessage, events: EventEmitter):
        """按模型返回的 tool_calls 顺序执行工具，并生成 ToolMessage 与 ToolTrace。

        ToolMessage 会放回消息历史供下一次模型读取；ToolTrace 只供本轮接口展示和调试。
        单个工具具有超时和异常降级，失败会转成结构化 ToolMessage，而不是直接打断整轮。

        """
        tool_messages: list[ToolMessage] = []
        traces: list[ToolTrace] = []
        for call in response.tool_calls:
            name = call.get("name", "unknown")
            args = call.get("args", {})
            tool = self.tools.get(name)
            started = time.perf_counter()
            await events.emit("tool.started", tool=name)
            if tool is None:
                content = json.dumps({"error": f"未知工具：{name}", "error_type": "unknown_tool", "tool": name}, ensure_ascii=False)
                status = "error"
                error_type = "unknown_tool"
            else:
                try:
                    # wait_for 提供工具级超时边界，避免一个外部服务无限阻塞 Agent Loop。
                    result = await asyncio.wait_for(tool.ainvoke(args), timeout=self.default_timeout)
                    content = result if isinstance(result, str) else json.dumps(result, ensure_ascii=False, default=str)
                    status = "success"
                    error_type = None
                except asyncio.TimeoutError:
                    content = json.dumps({"error": f"{name} 执行超时", "error_type": "tool_timeout", "tool": name}, ensure_ascii=False)
                    status = "error"
                    error_type = "tool_timeout"
                except Exception as exc:
                    content = json.dumps({"error": str(exc), "error_type": type(exc).__name__, "tool": name}, ensure_ascii=False)
                    status = "error"
                    error_type = type(exc).__name__
            content, truncated = truncate_tool_payload(content, self.tool_result_tokens)
            latency_ms = int((time.perf_counter() - started) * 1000)
            await events.emit(
                "tool.completed",
                tool=name,
                status=status,
                latency_ms=latency_ms,
                error_type=error_type,
            )
            traces.append(ToolTrace(
                name=name, arguments=args, status=status, preview=content[:1200],
                latency_ms=latency_ms, error_type=error_type, truncated=truncated,
            ))
            tool_messages.append(ToolMessage(
                content=content, tool_call_id=call.get("id", ""), name=name,
                status="error" if status == "error" else "success",
                id=str(uuid4()),
            ))
        return tool_messages, traces

    # ---------- Token 与指标 ----------
    @staticmethod
    def _accumulate(current: TurnTokenUsage, usage: dict) -> TurnTokenUsage:
        """把一次模型 Usage 累加到当前 Turn 的 Token 汇总中。

        一个 Turn 可能经历多次“模型 → 工具 → 模型”，所以不能只看最后一次调用。
        ``model_copy`` 创建新 Pydantic 对象，避免原地修改旧统计。
        """
        input_tokens = int(usage.get("input_tokens") or 0)
        output_tokens = int(usage.get("output_tokens") or 0)
        total = int(usage.get("total_tokens") or (input_tokens + output_tokens))
        return current.model_copy(update={
            "input_tokens": current.input_tokens + input_tokens,
            "output_tokens": current.output_tokens + output_tokens,
            "total_tokens": current.total_tokens + total,
            "model_calls": current.model_calls + 1,
            "estimated_calls": current.estimated_calls + (0 if usage.get("input_tokens") else 1),
        })

def build_game_assistant(session_store, attachment_loader: AttachmentLoader | None = None) -> GameAgent:
    """在 FastAPI 启动时组装并返回一个可复用的 ``GameAgent`` 实例。

    这是项目的 Composition Root：集中创建模型、读取上下文预算、创建 ContextManager，
    再注入全局 Tool/Skill Registry、数据库和附件加载器。请求处理函数只调用组装好的
    Agent，不需要了解这些对象如何构造。
    """
    model = create_model()
    budget = ContextBudget.from_env()
    context_manager = ContextManager(model, budget)
    return GameAgent(
        model=model,
        tools=AGENT_TOOLS,
        context_manager=context_manager,
        skill_registry=skill_registry,
        session_store=session_store,
        attachment_loader=attachment_loader,
    )
