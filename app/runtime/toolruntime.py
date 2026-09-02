"""统一工具描述、注册和执行策略。"""

from __future__ import annotations

import asyncio
import json
from typing import Literal
from uuid import uuid4

from langchain_core.messages import ToolMessage
from langchain_core.tools import BaseTool
from pydantic import ValidationError

from app.game_agent.observability import observe, update_current_span


ExecutionMode = Literal["serial", "parallel"]


class ToolSpec:
    """一个工具的实现，以及由 Harness 执行的运行策略。
    Spec是Specification的缩写，中文通常翻译为：规格、规范、配置说明。
    """

    def __init__(
        self,
        tool: BaseTool,
        timeout_seconds: float = 35.0,
        execution_mode: ExecutionMode = "serial",
        idempotent: bool = False,
        max_result_chars: int = 20_000,
    ) -> None:
        if timeout_seconds <= 0:
            raise ValueError("timeout_seconds 必须大于 0")
        if execution_mode not in {"serial", "parallel"}:
            raise ValueError("execution_mode 只能是 serial 或 parallel")
        if max_result_chars <= 0:
            raise ValueError("max_result_chars 必须大于 0")

        # 真正可执行的 LangChain 工具。
        self.tool = tool
        # 单次工具调用最多等待的秒数。
        self.timeout_seconds = timeout_seconds
        # serial 表示串行执行，parallel 表示允许与同批工具并发执行。
        self.execution_mode = execution_mode
        # 相同调用能否安全重试。
        self.idempotent = idempotent
        # 返回给模型的工具结果最大字符数。
        self.max_result_chars = max_result_chars


class ToolRegistry:
    """保存 ToolSpec，并通过同一入口执行所有工具。"""

    def __init__(self) -> None:
        self.tool_specs: dict[str, ToolSpec] = {}

    def register(
        self,
        tool: BaseTool,
        *,
        # *后面的参数必须通过“参数名”传递，不能只按照位置传递。
        timeout_seconds: float = 35.0,
        execution_mode: ExecutionMode = "serial",
        idempotent: bool = False,
        max_result_chars: int = 20_000,
    ) -> ToolSpec:
        """注册工具及策略；同名重复注册会立即报错。"""

        spec = ToolSpec(
            tool=tool,
            timeout_seconds=timeout_seconds,
            execution_mode=execution_mode,
            idempotent=idempotent,
            max_result_chars=max_result_chars,
        )
        tool_name = spec.tool.name
        if tool_name in self.tool_specs:
            raise ValueError(f"工具已注册：{tool_name}")
        self.tool_specs[tool_name] = spec
        return spec

    def get(self, name: str) -> ToolSpec | None:
        """根据名称查找工具配置"""
        return self.tool_specs.get(name)

    def specs(self) -> tuple[ToolSpec, ...]:
        """查看所有工具名称"""

        return tuple(self.tool_specs.values())

    def model_tools(self) -> list[BaseTool]:
        """返回只包含模型 Tool Schema 的列表，运行策略不会暴露给模型。"""

        return [spec.tool for spec in self.tool_specs.values()]

    def allows_parallel(self, call: dict) -> bool:
        """只有已注册且明确声明为 parallel 的工具才允许并发。"""

        tool_config = self.get(str(call.get("name", "")))
        return tool_config is not None and tool_config.execution_mode == "parallel"

    async def execute_many(
        self,
        calls: list[dict],
        *,
        step: int,
        max_parallel_tools: int,
    ) -> list[ToolMessage]:
        """按模型调用顺序调度工具，串行工具会成为并发组之间的屏障。"""

        semaphore = asyncio.Semaphore(max(1, max_parallel_tools))

        async def execute_limited(call: dict) -> ToolMessage:
            async with semaphore:
                return await self.execute(call, step=step)

        async def flush_parallel_group(group: list[dict]) -> list[ToolMessage]:
            if not group:
                return []
            # gather 并发等待，但结果顺序与 group 的原始顺序一致。
            return await asyncio.gather(*(execute_limited(call) for call in group))

        results: list[ToolMessage] = []
        parallel_group: list[dict] = []

        for call in calls:
            if self.allows_parallel(call):
                parallel_group.append(call)
                continue

            # 串行工具开始前，先等待前面的并发组全部结束。
            results.extend(await flush_parallel_group(parallel_group))
            parallel_group = []
            results.append(await self.execute(call, step=step))

        results.extend(await flush_parallel_group(parallel_group))
        return results

    @observe(name="tool_call", as_type="tool", capture_input=False, capture_output=False)
    async def execute(
        self,
        call: dict,
        *,
        step: int,
    ) -> ToolMessage:
        """统一完成查找、超时、异常、结果裁剪、观测和 ToolMessage 转换。"""
        name = str(call.get("name", "unknown"))
        call_id = str(call.get("id", "") or uuid4())
        raw_args = call.get("args", {})
        tool_config = self.get(name)
        # 不上传工具参数、会话标识和原始结果，避免观测数据泄露用户内容。
        update_current_span(
            name=f"tool:{name}",
            metadata={"step": step},
        )

        if tool_config is None:
            return self.error_message(
                call_id=call_id,
                name=name,
                message=f"未知工具：{name}",
                error_code="unknown_tool",
            )

        if not isinstance(raw_args, dict):
            return self.error_message(
                call_id=call_id,
                name=name,
                message="工具参数必须是 JSON 对象",
                error_code="invalid_args",
            )

        args = raw_args

        try:
            value = await asyncio.wait_for(
                tool_config.tool.ainvoke(args),
                timeout=tool_config.timeout_seconds,
            )
            content = value if isinstance(value, str) else json.dumps(
                value, ensure_ascii=False, default=str
            )
            if len(content) > tool_config.max_result_chars:
                omitted = len(content) - tool_config.max_result_chars
                content = content[:tool_config.max_result_chars] + f"\n[结果已截断，省略 {omitted} 个字符]"
            return ToolMessage(
                content=content,
                tool_call_id=call_id,
                name=name,
                status="success",
                id=str(uuid4()),
            )
        except ValidationError as exc:
            return self.error_message(
                call_id=call_id,
                name=name,
                message=str(exc),
                error_code="invalid_args",
            )
        except asyncio.TimeoutError:
            return self.error_message(
                call_id=call_id,
                name=name,
                message=f"{name} 执行超时",
                error_code="tool_timeout",
            )
        except Exception as exc:
            return self.error_message(
                call_id=call_id,
                name=name,
                message=str(exc),
                error_code="execution_failed",
            )

    def error_message(
        self,
        call_id: str,
        name: str,
        message: str,
        error_code: str,
    ) -> ToolMessage:
        """把工具错误转换成模型可以读取的 ToolMessage。"""
        content = json.dumps({
            "error_code": error_code,
            "message": message,
            "tool": name,
        }, ensure_ascii=False)
        update_current_span(
            output={"status": "error", "error_code": error_code},
            level="ERROR",
            status_message=error_code,
        )
        return ToolMessage(
            content=content,
            tool_call_id=call_id,
            name=name,
            status="error",
            id=str(uuid4()),
        )
