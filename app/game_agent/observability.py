"""Langfuse 可观测性接入（替代自建 agent_run_events 日志）。

先理解 Python 装饰器：

    @observe(name="agent_turn")
    async def run_turn(...):
        ...

在 Python 解释器看来，大致等价于：

    async def run_turn(...):
        ...

    run_turn = observe(name="agent_turn")(run_turn)

也就是说，``observe(name="agent_turn")`` 先创建一个“装饰器函数”，这个装饰器再
接收原来的 ``run_turn``，返回一个包装后的新函数。调用新函数时，包装器会在原函数
执行前开始计时/创建 Observation，执行后记录输出和耗时，异常时记录错误，然后仍然
把原函数的返回值或异常交还给调用方。业务函数本身不需要手写这些重复逻辑。

当被装饰函数发生嵌套调用时，Langfuse 通过异步上下文自动建立父子关系。例如：

    agent_turn
    ├── model_call
    │   └── ChatOpenAI generation
    └── tool_execution

其中前两层来自 ``@observe``，最内层的模型 Generation 来自 LangChain
``CallbackHandler``。

使用方式：
- 用 @observe() 装饰 Agent 方法，自动生成 Trace / Span；
- 把 langfuse_handler 作为 callback 传给 LangChain 模型调用，自动采集
  LLM 的输入、输出、Token 与延迟。

项目在 requirements.txt 中固定依赖当前 Langfuse SDK，因此这里只使用当前公开导入
路径，不维护旧 SDK 的兼容分支。没有配置密钥时不创建 LangChain CallbackHandler。
"""

from __future__ import annotations

import os

from dotenv import load_dotenv
from langfuse import observe
from langfuse.langchain import CallbackHandler

# 必须在读取环境变量之前加载 .env：main.py 的 load_dotenv() 在 import 之后才执行，
# 而本模块在 import 时就要判断是否已配置 Langfuse。
load_dotenv()

_PUBLIC_KEY = os.getenv("LANGFUSE_PUBLIC_KEY", "").strip().strip('"')
_SECRET_KEY = os.getenv("LANGFUSE_SECRET_KEY", "").strip().strip('"')
_CONFIGURED = bool(_PUBLIC_KEY and _SECRET_KEY)

# Handler 创建一次，模型调用时通过 LangChain config.callbacks 使用。没有配置密钥时
# 不挂载 CallbackHandler；@observe 仍可保留在业务代码中，由 Langfuse SDK 自行判断
# 是否存在可用的导出客户端。
langfuse_handler = CallbackHandler() if _CONFIGURED else None
