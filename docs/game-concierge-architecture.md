# GameRover：无框架单 Agent Harness

GameRover 是一个面向中文玩家的游戏资讯与玩法助手。它不使用 LangGraph，而是用一个显式的
`while` 循环实现单 Agent Tool Loop；会话状态由 Postgres 持久化，可观测性由 Langfuse 提供。

## 运行流程

```text
用户输入
  → 压缩测压（ContextCompaction，多数时候 no-op）
  → Agent（调用模型）
      ├─ 无 tool_calls → END
      └─ 有 tool_calls → 手动执行工具 → 回到压缩测压 → Agent
```

整个循环在 `agent.py` 的 `GameAgent.run_turn()` 里，是一个普通 `while True`：

```python
while True:
    compact(state)              # 超阈值才压缩
    response = call_model(...)  # 流式调用模型
    if not response.tool_calls:
        break
    execute_tools(...)          # 手动执行工具
```

## 核心文件

- `agent.py`：Agent Loop、模型调用、工具执行、Token 累计。
- `events.py`：统一的 AgentEvent 与本轮有序事件发布器。
- `memory.py`：上下文边界、滚动摘要和模型上下文组装。
- `models.py`：ContextSummary、Context Metrics、Tool Trace、请求/响应模型。
- `tools.py`：`read_skill_file` / `web_search` 两个工具。
- `skills/`：Agent Skills 元数据、工作流、references 与安全加载器。
- `search.py`：最小化 DuckDuckGo 网页搜索。
- `multimodal.py`：图片引用协议与模型调用前的临时 Hydration。
- `observability.py`：Langfuse `@observe` 与 LangChain CallbackHandler 接入。
- `session_store.py`：会话、Transcript、附件元数据与 Agent State 的持久化。
- `main.py`：FastAPI 接口与 SSE 流式输出。

## 持久化（状态与日志分离）

会话状态不再用 LangGraph Checkpointer，而是普通 dict + Postgres 的 `agent_state` 表
（`load_state` / `save_state`）。只有跨轮字段进入持久状态：

```text
active_messages     # 模型活动上下文中的近期原文，不是完整 Transcript
context_summary     # 早期消息的结构化摘要
turn_count          # 轮次计数
compaction_count    # 累计压缩次数
summary_version     # 摘要版本号
```

本轮临时字段（当前图片、本轮 Token、工具轨迹、本轮指标）只作为局部变量，每轮重新声明、
天然归零，绝不进入持久状态。

`session_store.py` 在 Postgres 中维护：

```text
chat_sessions       # 会话列表
chat_transcript     # 不可变聊天记录（前端展示用）
agent_state         # Agent 执行状态（JSONB）
```

## 上下文管理

每次模型调用前测压，超过预算阈值（默认有效窗口 80%）才触发压缩：

```text
New ContextSummary = Compress(Old ContextSummary + Newly Expired Complete Turns)
```

`compact()` 在模型调用前直接估算实际 `model_context`。达到 80% 后，从后向前选择
完整 Turn，近期原文总量严格不超过窗口的 16%；更早消息通过“旧摘要 + 本次过期消息”
生成新 `ContextSummary`，并滚动替换旧摘要。摘要模型超时、调用失败或结构校验失败都会
直接抛错，原状态不会被修改。

模型输入由 `ContextManager.build_model_context()` 组装：

```text
System Prompt（含 Skill 目录）
+ ContextSummary
+ Recent Messages
```

## 多模态输入

图片上传后立即写入 MinIO，对象键由 `session_id` 哈希和后端生成的 UUID 确定性组成；
Postgres Transcript 和 Agent State 只保存 `AttachmentRef`，不再维护附件映射表。模型调用前
直接根据引用从 MinIO 取回原图、临时组装成多模态块；这个临时副本不写回状态。

## Skill 与渐进式披露

启动时 `SkillRegistry.refresh()` 扫描 `skills/*/SKILL.md`，只提取 `name + description`
注入 System Prompt 作为目录。模型判断任务匹配某个 Skill 时调用
`read_skill_file(name, "SKILL.md")` 加载正文；只有正文明确要求且任务需要时，才用同一工具
加载 references。

```text
目录（name + description，常驻）→ SKILL.md（激活后加载）→ references/*.md（按需加载）
```

Skill 是文档而非图节点，新增 Skill 只增加目录、不改 Agent 循环。当前包含：

- `gameplay-guide`：任务路线、Boss、地图、解谜与下一步指导。
- `game-build-advisor`：配队与养成建议。
- `game-news`：最新公告、版本动态、新闻与传闻核验。

## 可观测性（Langfuse）

`observability.py` 提供 `@observe()` 装饰器与 `langfuse_handler`，生成层级化的 Trace：

```text
agent_turn → model_call → tool_execution
```

`langfuse_handler` 作为 callback 传给模型调用，自动采集 LLM 的输入、输出、Token 与延迟。
项目直接依赖当前 Langfuse SDK，不保留旧版本导入路径；没有配置 `LANGFUSE_*` 密钥时
不创建 CallbackHandler。

## 前端

`POST /ai/chat/stream` 通过 SSE 流式返回四类事件：

```text
token   # 打字机流式输出
agent_event # Turn、上下文、模型与工具生命周期事件
final   # 最终结果（tool_trace / context_metrics / token_usage / context_summary）
error   # 错误
```

## 启动

```bash
# 1. 启动依赖（Postgres；MinIO 需单独运行）
docker compose up -d postgres

# 2. 安装依赖
pip install -r requirements.txt

# 3. 启动服务
uvicorn app.main:app --reload --port 8000
```

打开 `http://localhost:8000/`。

调试接口：

- `GET /ai/sessions`：列出历史会话。
- `GET /ai/sessions/{session_id}/messages`：读取完整聊天记录。
- `GET /ai/sessions/{session_id}/state`：查看 Agent 执行状态。
- `GET /ai/health`：健康检查。
