# `app` 代码目录说明

代码按职责分层，查找功能时先看目录，再看具体文件。依赖方向是：

```text
main.py → api/ → runtime/ → game_agent/
   └────────────→ storage/ ←──────┘
```

## 目录结构

```text
app/
├── main.py                         # 应用入口与依赖组装
├── api/
│   └── routes.py                   # HTTP 路由、SSE 编码、请求响应转换
├── runtime/
│   ├── runmanager.py               # 后台 Run、状态查询、刷新恢复
│   └── toolruntime.py              # ToolSpec、ToolRegistry 与统一执行入口
├── storage/
│   ├── sessionstore.py             # PostgreSQL 会话、消息与 Agent State
│   └── attachmentstore.py          # MinIO 图片存取与清理
├── game_agent/
│   ├── agent.py                    # Agent 主循环、模型调用、工具调度
│   ├── memory.py                   # Token 预算、滚动摘要、上下文组装
│   ├── models.py                   # API、附件和摘要的数据模型
│   ├── stream.py                   # Agent 内部流式事件协议
│   ├── tools.py                    # read_skill、web_search 工具注册
│   ├── search.py                   # DuckDuckGo 搜索实现
│   ├── multimodal.py               # 图片引用与模型调用前按需加载
│   ├── prompts.py                  # 系统提示词和压缩提示词
│   ├── observability.py            # Langfuse 链路观测
│   └── skills/                     # Skill 注册器、SKILL.md 与参考资料
└── web/
    ├── index.html                  # 聊天页面、SSE 消费与刷新恢复
    └── favicon.svg                 # 页面图标
```

## 从需求定位代码

- 修改接口或 SSE 事件格式：从 `api/routes.py` 开始。
- 修改“刷新后任务仍继续”：看 `runtime/runmanager.py`。
- 修改数据库表、历史消息或附件：看 `storage/`。
- 修改模型如何思考、调用工具或压缩上下文：看 `game_agent/`。
- 修改页面显示和交互：看 `web/index.html`。

`main.py` 不写业务逻辑，只创建共享依赖并把各层连接起来。这样移动存储、
运行时或前端实现时，不需要修改 Agent 主循环。

## 测试目录

项目级测试统一放在 `tests/`：

- `conftest.py`：设置所有测试共享的环境。
- `test_runmanager.py`：验证断开 SSE 后继续执行、主动停止、单会话互斥和失败状态。
- `test_toolruntime.py`：验证工具注册、参数校验、超时和并发策略。
