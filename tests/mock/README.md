# Mock 测试

本目录按被测模块组织使用假依赖的测试，不访问真实大模型、搜索服务、PostgreSQL 或 Langfuse。

## 文件划分

- `test_toolruntime.py`：验证工具注册、参数错误、超时、异常、结果裁剪和串并行调度。
- `test_toolloop.py`：验证模型产生 Tool Call、Registry 执行、ToolMessage 回填和模型最终回答的完整循环。
- `test_runmanager.py`：验证后台 Run、SSE 断开、单会话互斥、失败状态、主动停止和部分回答保留。
- `test_websearch.py`：验证 SearXNG 请求参数、JSON 规范化、URL 去重、批量并发和部分失败降级。

## 什么时候使用 Mock

适合使用 Mock 的情况：

1. 被测逻辑依赖大模型、网络、数据库等外部系统，但本次只想验证自己的控制流程。
2. 需要稳定制造参数错误、超时、执行异常、连接断开等真实环境不容易重复出现的情况。
3. 需要精确控制执行顺序，验证工具并发、串行屏障、取消和事件顺序。
4. 希望测试快速、免费、可重复，并能在没有外部服务的环境中运行。

不应该只使用 Mock 的情况：

1. 验证真实模型是否能正确选择工具，应使用 Agent 评测集。
2. 验证真实搜索结果、网页解析质量和外部 API 兼容性，应使用集成测试。
3. 验证 PostgreSQL 表结构、事务和并发写入，应连接测试数据库。
4. 验证前端 SSE 展示和页面刷新交互，应使用浏览器端到端测试。

一句话区分：Mock 验证“我们的代码逻辑是否正确”，集成测试验证“外部组件接起来是否正常”，Agent 评测验证“最终回答是否足够好”。

## 运行方式

运行全部 Mock 测试：

```bash
python -m pytest tests/mock -q
```

只运行某个模块：

```bash
python -m pytest tests/mock/test_toolruntime.py -vv
python -m pytest tests/mock/test_toolloop.py -vv
python -m pytest tests/mock/test_runmanager.py -vv
```
