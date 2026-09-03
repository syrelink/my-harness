# Web Search设计与验证

## 当前结构

模型始终只看到一个 `web_search` 工具。SearXNG 是候选网页发现服务，不直接进入
Agent Loop，也不是 MCP Server。

```text
Agent
  ↓ Tool Call
web_search(queries)
  ↓ 并发调用
SearxngSearch
  ↓ HTTP JSON
SearXNG
  ↓ 多搜索来源
标题、摘要、URL和来源
  ↓ ToolMessage
Agent整理并回答
```

这次替换没有改变 `web_search` 的模型参数、ToolRegistry、Agent Loop、SSE 或前端。
以后更换搜索后端时，也只需要替换 `search.py`。

## 启动

启动本地SearXNG：

```bash
docker compose up -d searxng
```

确认JSON接口：

```bash
curl "http://127.0.0.1:8080/search?q=Python+asyncio&format=json"
```

本机直接运行GameRover时使用默认地址：

```text
SEARXNG_BASE_URL=http://127.0.0.1:8080
```

如果GameRover以后也放进同一个Docker Compose，则改成：

```text
SEARXNG_BASE_URL=http://searxng:8080
```

可选配置：

```text
SEARXNG_TIMEOUT_SECONDS=6
SEARXNG_LANGUAGE=all
GAME_SEARCH_TIMEOUT_SECONDS=35
```

`SEARXNG_TIMEOUT_SECONDS` 限制一次SearXNG HTTP请求；`GAME_SEARCH_TIMEOUT_SECONDS`
是ToolRegistry对完整工具调用的外层限制。

## 三层测试

### 1. Mock测试

不访问网络，验证我们自己的代码：

```bash
python -m pytest -p no:cacheprovider tests/mock/test_websearch.py -q
```

覆盖内容：

- 请求使用 `/search`、`format=json` 和正确Query；
- HTML摘要被清理，URL片段被移除，重复URL被过滤；
- `engines` 和发布时间被转换成内部字段；
- 多个Query确实并发，但返回顺序保持不变；
- 一个Query失败时，其他结果不会丢失。

### 2. 集成测试

启动SearXNG后执行真实搜索：

```bash
RUN_SEARXNG_INTEGRATION=1 python -m pytest \
  -p no:cacheprovider tests/integration/test_searxng.py -q
```

默认测试会跳过这一项，避免普通单元测试依赖网络和Docker。

### 3. 搜索质量评测

启动SearXNG后运行固定查询：

```bash
python tests/eval/run_websearch_eval.py
```

这层不判断每天都会变化的固定答案，而是检查：

- 是否在超时内返回；
- 有效结果数；
- 来源域名数量；
- 标题、URL和摘要是否完整；
- 游戏、中文、英文和技术查询是否都能覆盖。

第一版建议验收目标：单Query中位耗时不超过2.5秒，P95不超过5秒，每个Query
至少3条有效结果、至少2个不同域名。实际阈值应根据本机网络跑出的基线调整。

## 当前边界

SearXNG返回的是搜索摘要，不是完整新闻正文。只有评测证明摘要不足时，才增加
正文读取能力；第一版不引入浏览器抓取、Embedding重排、向量数据库或研究子Agent。
