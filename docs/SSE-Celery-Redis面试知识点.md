# SSE、Celery 与 Redis：Agent 长任务知识点

> 用于理解 AI 内容生产平台中的“长任务异步化”和“过程实时可见”。

## 1. 一页总览

```text
前端
├── POST：提交用户消息
└── GET SSE：持续接收进度

Django
├── 保存用户消息
├── task.delay() 投递后台任务
└── SSE 服务持续返回事件

Celery Worker
├── 从 Broker 领取任务
├── 执行 Agent
└── 主动把文本、工具状态和进度写入 Redis

Redis
├── Broker：传递待执行任务
├── Pub/Sub：通知 SSE 有新消息
└── SET/ZSET：保存正文和有序消息历史
```

一句话记忆：

> POST 负责提交，Celery 负责执行，Redis 负责传递和保存消息，SSE 负责实时推送。

---

## 2. SSE 前后端连接

### 2.1 SSE 是什么

SSE（Server-Sent Events）用于服务器向浏览器单向持续推送消息。

它本质上仍是 HTTP：浏览器只发送一次 GET，服务器只返回一个 HTTP Response，但响应暂时不结束，响应体中可以包含多条 SSE 事件。

```text
一个 GET 请求
    ↓
一个长期 HTTP Response
├── 事件1：正在分析
├── 事件2：工具开始
├── 事件3：工具完成
└── 事件4：最终回答
```

事件不是新的 GET/POST 请求，而是同一个响应体中的一条消息。

### 2.2 前端代码

```javascript
const source = new EventSource(
  "/messages?conversation_id=123"
);

source.onmessage = (event) => {
  const data = JSON.parse(event.data);
  console.log(data);
};
```

- `EventSource` 是浏览器原生 SSE 客户端，创建时会发起 GET。
- `source` 表示这条长期连接。
- 每收到一条默认事件，浏览器就调用一次 `onmessage`。
- `event` 是浏览器创建的 `MessageEvent`。
- `event.data` 是服务端 `data:` 后面的字符串。

原生 `EventSource` 只能发起 GET，不适合携带 JSON 请求体。因此 Agent 平台通常拆成：

```text
POST /send-message：提交任务
GET  /messages：接收过程和结果
```

### 2.3 服务端代码

```python
from django.http import StreamingHttpResponse


def event_stream():
    yield 'data: {"content":"开始处理"}\n\n'
    yield 'data: {"content":"处理完成"}\n\n'


def messages(request):
    return StreamingHttpResponse(
        event_stream(),
        content_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )
```

`StreamingHttpResponse` 不是 SSE 专用包，它只是 Django 的流式响应类。下面四项组合起来才是 SSE：

```text
流式 HTTP Response
+ 生成器 yield
+ Content-Type: text/event-stream
+ 合法的 SSE 事件格式
```

### 2.4 `yield` 是什么

`return` 一次性返回并结束函数：

```python
def answer():
    return "完整回答"
```

`yield` 返回一部分结果、暂停函数，下次从暂停位置继续：

```python
def stream():
    yield "第一段"
    yield "第二段"
```

流式响应会依次取得每个 `yield` 的内容并发给浏览器。

### 2.5 SSE 事件格式

最简单格式：

```text
data: {"content":"正在分析"}

```

对应：

```python
yield 'data: {"content":"正在分析"}\n\n'
```

- `data:`：事件数据。
- 第一个 `\n`：结束当前字段行。
- 第二个 `\n`：形成空行，表示整条事件结束。

命名事件：

```text
event: token
id: message-001
data: {"content":"正在分析"}

```

前端监听：

```javascript
source.addEventListener("token", (event) => {
  const data = JSON.parse(event.data);
  console.log(data.content);
});

source.addEventListener("final", (event) => {
  const data = JSON.parse(event.data);
  console.log(data.answer);
  source.close();
});
```

不写 `event:` 时是默认 `message` 事件，使用 `source.onmessage`。

### 2.6 生产注意点

- `Cache-Control: no-cache`：避免缓存事件流。
- `X-Accel-Buffering: no`：避免 Nginx 缓冲后一次性发送。
- 定期发送 `data: {}\n\n` 心跳，避免代理关闭空闲连接。
- 浏览器离开页面后关闭连接；服务端在 `finally` 中取消 Redis 订阅。
- SSE 适合服务器到浏览器的单向推送；双向高频通信更适合 WebSocket。

---

## 3. Celery

### 3.1 Celery 是什么

Celery 是 Python 分布式后台任务队列，负责后台执行、任务排队、并发控制、队列路由、超时、重试和定时调度。

Celery 不负责模型调用、Agent 规划、上下文压缩或 SSE，只负责“任务在哪里、什么时候、由谁执行”。

### 3.2 核心角色

```text
Producer → Broker → Celery Worker
```

| 角色 | 作用 |
|---|---|
| Celery App | 注册任务、保存配置 |
| Producer | 调用 `.delay()` 的 Django/FastAPI |
| Broker | 保存待执行任务，通常是 Redis/RabbitMQ |
| Worker | 从 Broker 领取并执行任务的后台进程 |
| Result Backend | 可选，保存最终状态和返回值 |
| Celery Beat | 可选，定时投递任务 |

Celery App 和 Worker 由 Celery 包提供；Broker 是外部服务。

### 3.3 配置、注册和提交

```python
from celery import Celery, shared_task


app = Celery(
    "aicenterwebui",
    broker="redis://localhost:6379/0",
)

@shared_task
def run_agent(session_id: str):
    # Worker 中真正执行
    ...
```

Django 提交：

```python
task = run_agent.delay("session-001")
```

`.delay()` 不会在 Django 进程中调用函数，而是向 Broker 写入：

```json
{
  "task": "app.tasks.run_agent",
  "id": "task-123",
  "args": ["session-001"],
  "kwargs": {}
}
```

Broker 传递的是任务名称和参数，不是函数源码。因此推荐只传 ID、字符串、数字、列表和字典；Worker 再根据 ID 查询完整数据。

### 3.4 启动 Worker

```bash
celery -A aicenterwebui worker --loglevel=info -P threads -c 32
```

- `-A`：加载项目的 Celery App。
- `worker`：启动后台执行器。
- `-P threads`：使用线程池。
- `-c 32`：最多 32 个并发执行槽位。

Worker 流程：

```text
加载任务注册表
→ 连接 Broker
→ 领取任务名称和参数
→ 找到对应 Python 函数
→ 执行函数
→ 记录成功或失败
```

Worker 不一定严格顺序执行；超出并发槽位的任务才继续排队。

### 3.5 Celery 与 `async/await`

| 对比 | `async/await` | Celery |
|---|---|---|
| 执行位置 | 当前进程事件循环 | 独立 Worker 进程/服务器 |
| HTTP 返回后继续执行 | 通常不适合 | 可以 |
| Broker | 不需要 | 需要 |
| 排队、路由、重试 | 不自带 | 支持 |
| 适用场景 | 短时 I/O 并发 | 长任务和后台任务 |

两者可以组合：Celery 决定在哪个 Worker 执行，Worker 内部再用异步并发执行多个检索。

推荐模式：

```text
普通短问答 → API 进程直接流式执行
深度调研/图片视频生成 → Celery + Redis/SSE
```

---

## 4. Redis 的三个职责

### 4.1 Celery Broker

```text
Django调用 task.delay()
→ Redis Broker保存任务
→ Worker领取任务
```

传递的是：“请执行 `run_agent(session-001)`”。

### 4.2 Pub/Sub 实时通知

Worker 发布：

```python
redis.publish(
    f"ROOM_SSE_MESSAGES:{conversation_id}",
    notification,
)
```

SSE 订阅：

```python
pubsub = redis.pubsub()
pubsub.subscribe(
    f"ROOM_SSE_MESSAGES:{conversation_id}"
)
```

Pub/Sub 传递的是：“这个会话有新进度，请读取。”

`pubsub.subscribe()` 是 Redis 客户端的发布/订阅能力。Pub/Sub 不是可靠队列：订阅者离线时消息会丢失，因此适合做低延迟通知，不适合单独充当消息账本。

### 4.3 SET/ZSET 保存消息历史

```python
# 保存正文
redis.set(
    f"chat_msg:{conversation_id}:{msg_id}",
    message,
)

# 按时间建立会话消息索引
redis.zadd(
    f"chat:{conversation_id}",
    {message_key: timestamp},
)

# 通知 SSE
redis.publish(channel, notification)
```

记忆：

```text
SET     = 消息正文
ZSET    = 可回放的有序消息账本
Pub/Sub = 新消息铃声
```

即使 SSE 断开时错过了“铃声”，重新连接后仍可从 ZSET 回放消息。

---

## 5. Agent 长任务完整链路

```text
1. 前端 new EventSource()
   → GET SSE连接
   → SSE服务订阅Redis频道

2. 前端 fetch POST
   → Django保存用户消息
   → run_agent.delay(session_id)
   → 立即返回“任务已提交”

3. Celery
   → 任务进入Broker
   → Worker领取任务
   → 执行Agent

4. Agent产生进度
   → Redis SET保存正文
   → Redis ZSET保存有序索引
   → Redis PUBLISH通知SSE

5. SSE收到通知
   → 查询新增消息
   → yield "data: {...}\n\n"
   → 浏览器触发onmessage
```

最小核心代码：

```python
# 1. 注册后台任务
@shared_task
def run_agent(conversation_id):
    for event in agent.run(conversation_id):
        save_message(event)
        redis.publish(
            f"agent:{conversation_id}",
            "new_message",
        )


# 2. Django提交任务
run_agent.delay(conversation_id)


# 3. SSE等待Redis通知并推送
while True:
    notification = pubsub.get_message(timeout=1)
    if notification:
        for message in load_new_messages(conversation_id):
            yield (
                "data: "
                + json.dumps(message, ensure_ascii=False)
                + "\n\n"
            )
```

重要：Celery 不会自动发布 Agent 进度，必须由 Agent 的事件输出器主动保存消息并 `publish()`。

---

## 6. 常见误区

1. **`StreamingHttpResponse` 就是 SSE**：错误。它只是流式响应类。
2. **每条 SSE 事件都是新请求**：错误。一次 GET 的响应体包含多条事件。
3. **Celery 自动流式返回进度**：错误。Agent 必须主动发布事件。
4. **Broker 把结果返回 Django**：错误。Broker主要把任务从 Producer 交给 Worker。
5. **Redis Pub/Sub 是可靠队列**：错误。订阅者离线时可能丢消息。
6. **Worker 一定顺序执行**：错误。它可以配置多个并发槽位。
7. **Celery 会让模型推理更快**：错误。它主要减少 HTTP 阻塞并提供排队、隔离和扩容。

---

## 7. 面试回答模板

### 为什么使用 Celery？

> Agent 深度调研和图片、视频生成可能持续几十秒到数分钟。如果直接在 HTTP 请求中执行，会长期占用 API Worker并面临网关超时。因此我们通过 `.delay()` 将任务写入 Redis Broker，由独立 Celery Worker后台执行。普通短问答仍直接流式处理，避免 Celery调度增加首字延迟。

### Celery、Worker和Broker是什么关系？

> Celery 是任务队列框架，Django是 Producer。调用 `.delay()` 后，Celery把任务名称和参数写入 Broker；Worker加载同一个 Celery App并连接同一个 Broker，领取任务后在独立进程中执行对应 Python 函数。Redis或RabbitMQ是外部 Broker。

### SSE如何实现实时输出？

> 前端通过 `EventSource` 发起一次 GET，服务端返回 `text/event-stream` 流式响应，并持续 `yield` `data: ...\n\n`。Agent Worker把执行进度发布到 Redis，SSE订阅对应会话频道，收到通知后读取新增消息并推给浏览器，所以原 POST结束后仍能展示执行过程。

### 为什么同时使用 Pub/Sub 和 ZSET？

> Pub/Sub延迟低，适合通知SSE有新消息，但它不持久化。因此先用 SET保存正文、用 ZSET按时间保存消息索引，再通过 Pub/Sub实时通知。Pub/Sub是铃声，ZSET是可回放的消息账本。

### Celery和异步有什么区别？

> `async/await` 解决当前进程内部等待 I/O 时的并发问题；Celery把任务交给独立 Worker，HTTP结束后仍可继续，并提供排队、路由和重试能力。两者可以组合：Celery负责跨进程调度，Worker内部再用异步并发检索。

---

## 8. 最终速记

```text
@shared_task       注册Celery任务
task.delay()       将任务名称和参数写入Broker
Celery Worker      从Broker领取并执行任务
Redis Broker       传递待执行任务
Redis SET/ZSET     保存可回放消息
Redis Pub/Sub      实时通知SSE
EventSource        浏览器SSE客户端
StreamingResponse  服务端流式HTTP响应
yield              分段产生响应体
data: ...\n\n     最简单的SSE事件
```

> 最终闭环：前端用 POST 提交任务，Django通过 `.delay()` 将 Agent任务写入 Broker，Celery Worker后台执行；Agent把进度保存并发布到 Redis，SSE持续监听并通过 `yield` 推给前端。
