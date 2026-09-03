"""对本地SearXNG容器执行真实搜索；默认跳过，避免普通测试依赖网络。"""

import os

import pytest

from app.game_agent.search import SearxngSearch


pytestmark = pytest.mark.skipif(
    os.getenv("RUN_SEARXNG_INTEGRATION") != "1",
    reason="设置RUN_SEARXNG_INTEGRATION=1后才运行真实SearXNG测试",
)


@pytest.mark.asyncio
async def test_local_searxng_returns_web_results():
    backend = SearxngSearch(
        base_url=os.getenv("SEARXNG_BASE_URL", "http://127.0.0.1:8080"),
        timeout_seconds=10,
    )

    results = await backend.search("Python asyncio official documentation", limit=3)

    assert results
    assert all(result.title for result in results)
    assert all(result.url.startswith(("http://", "https://")) for result in results)
