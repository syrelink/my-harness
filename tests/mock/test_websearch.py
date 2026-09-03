"""验证SearXNG适配、结果规范化和批量搜索降级。"""

import asyncio
import json

import httpx
import pytest

from app.game_agent import tools as game_tools
from app.game_agent.search import SearchResult, SearxngSearch


@pytest.mark.asyncio
async def test_searxng_query_and_results_are_normalized():
    received_request: httpx.Request | None = None

    async def handler(request: httpx.Request) -> httpx.Response:
        nonlocal received_request
        received_request = request
        return httpx.Response(200, json={
            "results": [
                {
                    "title": "<b>第一条</b> 新闻",
                    "url": "https://news.example.com/one#top",
                    "content": " 第一条   摘要 ",
                    "engines": ["bing", "duckduckgo"],
                    "publishedDate": "2026-09-01",
                },
                {
                    "title": "重复新闻",
                    "url": "https://news.example.com/one#comments",
                    "content": "重复摘要",
                },
                {
                    "title": "第二条新闻",
                    "url": "https://news.example.com/two",
                    "content": "第二条摘要",
                    "engine": "brave",
                },
                {"title": "无效结果", "url": "javascript:void(0)"},
            ]
        })

    backend = SearxngSearch(
        base_url="http://searxng:8080/",
        language="zh-CN",
        transport=httpx.MockTransport(handler),
    )
    results = await backend.search("  游戏新闻  ", limit=5)

    assert received_request is not None
    assert received_request.url.path == "/search"
    assert received_request.url.params["q"] == "游戏新闻"
    assert received_request.url.params["format"] == "json"
    assert received_request.url.params["language"] == "zh-CN"
    assert [result.url for result in results] == [
        "https://news.example.com/one",
        "https://news.example.com/two",
    ]
    assert results[0].title == "第一条 新闻"
    assert results[0].snippet == "第一条 摘要"
    assert results[0].engines == ["bing", "duckduckgo"]
    assert results[0].published_at == "2026-09-01"
    assert results[1].engines == ["brave"]


@pytest.mark.asyncio
async def test_invalid_searxng_payload_is_rejected():
    async def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"unexpected": []})

    backend = SearxngSearch(
        base_url="http://searxng:8080",
        transport=httpx.MockTransport(handler),
    )

    with pytest.raises(ValueError, match="缺少results列表"):
        await backend.search("测试")


@pytest.mark.asyncio
async def test_web_search_runs_queries_concurrently_and_preserves_order(monkeypatch):
    class FakeBackend:
        def __init__(self) -> None:
            self.active = 0
            self.maximum_active = 0

        async def search(self, query: str, limit: int = 5) -> list[SearchResult]:
            self.active += 1
            self.maximum_active = max(self.maximum_active, self.active)
            await asyncio.sleep(0.01 if query == "查询一" else 0)
            self.active -= 1
            return [SearchResult(
                title=f"{query}结果",
                url=f"https://example.com/{query}",
                snippet="摘要",
                source_domain="example.com",
            )]

    backend = FakeBackend()
    monkeypatch.setattr(game_tools, "search_backend", backend)

    raw_result = await game_tools.web_search.ainvoke({
        "queries": ["查询一", "查询二"],
    })
    payload = json.loads(raw_result)

    assert backend.maximum_active == 2
    assert [item["query"] for item in payload["searches"]] == ["查询一", "查询二"]

# 这是异步测试，请为它创建并运行事件循环。
@pytest.mark.asyncio
async def test_one_failed_query_does_not_discard_other_results(monkeypatch):
    class PartlyBrokenBackend:
        async def search(self, query: str, limit: int = 5) -> list[SearchResult]:
            if query == "失败查询":
                raise httpx.ConnectError("SearXNG无法连接")
            return [SearchResult(
                title="正常结果",
                url="https://example.com/ok",
                snippet="摘要",
                source_domain="example.com",
            )]

    monkeypatch.setattr(game_tools, "search_backend", PartlyBrokenBackend())

    raw_result = await game_tools.web_search.ainvoke({
        "queries": ["失败查询", "正常查询"],
    })
    payload = json.loads(raw_result)

    assert payload["searches"][0]["error_code"] == "search_failed"
    assert payload["searches"][1]["results"][0]["title"] == "正常结果"
