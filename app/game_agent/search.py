"""SearXNG 搜索服务的轻量客户端。

本模块只负责把 SearXNG JSON API 转换成项目内部稳定的搜索结果。模型工具、
超时降级和并发调度仍由 tools.py 与 ToolRegistry 负责。
"""

from __future__ import annotations

import re
from urllib.parse import urldefrag, urlparse

import httpx


def plain_text(value: str) -> str:
    """压缩搜索摘要中的 HTML 标签和多余空白。"""

    return re.sub(r"\s+", " ", re.sub(r"<[^>]+>", " ", value)).strip()


class SearchResult:
    """项目内部使用的统一搜索结果。"""

    def __init__(
        self,
        title: str,
        url: str,
        snippet: str,
        source_domain: str,
        engines: list[str] | None = None,
        published_at: str | None = None,
    ) -> None:
        self.title = title
        self.url = url
        self.snippet = snippet
        self.source_domain = source_domain
        self.engines = engines or []
        self.published_at = published_at


class SearxngSearch:
    """通过 HTTP 调用自托管 SearXNG，并规范化它返回的 JSON。"""

    def __init__(
        self,
        base_url: str,
        timeout_seconds: float = 6.0,
        language: str = "all",
        transport=None,
    ) -> None:
        if not base_url.strip():
            raise ValueError("SearXNG地址不能为空")
        if timeout_seconds <= 0:
            raise ValueError("SearXNG超时时间必须大于0")

        self.base_url = base_url.rstrip("/")
        self.timeout_seconds = timeout_seconds
        self.language = language or "all"
        # transport 主要用于测试，也允许以后接入自定义代理或网络传输层。
        self.transport = transport

    async def search(self, query: str, limit: int = 5) -> list[SearchResult]:
        """搜索单个Query，最多返回10条去重后的结果。"""

        normalized_query = query.strip()
        if not normalized_query:
            raise ValueError("搜索Query不能为空")

        result_limit = max(1, min(limit, 10))
        async with httpx.AsyncClient(
            timeout=self.timeout_seconds,
            transport=self.transport,
        ) as client:
            response = await client.get(
                f"{self.base_url}/search",
                params={
                    "q": normalized_query,
                    "format": "json",
                    "language": self.language,
                    "safesearch": 0,
                },
                headers={"Accept": "application/json"},
            )
            response.raise_for_status()

        payload = response.json()
        raw_results = payload.get("results")
        if not isinstance(raw_results, list):
            raise ValueError("SearXNG返回数据缺少results列表")

        results: list[SearchResult] = []
        seen_urls: set[str] = set()

        for item in raw_results:
            if not isinstance(item, dict):
                continue

            title = plain_text(str(item.get("title") or ""))
            url = urldefrag(str(item.get("url") or "").strip()).url
            if not title or not url.startswith(("http://", "https://")):
                continue
            if url in seen_urls:
                continue
            seen_urls.add(url)

            raw_engines = item.get("engines")
            if isinstance(raw_engines, list):
                engines = [str(engine) for engine in raw_engines if engine]
            elif item.get("engine"):
                engines = [str(item["engine"])]
            else:
                engines = []

            published_at = item.get("publishedDate") or item.get("published_date")
            results.append(SearchResult(
                title=title,
                url=url,
                snippet=plain_text(str(item.get("content") or "")),
                source_domain=urlparse(url).netloc.lower(),
                engines=engines,
                published_at=str(published_at) if published_at else None,
            ))

            if len(results) >= result_limit:
                break

        return results
