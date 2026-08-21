"""最小可用的网页搜索。

只保留「网页检索」这一能力：直接调用 DuckDuckGo HTML 端点，返回标题、摘要
与链接。原 query_rewriter / reranker / sufficiency 等 Agentic Search 管线已移除。
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from html.parser import HTMLParser
from urllib.parse import parse_qs, unquote, urlparse

import httpx


def plain_text(value: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"<[^>]+>", " ", value)).strip()


@dataclass
class SearchResult:
    title: str
    url: str
    snippet: str
    source_domain: str = ""


class _DuckDuckGoParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.results: list[dict] = []
        self._current: dict | None = None
        self._capture: str | None = None

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        attributes = dict(attrs)
        classes = set((attributes.get("class") or "").split())
        if tag == "a" and "result__a" in classes:
            self._current = {"title": "", "snippet": "", "url": self._clean_url(attributes.get("href", ""))}
            self._capture = "title"
        elif self._current is not None and "result__snippet" in classes:
            self._capture = "snippet"

    def handle_endtag(self, tag: str) -> None:
        if tag == "a" and self._capture == "title":
            self._capture = None
        elif self._current is not None and self._capture == "snippet" and tag in {"a", "div"}:
            self.results.append(self._current)
            self._current = None
            self._capture = None

    def handle_data(self, data: str) -> None:
        if self._current is not None and self._capture:
            self._current[self._capture] += data.strip() + " "

    @staticmethod
    def _clean_url(url: str) -> str:
        parsed = urlparse(url)
        redirected = parse_qs(parsed.query).get("uddg")
        return unquote(redirected[0]) if redirected else url


class DuckDuckGoSearch:
    endpoint = "https://html.duckduckgo.com/html/"

    def __init__(self, timeout_seconds: float = 6) -> None:
        self.timeout_seconds = timeout_seconds

    async def search(self, query: str, limit: int = 5) -> list[SearchResult]:
        params = {"q": query}
        async with httpx.AsyncClient(timeout=self.timeout_seconds, follow_redirects=True) as client:
            response = await client.get(
                self.endpoint, params=params, headers={"User-Agent": "Mozilla/5.0"}
            )
            response.raise_for_status()
        parser = _DuckDuckGoParser()
        parser.feed(response.text)
        results = []
        for item in parser.results[: max(1, min(limit, 10))]:
            url = item["url"]
            results.append(SearchResult(
                title=plain_text(item["title"]),
                snippet=plain_text(item["snippet"]),
                url=url,
                source_domain=urlparse(url).netloc.lower(),
            ))
        return results
