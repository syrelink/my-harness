"""提供给主 Agent 的 Tool Calling 接口。

本文件只定义模型可见的工具名称、参数和用途。Skill 以文档形式按需加载；
网页搜索通过一个薄适配层调用自托管 SearXNG。
"""

from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path

from langchain.tools import tool

from app.game_agent.search import SearxngSearch
from app.game_agent.skills import SkillRegistry
from app.runtime.toolruntime import ToolRegistry


skill_registry = SkillRegistry(
    Path(os.getenv("GAME_SKILLS_DIR", Path(__file__).resolve().parent / "skills"))
)

search_backend = SearxngSearch(
    base_url=os.getenv("SEARXNG_BASE_URL", "http://127.0.0.1:8080"),
    timeout_seconds=float(os.getenv("SEARXNG_TIMEOUT_SECONDS", "6")),
    language=os.getenv("SEARXNG_LANGUAGE", "all"),
)


@tool
def read_skill(name: str, paths: list[str]) -> str:
    """批量读取一个 Skill 的 SKILL.md 或必要 references；paths 应包含 1 至 5 个路径。"""
    normalized_paths = list(dict.fromkeys(path.strip() for path in paths if path.strip()))
    if not normalized_paths or len(normalized_paths) > 5:
        return json.dumps({
            "error_code": "invalid_args",
            "message": "paths 必须包含 1 至 5 个不重复路径",
            "tool": "read_skill",
        }, ensure_ascii=False)

    documents = []
    errors = []
    for path in normalized_paths:
        try:
            document = skill_registry.load(
                name.strip(),
                None if path == "SKILL.md" else path,
            )
            documents.append({
                "path": path,
                "content": document.content,
            })
        except Exception as exc:
            errors.append({
                "path": path,
                "message": str(exc),
                "error_code": "skill_read_failed",
            })

    return json.dumps({
        "status": "loaded" if documents else "error",
        "skill": name.strip(),
        "documents": documents,
        "errors": errors,
    }, ensure_ascii=False)


@tool
async def web_search(queries: list[str]) -> str:
    """通过SearXNG批量搜索1至4个问题，返回标题、摘要、链接和来源。"""
    search_queries = list(dict.fromkeys(query.strip() for query in queries if query.strip()))
    if not search_queries or len(search_queries) > 4:
        return json.dumps({
            "error_code": "invalid_args",
            "message": "queries 必须包含 1 至 4 个不重复查询",
            "tool": "web_search",
        }, ensure_ascii=False)

    async def search_one(query: str) -> dict:
        try:
            results = await search_backend.search(query, limit=5)
        except Exception as exc:
            return {
                "query": query,
                "message": str(exc),
                "error_code": "search_failed",
            }
        return {
            "query": query,
            "results": [
                {
                    "title": r.title,
                    "url": r.url,
                    "snippet": r.snippet,
                    "domain": r.source_domain,
                    **({"engines": r.engines} if r.engines else {}),
                    **({"published_at": r.published_at} if r.published_at else {}),
                }
                for r in results
            ],
            **({"note": "未找到结果"} if not results else {}),
        }

    searches = await asyncio.gather(*(search_one(query) for query in search_queries))
    return json.dumps({"searches": searches}, ensure_ascii=False)


# 工具能力和运行策略统一注册；只有 Tool Schema 会通过 bind_tools 暴露给模型。
tool_registry = ToolRegistry()
tool_registry.register(
    read_skill,
    timeout_seconds=float(os.getenv("GAME_SKILL_TIMEOUT_SECONDS", "8")),
    execution_mode="parallel",
    idempotent=True,
    max_result_chars=30_000,
)
tool_registry.register(
    web_search,
    timeout_seconds=float(os.getenv("GAME_SEARCH_TIMEOUT_SECONDS", "35")),
    execution_mode="parallel",
    idempotent=True,
    max_result_chars=20_000,
)
