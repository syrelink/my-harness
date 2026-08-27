"""提供给主 Agent 的 Tool Calling 接口。

本文件只定义模型可见的工具名称、参数和用途。Skill 以文档形式按需加载；
网页搜索直接调用最小化的 DuckDuckGo 实现。
"""

from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path

from langchain.tools import tool

from app.game_agent.search import DuckDuckGoSearch
from app.game_agent.skills import SkillRegistry


skill_registry = SkillRegistry(
    Path(os.getenv("GAME_SKILLS_DIR", Path(__file__).resolve().parent / "skills"))
)

search_backend = DuckDuckGoSearch()


@tool
def read_skill(name: str, paths: list[str]) -> str:
    """批量读取一个 Skill 的 SKILL.md 或必要 references；paths 应包含 1 至 5 个路径。"""
    normalized_paths = list(dict.fromkeys(path.strip() for path in paths if path.strip()))
    if not normalized_paths or len(normalized_paths) > 5:
        return json.dumps({
            "error": "paths 必须包含 1 至 5 个不重复路径",
            "error_type": "ValueError",
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
                "error": str(exc),
                "error_type": exc.__class__.__name__,
            })

    return json.dumps({
        "status": "loaded" if documents else "error",
        "skill": name.strip(),
        "documents": documents,
        "errors": errors,
    }, ensure_ascii=False)


@tool
async def web_search(queries: list[str]) -> str:
    """批量搜索 1 至 4 个独立问题；查询会并行执行并按输入顺序返回标题、摘要与链接。"""
    search_queries = list(dict.fromkeys(query.strip() for query in queries if query.strip()))
    if not search_queries or len(search_queries) > 4:
        return json.dumps({
            "error": "queries 必须包含 1 至 4 个不重复查询",
            "error_type": "ValueError",
            "tool": "web_search",
        }, ensure_ascii=False)

    async def search_one(query: str) -> dict:
        try:
            results = await search_backend.search(query, limit=5)
        except Exception as exc:
            return {
                "query": query,
                "error": str(exc),
                "error_type": exc.__class__.__name__,
            }
        return {
            "query": query,
            "results": [
                {"title": r.title, "url": r.url, "snippet": r.snippet, "domain": r.source_domain}
                for r in results
            ],
            **({"note": "未找到结果"} if not results else {}),
        }

    searches = await asyncio.gather(*(search_one(query) for query in search_queries))
    return json.dumps({"searches": searches}, ensure_ascii=False)


# 并发策略只供 Harness 调度器读取，不会进入模型看到的 Tool Schema。
read_skill.metadata = {"execution_mode": "parallel"}
web_search.metadata = {"execution_mode": "parallel"}

# 只有列表中的函数会通过 model.bind_tools 暴露给主 Agent。
AGENT_TOOLS = [read_skill, web_search]
