"""运行一组稳定查询，测量SearXNG速度、结果数和来源覆盖。"""

from __future__ import annotations

import asyncio
import json
import os
import statistics
import sys
import time
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from app.game_agent.search import SearxngSearch


async def main() -> int:
    cases = json.loads(
        (Path(__file__).with_name("websearch_cases.json")).read_text(encoding="utf-8")
    )
    backend = SearxngSearch(
        base_url=os.getenv("SEARXNG_BASE_URL", "http://127.0.0.1:8080"),
        timeout_seconds=float(os.getenv("SEARXNG_TIMEOUT_SECONDS", "6")),
        language=os.getenv("SEARXNG_LANGUAGE", "all"),
    )

    durations = []
    reports = []

    for case in cases:
        started_at = time.perf_counter()
        try:
            results = await backend.search(case["query"], limit=5)
            error = None
        except Exception as exc:
            results = []
            error = str(exc)
        elapsed_ms = round((time.perf_counter() - started_at) * 1000)
        durations.append(elapsed_ms)

        domain_count = len({result.source_domain for result in results})
        passed = (
            error is None
            and len(results) >= case["min_results"]
            and domain_count >= case["min_domains"]
        )
        reports.append({
            "id": case["id"],
            "elapsed_ms": elapsed_ms,
            "result_count": len(results),
            "domain_count": domain_count,
            "passed": passed,
            "error": error,
            "top_results": [
                {"title": result.title, "url": result.url}
                for result in results[:3]
            ],
        })

    sorted_durations = sorted(durations)
    p95_index = max(0, min(len(sorted_durations) - 1, int(len(sorted_durations) * 0.95)))
    summary = {
        "passed": sum(report["passed"] for report in reports),
        "total": len(reports),
        "median_ms": round(statistics.median(durations)),
        "p95_ms": sorted_durations[p95_index],
        "cases": reports,
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0 if summary["passed"] == summary["total"] else 1


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
