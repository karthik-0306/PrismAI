"""
backend/subagents/web_search_agent.py

Web Search Subagent — handles queries classified as 'web_search'.

Uses the Tavily Search API (built for AI agents) over plain httpx.

Workflow:
  1. Search Tavily with the router's self-contained query (so a follow-up like
     "and in Hyderabad?" searches for the full question, not three words).
  2. Build a synthesis prompt with numbered sources + recent conversation.
  3. The orchestrator streams the synthesis; `sources` are returned separately
     so the frontend can render them as citation chips.
"""

import logging
import os
from datetime import date
from typing import List, Tuple

import httpx

from backend import config
from backend.pipeline.memory import MemoryInjector

logger = logging.getLogger(__name__)

TAVILY_API_URL = "https://api.tavily.com/search"

WEB_SEARCH_SYSTEM_PROMPT = """\
You are PrismAI's web research assistant. Today's date is {today}.
You are given fresh web search results for the user's latest question.

- Answer the question directly first, then add supporting detail.
- Ground every factual claim in the results and cite inline with the source number in
  square brackets, e.g. "Gold rose 2% today [2]". Only cite numbers that exist.
- Give exact figures (prices, dates, scores) when the results contain them, and say
  when the information was published if the results show it.
- If results conflict, say so. If they are insufficient, say what is missing and give
  your best general knowledge clearly labelled as possibly outdated.
- Do not add a separate sources list — the app displays the sources itself.
- Use clean Markdown (short paragraphs, bullets, tables where useful).
"""

NO_RESULTS_NOTE = (
    "Web search is unavailable right now (no results were returned). Answer from your "
    "general knowledge, and clearly tell the user that live information could not be "
    "retrieved, so figures may be outdated."
)


async def fetch_tavily_results(query: str, max_results: int = 6) -> List[dict]:
    """Search Tavily; returns [] on any failure (missing key, HTTP error, timeout)."""
    api_key = os.getenv("TAVILY_API_KEY", "").strip()
    if not api_key or api_key.startswith("your_") or api_key == "dummy_tavily_key":
        logger.error("WebSearch: TAVILY_API_KEY not set.")
        return []

    async def _call(q: str) -> List[dict]:
        payload = {
            "query": q[:400],
            "search_depth": "basic",
            "max_results": max_results,
            "include_answer": False,
            "include_raw_content": False,
        }
        async with httpx.AsyncClient(timeout=15.0) as client:
            resp = await client.post(TAVILY_API_URL, json=payload,
                                     headers={"Authorization": f"Bearer {api_key}"})
            resp.raise_for_status()
            return resp.json().get("results", []) or []

    try:
        results = await _call(query)
        # Hyper-local queries ("gold price in Tenali") are often sparse — broaden once.
        if len(results) < 2 and " in " in query.lower():
            broad = query[: query.lower().rfind(" in ")].strip()
            logger.info("WebSearch: sparse results — broadening to %r", broad)
            broad_results = await _call(broad)
            if len(broad_results) > len(results):
                results = broad_results
        logger.info("WebSearch: %d results for %r", len(results), query[:60])
        return results
    except httpx.HTTPStatusError as e:
        logger.error("WebSearch: Tavily HTTP %s: %s", e.response.status_code, e.response.text[:200])
    except Exception as e:
        logger.error("WebSearch: Tavily request failed: %s", e)
    return []


def _format_results(results: List[dict]) -> str:
    lines = []
    for i, r in enumerate(results, 1):
        content = " ".join((r.get("content") or "").split())[:1200]
        published = r.get("published_date")
        lines.append(f"[{i}] {r.get('title') or 'Untitled'}"
                     + (f" (published {published})" if published else ""))
        lines.append(f"URL: {r.get('url', '')}")
        lines.append(content)
        lines.append("")
    return "\n".join(lines)


class WebSearchSubagent:
    """Searches the web and builds the synthesis prompt; the orchestrator streams it."""

    async def prepare(self, query: str, history: List[dict]) -> Tuple[List[dict], List[dict]]:
        """Returns (messages_for_llm, sources) where sources = [{title, url}]."""
        results = await fetch_tavily_results(query)
        sources = [{"title": r.get("title") or r.get("url", ""), "url": r.get("url", "")}
                   for r in results if r.get("url")]

        if results:
            user_turn = (f"Question: {query}\n\n<SEARCH_RESULTS retrieved=\"{date.today().isoformat()}\">\n"
                         f"{_format_results(results)}</SEARCH_RESULTS>")
            system = WEB_SEARCH_SYSTEM_PROMPT
        else:
            user_turn = query
            system = WEB_SEARCH_SYSTEM_PROMPT + "\n" + NO_RESULTS_NOTE

        # Only the last couple of turns — search results already use most of the budget.
        recent = [m for m in history if m["role"] != "system"][-4:]
        return MemoryInjector.build_prompt(recent, user_turn, system_prompt=system), sources

    @property
    def chain(self) -> List[str]:
        return config.ROUTE_CHAINS["web_search"]
