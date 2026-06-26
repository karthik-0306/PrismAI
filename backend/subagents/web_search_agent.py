"""
backend/subagents/web_search_agent.py

Web Search Subagent — handles queries classified as 'web_search'.

Uses the Tavily Search API (built for AI agents) to fetch high-quality,
pre-extracted web content without needing any additional package beyond
httpx (already installed via FastAPI).

Workflow:
  1. Call Tavily API async via httpx → get top results (title, URL, content)
  2. Build a synthesis prompt with the search context
  3. Stream the LLM synthesis token by token back to the orchestrator
"""

import logging
import os
import httpx
from typing import AsyncGenerator, Tuple

from backend.llm.client import LLMClient

logger = logging.getLogger(__name__)

# Tavily API endpoint
TAVILY_API_URL = "https://api.tavily.com/search"

# Model for synthesizing search results into a clean answer
WEB_SEARCH_PRIMARY   = "gemini/gemini-3.5-flash"
WEB_SEARCH_FALLBACKS = [
    "groq/llama-3.3-70b-versatile",
    "groq/openai/gpt-oss-120b",
]

WEB_SEARCH_SYSTEM_PROMPT = """\
You are a helpful web research assistant with access to real-time search results.

Your task:
1. Read the search results carefully.
2. Write a well-organized, accurate answer to the user's question.
3. Cite your sources inline using markdown links, e.g. [Source Name](https://...).
4. At the end, add a "### Sources" section listing all URLs you referenced.

CRITICAL RULES:
- Ground your answer in the search results provided. They are real and current.
- For price queries (gold, stocks, currency), give the exact figures from the results.
- For local queries (e.g. "gold price in Tenali"), note that local prices track
  state/national rates closely and provide the rate you found with context.
- NEVER say you couldn't find results if results are provided — use them.
- If results are genuinely insufficient, still give your best estimate from training
  knowledge and clearly label it as an estimate.

Format your response using clean markdown: bold headers, bullet points, tables where helpful.
"""


async def _fetch_tavily_results(query: str, max_results: int = 5) -> list[dict]:
    """
    Async call to Tavily Search API.
    If the first query returns sparse results (< 2), automatically broadens
    the query by stripping hyper-local terms (e.g. "in Tenali" → national rate).
    """
    api_key = os.getenv("TAVILY_API_KEY", "")
    if not api_key or api_key == "dummy_tavily_key":
        logger.error("WebSearch: TAVILY_API_KEY not set. Cannot search.")
        return []

    async def _call_tavily(q: str) -> list[dict]:
        payload = {
            "query": q,
            "search_depth": "basic",
            "max_results": max_results,
            "include_answer": True,
            "include_raw_content": False,
        }
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }
        async with httpx.AsyncClient(timeout=15.0) as client:
            resp = await client.post(TAVILY_API_URL, json=payload, headers=headers)
            resp.raise_for_status()
            data = resp.json()

        results = data.get("results", [])
        tavily_answer = data.get("answer", "")
        if tavily_answer:
            results.insert(0, {
                "title": "Tavily Quick Answer",
                "url": "https://tavily.com",
                "content": tavily_answer,
                "score": 1.0,
            })
        return results

    try:
        # Strategy 1: Try the original query
        results = await _call_tavily(query)
        logger.info("WebSearch: Tavily returned %d results for: %s", len(results), query[:60])

        # Strategy 2: If sparse, broaden by removing hyper-local "in <city>" suffix
        if len(results) < 2 and " in " in query.lower():
            broad_query = query.lower().split(" in ")[0].strip()
            # Add "india today" for price/rate queries to keep results relevant
            price_keywords = {"price", "rate", "cost", "value"}
            if any(k in broad_query for k in price_keywords):
                broad_query = f"{broad_query} india today"
            logger.info("WebSearch: Sparse results — broadening query to: '%s'", broad_query)
            results = await _call_tavily(broad_query)
            logger.info("WebSearch: Broadened query returned %d results", len(results))

        return results

    except httpx.HTTPStatusError as e:
        logger.error("WebSearch: Tavily API error %s: %s", e.response.status_code, e.response.text)
        return []
    except Exception as e:
        logger.error("WebSearch: Tavily request failed: %s", e)
        return []



def _build_search_context(results: list[dict]) -> str:
    """
    Format Tavily results into a clean, LLM-readable context block.
    """
    if not results:
        return "No search results were returned by the search engine."

    lines = []
    for i, r in enumerate(results, 1):
        lines.append(f"[{i}] **{r.get('title', 'Untitled')}**")
        lines.append(f"    URL: {r.get('url', '')}")
        lines.append(f"    Content: {r.get('content', '')}")
        lines.append("")

    return "\n".join(lines)


class WebSearchSubagent:
    """
    Specialized agent for queries that require real-time web information.
    Uses Tavily Search API for high-quality, AI-optimized results.
    """

    def __init__(self):
        self._llm = LLMClient()

    async def solve(self, query: str) -> str:
        """
        Full (non-streaming) web search + LLM synthesis.
        """
        logger.info("WebSearch Agent: Searching (Tavily) for: %s...", query[:60])

        results = await _fetch_tavily_results(query)
        search_context = _build_search_context(results)

        prompt_content = (
            f"<USER_QUESTION>\n{query}\n</USER_QUESTION>\n\n"
            f"<SEARCH_RESULTS>\n{search_context}\n</SEARCH_RESULTS>"
        )

        messages = [
            {"role": "system", "content": WEB_SEARCH_SYSTEM_PROMPT},
            {"role": "user",   "content": prompt_content},
        ]

        try:
            result = await self._llm.async_complete(
                model=WEB_SEARCH_PRIMARY,
                messages=messages,
                fallback_models=WEB_SEARCH_FALLBACKS,
                temperature=1.0,   # Gemini requires temperature=1.0
                max_tokens=2048,
            )
            return result.content.strip()
        except Exception as e:
            logger.error("WebSearch Agent: LLM synthesis failed: %s", e)
            raise

    async def stream_solve(self, query: str) -> AsyncGenerator[Tuple[str, str], None]:
        """
        Streaming version: fetches Tavily results first (async, non-blocking),
        then streams the LLM synthesis token by token.

        Yields (chunk: str, model_used: str) tuples — same contract as
        LLMClient.async_stream() and DSASubagent.stream_solve().
        """
        logger.info("WebSearch Agent: Streaming search (Tavily) for: %s...", query[:60])

        # Tavily fetch is now fully async — no thread pool needed
        results = await _fetch_tavily_results(query)
        search_context = _build_search_context(results)

        prompt_content = (
            f"<USER_QUESTION>\n{query}\n</USER_QUESTION>\n\n"
            f"<SEARCH_RESULTS>\n{search_context}\n</SEARCH_RESULTS>"
        )

        messages = [
            {"role": "system", "content": WEB_SEARCH_SYSTEM_PROMPT},
            {"role": "user",   "content": prompt_content},
        ]

        async for chunk, model_used in self._llm.async_stream(
            model=WEB_SEARCH_PRIMARY,
            messages=messages,
            fallback_models=WEB_SEARCH_FALLBACKS,
            temperature=1.0,
            max_tokens=2048,
        ):
            yield chunk, model_used
