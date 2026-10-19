"""
backend/pipeline/router.py

The SmartRouter reads the user's query (plus a short snippet of the conversation)
and decides:
  1. Which category of task it is.
  2. Whether it contains several INDEPENDENT intents that should be answered
     by different specialists in parallel.
  3. A self-contained version of each task — follow-ups like "what is its time
     complexity?" are rewritten to "What is the time complexity of binary search?"
     using the conversation context. Without this, subagents (which are given
     the task text) had no idea what "it" referred to.

It calls the cheap utility chain only. On any failure it falls back to a single
"general" task with the original query, so the orchestrator always gets a valid
list and never needs to handle None.
"""

import json
import logging
import re
from typing import List

from backend import config
from backend.llm.client import LLMClient

logger = logging.getLogger(__name__)

VALID_CATEGORIES = {"dsa", "coding", "reasoning", "math", "summarize", "evaluate", "general", "fast", "web_search"}
MAX_TASKS = 3

# Obvious small talk is answered by the fast route without spending a router call
# (saves ~1s of latency and a slice of the per-minute token budget).
_SMALL_TALK_RE = re.compile(
    r"^\s*(hi+|hey+|hello+|yo|hola|namaste|good (morning|afternoon|evening|night)|thanks?( you)?( so much)?|"
    r"thank you|thx|ty|ok(ay)?|cool|great|nice|awesome|bye|goodbye|see you|got it|perfect|lol|wow)"
    r"[\s!.,?]*(there|prism(ai)?|buddy|man|bro|again|a lot)?[\s!.,?😀-🙏]*$",
    re.IGNORECASE,
)

CLASSIFICATION_PROMPT = """\
You route queries for a multi-agent AI assistant. Classify the LATEST user message.

Categories:
- web_search: needs current/real-time info (news, prices, weather, scores, releases, "latest", "today", "now", events after 2024) or the user asks to search/look up online.
- dsa: data structures, algorithms, Big-O, LeetCode-style problems (even if code is requested).
- coding: real-world software work — apps, APIs, debugging, frameworks, DevOps, system design, SQL.
- math: calculation, algebra, calculus, statistics, probability, proofs.
- reasoning: logic puzzles, riddles, argument analysis, step-by-step deduction.
- summarize: shorten/condense a GIVEN text (the whole text is one task).
- evaluate: score/grade/rate/critique a GIVEN response, text or code.
- fast: greetings, thanks, trivial facts, definitions, unit conversions.
- general: everything else (explanations, advice, writing, chit-chat, opinions).

Rules:
1. Priority: web_search > evaluate > dsa > summarize > math > others.
2. Split into multiple tasks ONLY for clearly independent requests from DIFFERENT categories (max 3). Never split evaluate or summarize payloads.
3. Each task "query" must be self-contained: resolve pronouns and follow-ups ("it", "that", "more", "the second one") using the conversation, and keep any code/text payload verbatim. Do NOT answer the question.
4. A follow-up keeps the category of what it is about (e.g. "what's its complexity?" after binary search → dsa).

Respond with JSON only:
{{"tasks": [{{"category": "<category>", "query": "<self-contained task>"}}]}}

Conversation so far (may be empty):
<CONTEXT>
{context}
</CONTEXT>

Latest user message:
<MESSAGE>
{query}
</MESSAGE>"""


class SmartRouter:
    """Stateless — safe to share across concurrent async requests."""

    def __init__(self):
        self._llm = LLMClient()

    async def classify_and_split(self, query: str, context: str = "") -> List[dict]:
        """
        Returns a non-empty list of {"category": str, "sub_query": str}.
        Never raises.
        """
        if _SMALL_TALK_RE.match(query):
            logger.info("Router → ['fast'] (small talk, no LLM call)")
            return [{"category": "fast", "sub_query": query}]

        prompt = CLASSIFICATION_PROMPT.format(context=context.strip() or "(none)", query=query)
        try:
            result = await self._llm.complete(
                models=config.UTILITY_CHAIN,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=max(300, min(1500, len(query) // 3 + 200)),  # room to echo payloads
                low_reasoning=True,
                json_mode=True,
            )
        except Exception as e:
            logger.warning("Router LLM call failed: %s — using general fallback", e)
            return self._fallback(query)

        tasks = self._parse(result.content, query)
        logger.info("Router → %s", [t["category"] for t in tasks])
        return tasks

    # ── Private helpers ────────────────────────────────────────────────────────

    def _parse(self, raw: str, query: str) -> List[dict]:
        data = _extract_json(raw)
        if isinstance(data, dict):
            data = data.get("tasks") or data.get("items") or [data]
        if not isinstance(data, list):
            logger.warning("Router returned unparseable output: %r", raw[:200])
            return self._fallback(query)

        tasks: List[dict] = []
        for item in data:
            if not isinstance(item, dict):
                continue
            category = str(item.get("category", "")).strip().lower()
            sub_query = str(item.get("query") or item.get("sub_query") or "").strip()
            if category not in VALID_CATEGORIES:
                category = "general"
            if not sub_query:
                continue
            tasks.append({"category": category, "sub_query": sub_query})

        if not tasks:
            return self._fallback(query)

        # Merge tasks that landed in the same category — one specialist can answer both.
        merged: dict = {}
        for t in tasks:
            if t["category"] in merged:
                merged[t["category"]]["sub_query"] += "\n\n" + t["sub_query"]
            else:
                merged[t["category"]] = dict(t)
        tasks = list(merged.values())[:MAX_TASKS]

        # Payload-carrying tasks must never be split from the rest of the message.
        if len(tasks) > 1 and any(t["category"] in ("evaluate", "summarize") for t in tasks):
            keep = next(t for t in tasks if t["category"] in ("evaluate", "summarize"))
            return [{"category": keep["category"], "sub_query": query}]
        return tasks

    @staticmethod
    def _fallback(query: str) -> List[dict]:
        return [{"category": "general", "sub_query": query}]


def _extract_json(raw: str):
    text = raw.strip()
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text)
    try:
        return json.loads(text)
    except (ValueError, TypeError):
        pass
    for open_ch, close_ch in (("{", "}"), ("[", "]")):
        start, end = text.find(open_ch), text.rfind(close_ch)
        if start != -1 and end > start:
            try:
                return json.loads(text[start:end + 1])
            except (ValueError, TypeError):
                continue
    return None
