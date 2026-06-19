"""
backend/pipeline/router.py

The SmartRouter reads the user's query and decides:
  1. What category of task is it? (dsa, coding, math, reasoning, summarize, fast, general)
  2. If the query contains multiple unrelated intents, split them into separate sub-queries.

This module calls the CHEAP utility LLM (llama-3.1-8b-instant) — never the expensive models.
The expensive models are only called by the orchestrator AFTER the router has decided the route.

How it works:
  - We send the user's query to the utility LLM with a strict classification prompt.
  - The LLM must respond ONLY with a JSON array. Example:
      [{"category": "dsa", "sub_query": "Explain binary search"}]
      [{"category": "dsa",  "sub_query": "..."}, {"category": "math", "sub_query": "..."}]
  - We parse and validate the JSON.
  - If the LLM returns garbage or invalid JSON, we fall back to:
      [{"category": "general", "sub_query": original_query}]
    This guarantees the orchestrator always gets a valid list — no crashes.

Categories and their meaning:
  dsa       — data structures, algorithms, leetcode problems, complexity analysis
  coding    — writing code, debugging, refactoring, programming concepts
  reasoning — logic puzzles, argument analysis, step-by-step deduction
  math      — arithmetic, algebra, calculus, proofs, numerical computation
  summarize — condense a given block of text into a shorter form
  fast      — trivial/quick questions that need a fast cheap answer
  general   — anything that doesn't fit neatly into the above categories
"""

import json
import logging
from typing import List

from backend.llm.client import LLMClient  # our unified LLM abstraction layer

logger = logging.getLogger(__name__)

# ── Utility model config ───────────────────────────────────────────────────────
# These are the CHEAP models used only for internal tasks (classify, judge, aggregate).
# They are fast and inexpensive — NOT used to answer the user's actual question.
UTILITY_PRIMARY  = "gemini/gemini-3.5-flash"
UTILITY_FALLBACK = "groq/llama-3.1-8b-instant"

# ── Known valid categories ─────────────────────────────────────────────────────
# If the LLM returns a category not in this set, we normalize it to "general".
VALID_CATEGORIES = {"dsa", "coding", "reasoning", "math", "summarize", "fast", "general"}

# ── Classification prompt ──────────────────────────────────────────────────────
# This is the exact instruction we send to the utility LLM.
# The {query} placeholder is replaced with the actual user message at runtime.
#
# DESIGN GOAL: Be extremely strict about category boundaries.
# The LLM must follow the PRIORITY RULES — no free interpretation.
CLASSIFICATION_PROMPT = """\
You are a strict query classifier for an AI routing system.
Classify the user's query into one or more of these 7 categories:

CATEGORY DEFINITIONS (memorize these boundaries exactly):
- dsa       : Data structures (arrays, linked lists, trees, graphs, stacks, queues, heaps, tries)
              OR algorithms (sorting, searching, dynamic programming, greedy, backtracking, recursion)
              OR complexity analysis (Big-O, time/space complexity)
              OR competitive programming / LeetCode-style problems
              IMPORTANT: If you are asked to WRITE CODE for a data structure or algorithm, it is STILL dsa, not coding.
- coding    : Real-world software engineering ONLY: building APIs, debugging production bugs,
              open source contributions, web/app development, system design, refactoring, DevOps.
              NOT for algorithm implementation or data structure coding — that is dsa.
- math      : Pure mathematics: arithmetic, algebra, calculus, number theory, proofs, statistics, probability
- reasoning : Logic puzzles, argument analysis, critical thinking, step-by-step deduction
- summarize : The user wants a SHORTER version of a GIVEN block of text.
              The entire text to summarize is ONE summarize task — do not split the text into parts.
- fast      : Trivial factual lookups, simple definitions, yes/no questions, unit conversions
- general   : Anything that does not fit any of the above categories

PRIORITY RULES (apply in this strict order):
1. If the query involves data structures OR algorithms OR Big-O → use dsa ONLY, even if writing code is required.
2. If the query asks to summarize a given text → use summarize ONLY for the whole request (one item).
3. If the query asks to prove OR compute mathematically → use math ONLY (not reasoning).
4. Only split if the query has TWO clearly INDEPENDENT intents from DIFFERENT categories.
5. Never duplicate the same sub_query under two different categories.

OUTPUT FORMAT:
Respond ONLY with a valid JSON array. No explanation. No markdown code fences. No extra text.
[
  {{"category": "<category>", "sub_query": "<self-contained question>"}},
  ...
]

EXAMPLES:
Query: "Explain binary search step by step"
Output: [{{"category": "dsa", "sub_query": "Explain binary search step by step"}}]

Query: "Write Python code to reverse a linked list in place"
Output: [{{"category": "dsa", "sub_query": "Write Python code to reverse a linked list in place"}}]

Query: "Prove quicksort is O(n log n) AND explain how merge sort works"
Output: [{{"category": "math", "sub_query": "Prove quicksort average time complexity is O(n log n)"}}, {{"category": "dsa", "sub_query": "Explain how merge sort works"}}]

Query: "Summarize: The quick brown fox jumps over the lazy dog. This sentence is used in typing exercises."
Output: [{{"category": "summarize", "sub_query": "Summarize: The quick brown fox jumps over the lazy dog. This sentence is used in typing exercises."}}]

Query: "What is 15 multiplied by 7?"
Output: [{{"category": "fast", "sub_query": "What is 15 multiplied by 7?"}}]

Now classify this query:
{query}"""


class SmartRouter:
    """
    Classifies a user query into one or more (category, sub_query) pairs.

    Instantiated once as a module-level singleton in orchestrator.py.
    Stateless — safe to share across concurrent async requests.
    """

    def __init__(self):
        # LLMClient is our wrapper around LiteLLM — handles fallbacks automatically
        self._llm = LLMClient()

    async def classify_and_split(self, query: str) -> List[dict]:
        """
        Classify the user query and split it into sub-queries if it contains
        multiple distinct intents.

        Args:
            query: The raw user input text (or the rewritten query in Phase 2+).
        Returns:
            List[dict]: A list of dicts, each with keys "category" and "sub_query".
                        Always has at least one item. Never empty. Never raises.
        Example returns:
            Single intent:   [{"category": "dsa",  "sub_query": "Explain quicksort"}]
            Compound intent: [{"category": "dsa",  "sub_query": "Explain quicksort"},
                              {"category": "math", "sub_query": "Prove O(n log n)"}]
        """
        # ── 1. Build the classification prompt ──────────────────────────────
        # Inject the actual user query into the prompt template
        prompt_text = CLASSIFICATION_PROMPT.format(query=query)

        # Wrap in the message format LiteLLM expects: list of role/content dicts
        messages = [{"role": "user", "content": prompt_text}]

        # ── 2. Call the cheap utility LLM ────────────────────────────────────
        try:
            result = await self._llm.async_complete(
                model=UTILITY_PRIMARY,
                messages=messages,
                fallback_models=[UTILITY_FALLBACK],
                temperature=0.0,    # 0.0 = deterministic, consistent JSON output
                max_tokens=512,     # classification response is always short
            )
            raw_text = result.content.strip()
            logger.debug("Router raw LLM output: %s", raw_text)

        except Exception as e:
            # If the LLM call itself fails entirely (all fallbacks exhausted),
            # log the error and return the safe fallback immediately.
            logger.warning("Router LLM call failed: %s — using general fallback", e)
            return self._fallback(query)

        # ── 3. Parse the JSON response ───────────────────────────────────────
        # The LLM is instructed to return only JSON, but LLMs sometimes add
        # markdown fences like ```json ... ``` — we strip those defensively.
        cleaned = self._strip_markdown_fences(raw_text)

        try:
            parsed = json.loads(cleaned)
        except json.JSONDecodeError as e:
            logger.warning("Router JSON parse failed: %s | raw: %s", e, raw_text)
            return self._fallback(query)

        # ── 4. Validate structure ────────────────────────────────────────────
        # parsed must be a non-empty list of dicts with "category" and "sub_query"
        if not isinstance(parsed, list) or len(parsed) == 0:
            logger.warning("Router returned non-list or empty list: %s", parsed)
            return self._fallback(query)

        validated = []
        for item in parsed:
            # Each item must be a dict with both required keys
            if not isinstance(item, dict):
                logger.warning("Router item is not a dict: %s", item)
                continue

            category  = item.get("category", "").strip().lower()
            sub_query = item.get("sub_query", "").strip()

            # Normalize unknown categories to "general"
            if category not in VALID_CATEGORIES:
                logger.warning(
                    "Router returned unknown category '%s' — normalizing to 'general'", category
                )
                category = "general"

            # Skip items with empty sub_query
            if not sub_query:
                logger.warning("Router returned empty sub_query — skipping item")
                continue

            validated.append({"category": category, "sub_query": sub_query})

        # If validation wiped out all items, fall back to general
        if not validated:
            logger.warning("Router validation produced empty list — using general fallback")
            return self._fallback(query)

        logger.info(
            "Router classified query into %d sub-task(s): %s",
            len(validated),
            [(v["category"]) for v in validated]
        )
        return validated

    # ── Private helpers ────────────────────────────────────────────────────────

    def _fallback(self, query: str) -> List[dict]:
        """
        Safe fallback used whenever the LLM call fails, returns bad JSON,
        or returns an empty/invalid structure.

        Returns the original query as a single "general" sub-task.
        The orchestrator always gets a valid list — it never needs to handle None.

        Args:
            query: the original user input text.
        Returns:
            List[dict]: single-item list with category "general".
        """
        return [{"category": "general", "sub_query": query}]

    @staticmethod
    def _strip_markdown_fences(text: str) -> str:
        """
        Remove markdown code fences if the LLM wraps its JSON in them.
        Example input:  ```json\n[{"category": ...}]\n```
        Example output: [{"category": ...}]

        Args:
            text: raw LLM output string.
        Returns:
            str: text with leading/trailing markdown fences removed.
        """
        # Remove opening fence (```json or just ```)
        if text.startswith("```"):
            # Find the end of the first line (the fence line)
            first_newline = text.find("\n")
            if first_newline != -1:
                text = text[first_newline + 1:]

        # Remove closing fence
        if text.endswith("```"):
            text = text[:-3]

        return text.strip()
