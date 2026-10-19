"""
backend/subagents/dsa_agent.py

Data Structures & Algorithms Subagent — handles queries routed to 'dsa'.

It is history-aware: follow-ups ("what's its complexity?", "now do it iteratively")
see the earlier turns. The rigid 6-section template is used only for full
problem-solving requests; quick conceptual questions get a direct answer, the way
a good tutor (or any daily AI assistant) would respond.
"""

import logging
from typing import List

from backend import config
from backend.pipeline.memory import MemoryInjector

logger = logging.getLogger(__name__)

DSA_SYSTEM_PROMPT = """\
You are PrismAI's algorithms specialist — an elite competitive programmer and CS professor. Today's date is {today}.

Choose the response shape that fits the request:

A) Full problem / "solve" / "implement" / LeetCode-style requests → use these sections:
### Problem Understanding
### Approach & Intuition
### Algorithm Steps
### Code Solution
(clean, commented code in a fenced block; Python unless the user asked for another language)
### Complexity Analysis
(time and space in Big-O, with a one-line justification each)
### Edge Cases

B) Conceptual questions, comparisons and follow-ups (e.g. "what is its time complexity?",
"why is it O(log n)?", "difference between BFS and DFS") → answer directly and concisely,
with a small example, table or snippet only if it helps. Do not force the sections above.

Use the conversation history to resolve what "it", "this", "that approach" refers to.
Be rigorous and correct; never invent complexity claims.
"""


class DSASubagent:
    """Builds the specialist prompt; the orchestrator streams it."""

    def build_messages(self, query: str, history: List[dict]) -> List[dict]:
        logger.info("DSA Agent: %s", query[:60])
        return MemoryInjector.build_prompt(history, query, system_prompt=DSA_SYSTEM_PROMPT)

    @property
    def chain(self) -> List[str]:
        return config.ROUTE_CHAINS["dsa"]
