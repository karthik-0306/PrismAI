"""
backend/subagents/dsa_agent.py

Phase 5: Data Structures & Algorithms Subagent.
Handles queries routed to the 'dsa' category in auto mode.

Workflow:
1. Grounding (Web Search via Tavily - stubbed with graceful fallback)
2. Prompt construction with strict 6-section structure enforcement
3. LLM Generation
"""

import logging
from backend.llm.client import LLMClient

logger = logging.getLogger(__name__)

# Primary model for complex algorithmic reasoning
DSA_PRIMARY_MODEL = "groq/openai/gpt-oss-120b"
DSA_FALLBACK_MODELS = [
    "groq/llama-3.3-70b-versatile",
    "gemini/gemini-3.5-flash"
]

DSA_SYSTEM_PROMPT = """\
You are an elite competitive programmer and computer science professor.
Your task is to explain and solve the given Data Structures & Algorithms problem.

You MUST format your response using EXACTLY these 6 markdown headers, in order:

### 1. Problem Understanding
(Briefly restate the problem, inputs, and expected outputs)

### 2. Approach & Intuition
(Explain the logic, data structures used, and why this approach was chosen)

### 3. Algorithm Steps
(Step-by-step plain English breakdown of the execution)

### 4. Code Solution
(Production-ready Python code with clear comments and type hints)

### 5. Complexity Analysis
(Time and Space complexity using Big-O notation, with a short justification)

### 6. Edge Cases
(What happens on empty input, extreme values, or weird constraints?)

Do NOT add any other top-level headers. Do NOT skip any sections.
Be concise but rigorous.
"""


class DSASubagent:
    """
    Specialized agent for handling DSA queries.
    """
    def __init__(self):
        self._llm = LLMClient()

    async def _ground_with_search(self, query: str) -> str:
        """
        Attempt to search the web for recent discussions or similar LeetCode problems.
        Currently stubs out the Tavily API call with a graceful fallback.
        """
        # TODO: Implement real Tavily API call here when key is available
        logger.debug("DSA Agent: Web search skipped (Tavily key pending)")
        return ""

    async def solve(self, query: str) -> str:
        """
        Process the DSA query and return a structured markdown response.
        """
        logger.info("DSA Agent: Solving query: %s...", query[:50])

        search_context = await self._ground_with_search(query)

        prompt_content = f"<USER_QUERY>\n{query}\n</USER_QUERY>"
        if search_context:
            prompt_content += f"\n\n<SEARCH_CONTEXT>\n{search_context}\n</SEARCH_CONTEXT>"

        messages = [
            {"role": "system", "content": DSA_SYSTEM_PROMPT},
            {"role": "user", "content": prompt_content}
        ]

        try:
            result = await self._llm.async_complete(
                model=DSA_PRIMARY_MODEL,
                messages=messages,
                fallback_models=DSA_FALLBACK_MODELS,
                temperature=0.1,  # Low temp for strict structural adherence
                max_tokens=4000
            )
            return result.content.strip()
        except Exception as e:
            logger.error("DSA Agent: LLM failure: %s", e)
            raise e
