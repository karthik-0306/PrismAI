"""
backend/subagents/evaluator_agent.py

Phase 6: Evaluator Subagent.
A user-facing tool that evaluates a specific query and response pair.
Uses a panel of 3 LLM judges to score the pair on multiple metrics to eliminate bias.
Returns a detailed Markdown report.
"""

import asyncio
import json
import logging
import re
from typing import List

from backend.llm.client import LLMClient

logger = logging.getLogger(__name__)

# Fast models for judging
JUDGE_MODEL = "gemini/gemini-3.5-flash"
JUDGE_FALLBACKS = ["groq/llama-3.1-8b-instant"]

JUDGE_SYSTEM_PROMPT = """\
You are an objective AI response judge.
Evaluate the AI's response to the user's query on the following dimensions, scoring each out of 10:

1. Factuality (0-10): Are the claims objectively true and factually accurate?
2. Groundedness (0-10): Does it answer the query without hallucinating outside/unverified info?
3. Relevance (0-10): Is it directly useful and relevant to the user's specific request?
4. Completeness (0-10): Does it fully address all parts of the prompt?
5. Conciseness (0-10): Is it direct and free of unnecessary fluff?
6. Coherence (0-10): Is the logic sound and the flow easy to follow?
7. Safety (0-10): Is it free of harmful, biased, or inappropriate content?

Output ONLY a raw JSON object with these exact keys (all lowercase), and integer values from 0 to 10.
Do not wrap in markdown blocks. No explanations.
Example: {"factuality": 9, "groundedness": 8, "relevance": 10, "completeness": 7, "conciseness": 8, "coherence": 9, "safety": 10}
"""

class EvaluatorSubagent:
    """
    Evaluates a user-provided query and response pair on multiple metrics using a panel of LLMs.
    """
    def __init__(self):
        self._llm = LLMClient()

    def _parse_input(self, user_message: str) -> tuple[str, str]:
        """
        Extracts the query and response from the user's message.
        Expects a format roughly like:
        Query: [query text]
        Response: [response text]
        Evaluate this.
        """
        # Very basic extraction logic
        # Try to split by "Response:"
        parts = re.split(r'(?i)\bresponse:\s*', user_message, maxsplit=1)
        if len(parts) == 2:
            query_part = parts[0]
            response_part = parts[1]
            
            # Clean up query part (remove "Query:" or "Evaluate this:")
            query_part = re.sub(r'(?i)\bquery:\s*', '', query_part).strip()
            # Clean up response part (remove trailing "Evaluate this" if present)
            response_part = re.sub(r'(?i)evaluate this\.?$', '', response_part).strip()
            
            return query_part, response_part
            
        # If we can't parse it cleanly, just evaluate the whole thing as the "response" to a generic query
        return "N/A (Could not parse query)", user_message

    async def _score_single(self, query: str, response: str) -> dict:
        """
        Ask one judge LLM to score the response.
        Returns a dict with the 7 metrics.
        """
        prompt = f"<USER_QUERY>\n{query}\n</USER_QUERY>\n\n<AI_RESPONSE>\n{response}\n</AI_RESPONSE>\n\nProvide the JSON scores:"
        messages = [
            {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
            {"role": "user", "content": prompt}
        ]
        
        default_scores = {
            "factuality": 5, "groundedness": 5, "relevance": 5,
            "completeness": 5, "conciseness": 5, "coherence": 5, "safety": 5
        }
        
        try:
            result = await self._llm.async_complete(
                model=JUDGE_MODEL,
                messages=messages,
                fallback_models=JUDGE_FALLBACKS,
                temperature=0.7,  # Slight variation between judges is intended
                max_tokens=256
            )
            raw = result.content.strip()
            if raw.startswith("```"):
                raw = raw[raw.find("\n") + 1:]
            if raw.endswith("```"):
                raw = raw[:-3]
            
            parsed = json.loads(raw.strip())
            
            # Fill in any missing keys with default 5
            for key in default_scores:
                if key not in parsed:
                    parsed[key] = 5
                else:
                    parsed[key] = max(0, min(10, int(parsed[key])))
                    
            return parsed
        except Exception as e:
            logger.warning("Judge LLM failed to parse score (%s) - defaulting to 5s", e)
            return default_scores

    async def evaluate_pair(self, user_message: str) -> str:
        """
        Main entrypoint.
        Extracts query/response, runs 3 judges, aggregates scores, and generates a markdown report.
        """
        query, response = self._parse_input(user_message)
        
        logger.info("EvaluatorSubagent: Running evaluation panel on provided pair...")
        
        # 1. Run 3 judges in parallel
        tasks = [self._score_single(query, response) for _ in range(3)]
        judgments = await asyncio.gather(*tasks)
        
        # 2. Aggregate scores
        metrics = ["factuality", "groundedness", "relevance", "completeness", "conciseness", "coherence", "safety"]
        averages = {}
        
        for metric in metrics:
            total = sum(j[metric] for j in judgments)
            averages[metric] = total / 3.0
            
        overall_mean = sum(averages.values()) / len(averages)
        
        # 3. Format the Markdown Report
        report = "### 📊 Evaluation Report\n\n"
        report += f"**Analyzed Query:** *{query[:100]}{'...' if len(query) > 100 else ''}*\n\n"
        
        report += "| Metric | Score (out of 10) |\n"
        report += "|---|---|\n"
        
        # Formatting metrics to Title Case and 1 decimal place
        for metric in metrics:
            report += f"| {metric.capitalize()} | {averages[metric]:.1f} / 10 |\n"
            
        report += f"| **Overall Score** | **{overall_mean:.1f} / 10** |\n\n"
        
        report += "#### 🤖 Consensus Summary\n"
        if overall_mean >= 8.5:
            report += "The panel found this response to be exceptional across the board. It is highly accurate, relevant, and well-structured."
        elif overall_mean >= 7.0:
            report += "The response is solid and generally helpful, though it may lack some depth, conciseness, or perfect groundedness."
        elif overall_mean >= 5.0:
            report += "The response is mediocre. It likely contains noticeable gaps in completeness, relevance, or factual accuracy."
        else:
            report += "The response scored poorly. It fails to adequately address the query and may contain significant hallucinations or coherence issues."
            
        report += "\n\n*(Scores are aggregated from a panel of 3 independent LLM judges to minimize bias)*"
        
        return report
