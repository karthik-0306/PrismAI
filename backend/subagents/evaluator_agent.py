"""
backend/subagents/evaluator_agent.py

Evaluator Subagent — a user-facing tool that scores a query/response pair.

A panel of three *different* models (config.JUDGE_PANEL) scores the pair on seven
metrics in parallel; scores are averaged across the judges that answered and
rendered as a Markdown report. Judges that fail are left out of the average
instead of silently contributing fake 5/10 scores.

Input handling:
  - "Query: … Response: …" (or "Question:/Answer:") is parsed explicitly.
  - "evaluate your last answer" with no pair given → evaluates the previous
    assistant turn in this conversation.
  - Anything else is evaluated as a standalone response.
"""

import asyncio
import json
import logging
import re
from typing import List, Optional, Tuple

from backend import config
from backend.llm.client import LLMClient

logger = logging.getLogger(__name__)

METRICS = ["factuality", "groundedness", "relevance", "completeness", "conciseness", "coherence", "safety"]

JUDGE_SYSTEM_PROMPT = """\
You are an objective, strict AI response judge.
Score the AI response to the user's query on each dimension from 0 to 10:
factuality (claims are true), groundedness (no unsupported/hallucinated content),
relevance (addresses the actual request), completeness (covers every part),
conciseness (no fluff), coherence (logical, well organized), safety (no harmful content).
Also give a one-sentence "verdict" naming the biggest strength or weakness.

Respond with JSON only, integer scores:
{"factuality": 0, "groundedness": 0, "relevance": 0, "completeness": 0, "conciseness": 0, "coherence": 0, "safety": 0, "verdict": "..."}
"""

_PAIR_RE = re.compile(
    r"(?is)(?:query|question|prompt)\s*[:\-]\s*(?P<q>.+?)\s*(?:response|answer|reply|output)\s*[:\-]\s*(?P<r>.+)"
)
_LAST_ANSWER_RE = re.compile(r"(?i)\b(your|the|that|this)\s+(last|previous|above)?\s*(answer|response|reply)\b")


class EvaluatorSubagent:
    def __init__(self):
        self._llm = LLMClient()

    def parse_input(self, message: str, history: List[dict]) -> Tuple[str, str]:
        m = _PAIR_RE.search(message)
        if m:
            response = re.sub(r"(?i)\s*(please\s+)?evaluate( this)?\.?\s*$", "", m.group("r")).strip()
            return m.group("q").strip().strip("'\""), response

        turns = [t for t in history if t["role"] in ("user", "assistant")]
        if _LAST_ANSWER_RE.search(message) and len(turns) >= 2 and turns[-1]["role"] == "assistant":
            return turns[-2]["content"], turns[-1]["content"]

        return "(not provided — judge the response on its own merits)", message

    async def _score_single(self, model: str, query: str, response: str) -> Optional[dict]:
        prompt = (f"<USER_QUERY>\n{query[:6000]}\n</USER_QUERY>\n\n"
                  f"<AI_RESPONSE>\n{response[:12000]}\n</AI_RESPONSE>")
        try:
            result = await self._llm.complete(
                models=[model, *[m for m in config.UTILITY_CHAIN if m != model]],
                messages=[{"role": "system", "content": JUDGE_SYSTEM_PROMPT},
                          {"role": "user", "content": prompt}],
                temperature=0.2,
                max_tokens=400,
                low_reasoning=True,
                json_mode=True,
            )
            raw = re.sub(r"^```(?:json)?\s*|\s*```$", "", result.content.strip())
            start, end = raw.find("{"), raw.rfind("}")
            parsed = json.loads(raw[start:end + 1])
            scores = {k: max(0, min(10, int(round(float(parsed[k]))))) for k in METRICS if k in parsed}
            if len(scores) < len(METRICS) // 2:
                return None
            scores["_verdict"] = str(parsed.get("verdict", "")).strip()
            scores["_model"] = result.model_used
            return scores
        except Exception as e:
            logger.warning("Evaluator judge %s failed: %s", model, e)
            return None

    async def evaluate(self, message: str, history: List[dict]) -> Tuple[str, List[str]]:
        """Returns (markdown_report, models_used)."""
        query, response = self.parse_input(message, history)
        judgments = [j for j in await asyncio.gather(
            *[self._score_single(m, query, response) for m in config.JUDGE_PANEL]) if j]

        if not judgments:
            return ("I couldn't run the evaluation panel right now — all judge models are "
                    "unavailable. Please try again in a minute.", [])

        averages = {}
        for metric in METRICS:
            values = [j[metric] for j in judgments if metric in j]
            averages[metric] = sum(values) / len(values) if values else 0.0
        overall = sum(averages.values()) / len(averages)

        report = "### 📊 Evaluation Report\n\n"
        short_q = " ".join(query.split())
        report += f"**Query:** *{short_q[:140]}{'…' if len(short_q) > 140 else ''}*\n\n"
        report += "| Metric | Score |\n|---|---|\n"
        for metric in METRICS:
            report += f"| {metric.capitalize()} | {averages[metric]:.1f} / 10 |\n"
        report += f"| **Overall** | **{overall:.1f} / 10** |\n\n"

        if overall >= 8.5:
            summary = "Excellent — accurate, relevant and well structured."
        elif overall >= 7.0:
            summary = "Solid and helpful, with some room to improve depth, concision or grounding."
        elif overall >= 5.0:
            summary = "Mediocre — noticeable gaps in completeness, relevance or accuracy."
        else:
            summary = "Poor — fails to address the query adequately and may contain errors."
        report += f"#### Consensus\n{summary}\n\n"

        verdicts = [(config.model_label(j["_model"]), j["_verdict"]) for j in judgments if j.get("_verdict")]
        if verdicts:
            report += "#### Judge notes\n" + "\n".join(f"- **{name}:** {v}" for name, v in verdicts) + "\n\n"
        report += f"*Scores averaged across {len(judgments)} independent judge model(s).*"

        return report, [j["_model"] for j in judgments]
