"""
Offline test harness: a scriptable fake LLM replaces litellm.acompletion, and each
test gets a fresh temporary SQLite database. No network, no API keys needed.

    python -m pytest backend/tests -q
"""

import json
import os
import tempfile
from pathlib import Path
from types import SimpleNamespace

# Must be set before any backend module is imported (config reads it at import time).
_TMP = Path(tempfile.mkdtemp(prefix="prismai-tests-"))
os.environ["PRISMAI_DB_PATH"] = str(_TMP / "bootstrap.db")
os.environ.setdefault("TAVILY_API_KEY", "")
os.environ["PRISMAI_EMBEDDINGS"] = "off"

import litellm  # noqa: E402
import pytest  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

from backend.database import connection  # noqa: E402
from backend.llm import client as llm_client  # noqa: E402


class FakeLLM:
    """
    Records every call and answers based on what kind of prompt it sees.

    Attributes tests can set:
      route_tasks   list of {"category", "query"} the router returns (default: general)
      fail          {model: exception or callable(call) -> exception|None}
      midstream_fail {model: exception} — raise after the first streamed chunk
      answer        callable(call) -> str for user-facing answers
    """

    def __init__(self):
        self.calls = []
        self.route_tasks = None
        self.fail = {}
        self.midstream_fail = {}
        self.answer = lambda call: f"answer #{len(self.answer_calls())}"

    # ── helpers for assertions ───────────────────────────────────────────────
    @staticmethod
    def kind_of(messages) -> str:
        text = "\n".join(m["content"] for m in messages)
        if "You route queries" in text:
            return "router"
        if "Write a short title" in text:
            return "title"
        if "QUERY COMPRESSOR" in text:
            return "rewriter"
        if "Rate the response: 0 = fully correct" in text:
            return "judge"
        if "strict AI response judge" in text:
            return "evaluator"
        return "answer"

    def answer_calls(self):
        return [c for c in self.calls if c["kind"] == "answer"]

    def router_calls(self):
        return [c for c in self.calls if c["kind"] == "router"]

    # ── the fake acompletion ─────────────────────────────────────────────────
    async def acompletion(self, **kwargs):
        model = kwargs["model"]
        call = {"model": model, "messages": kwargs["messages"], "kind": self.kind_of(kwargs["messages"]),
                "stream": kwargs.get("stream", False), "kwargs": kwargs}
        self.calls.append(call)

        failure = self.fail.get(model)
        if callable(failure) and not isinstance(failure, Exception):
            failure = failure(call)
        if failure:
            raise failure

        if call["kind"] == "router":
            text = json.dumps({"tasks": self.route_tasks or [
                {"category": "general", "query": kwargs["messages"][0]["content"].split("<MESSAGE>\n")[-1]
                 .split("\n</MESSAGE>")[0]}]})
        elif call["kind"] == "title":
            text = "Test Title"
        elif call["kind"] == "judge":
            text = '{"severity": 0, "reason": "fine"}'
        elif call["kind"] == "evaluator":
            text = json.dumps({k: 8 for k in ["factuality", "groundedness", "relevance", "completeness",
                                              "conciseness", "coherence", "safety"]} | {"verdict": "good"})
        elif call["kind"] == "rewriter":
            text = kwargs["messages"][-1]["content"]
        else:
            text = self.answer(call)

        if call["stream"]:
            return self._stream(text, model)
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=text))],
            usage=SimpleNamespace(prompt_tokens=10, completion_tokens=5),
        )

    async def _stream(self, text, model):
        pieces = [text[i:i + 5] for i in range(0, len(text), 5)] or [""]
        for i, piece in enumerate(pieces):
            if i == 1 and model in self.midstream_fail:
                raise self.midstream_fail[model]
            yield SimpleNamespace(choices=[SimpleNamespace(delta=SimpleNamespace(content=piece))])


def rate_limit(model="m"):
    return litellm.RateLimitError("Rate limit reached. Please try again in 2.5s.", llm_provider="groq", model=model)


def denied(model="m"):
    return litellm.NotFoundError("The model does not exist", llm_provider="groq", model=model)


@pytest.fixture
def fake_llm(monkeypatch):
    fake = FakeLLM()
    monkeypatch.setattr(llm_client.litellm, "acompletion", fake.acompletion)
    llm_client._health.clear()
    return fake


@pytest.fixture
def db_path(monkeypatch, tmp_path):
    path = tmp_path / "test.db"
    monkeypatch.setattr(connection, "DB_PATH", path)
    return path


@pytest.fixture
def api(fake_llm, db_path):
    from backend.main import app
    with TestClient(app) as client:
        yield client


def stream_chat(api, **body):
    """POST /api/chat/stream and return (events, full_text)."""
    with api.stream("POST", "/api/chat/stream", json=body) as resp:
        assert resp.status_code == 200, resp.read()
        events, text = [], ""
        for line in resp.iter_lines():
            if not line.startswith("data: "):
                continue
            raw = line[6:]
            if raw == "[DONE]":
                break
            event = json.loads(raw)
            events.append(event)
            if event["type"] == "token":
                text += event["content"]
            elif event["type"] == "reset":
                text = ""
    return events, text


def event(events, kind):
    return next((e for e in events if e["type"] == kind), None)
