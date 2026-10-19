"""
End-to-end pipeline tests through the HTTP API with a fake LLM.
Each test pins down a bug that existed in the previous version.
"""

import asyncio
import uuid

from backend import config
from backend.database import queries
from backend.tests.conftest import denied, event, rate_limit, stream_chat


def new_session():
    return str(uuid.uuid4())


def roles(messages):
    return [m["role"] for m in messages]


# ── Multi-turn conversation ──────────────────────────────────────────────────

def test_second_message_succeeds_with_clean_history(api, fake_llm):
    sid = new_session()
    events1, text1 = stream_chat(api, session_id=sid, message="Hi, my name is Ravi.")
    meta1 = event(events1, "metadata")
    assert meta1 and text1 == "answer #1"

    events2, text2 = stream_chat(api, session_id=sid, chat_id=meta1["chat_id"], message="What is my name?")
    assert event(events2, "error") is None
    assert event(events2, "metadata")["chat_id"] == meta1["chat_id"]

    prompt = fake_llm.answer_calls()[-1]["messages"]
    # The current question must appear exactly once (it used to be sent twice).
    assert roles(prompt) == ["system", "user", "assistant", "user"]
    assert prompt[1]["content"] == "Hi, my name is Ravi."
    assert prompt[2]["content"] == "answer #1"
    assert prompt[3]["content"] == "What is my name?"


def test_many_turns_in_a_row(api, fake_llm):
    sid, chat_id = new_session(), None
    for i in range(6):
        events, _ = stream_chat(api, session_id=sid, chat_id=chat_id, message=f"question {i}")
        assert event(events, "error") is None, events
        chat_id = event(events, "metadata")["chat_id"]
    msgs = api.get(f"/api/chats/{chat_id}/messages", params={"session_id": sid}).json()
    assert [m["role"] for m in msgs] == ["user", "assistant"] * 6


def test_failed_turn_does_not_poison_the_next_one(api, fake_llm):
    sid = new_session()
    chat_id = event(stream_chat(api, session_id=sid, message="first")[0], "metadata")["chat_id"]

    # Every model fails for the answer → error event, but the request completes cleanly.
    fake_llm.fail = {m: (lambda call: rate_limit() if call["kind"] == "answer" else None) for m in config.ALL_MODELS}
    events, _ = stream_chat(api, session_id=sid, chat_id=chat_id, message="this one fails")
    assert event(events, "error") is not None
    assert "rate limit" in event(events, "error")["detail"].lower()

    # Models recover; the orphaned question must not be resent as a second user turn.
    fake_llm.fail = {}
    from backend.llm import client as llm_client
    llm_client._health.clear()
    events, _ = stream_chat(api, session_id=sid, chat_id=chat_id, message="this one works")
    assert event(events, "metadata") is not None
    prompt = fake_llm.answer_calls()[-1]["messages"]
    assert roles(prompt) == ["system", "user", "assistant", "user"]
    assert all(m["content"] != "this one fails" for m in prompt)


# ── Fallback chains / rate limits ────────────────────────────────────────────

def test_rate_limited_primary_falls_back_and_is_then_skipped(api, fake_llm):
    primary = config.ROUTE_CHAINS["general"][0]
    fake_llm.fail = {primary: rate_limit()}
    sid = new_session()

    events, text = stream_chat(api, session_id=sid, message="hello there")
    assert text.startswith("answer")
    assert event(events, "metadata")["models_used"][0] != primary

    attempts_before = sum(1 for c in fake_llm.calls if c["model"] == primary)
    stream_chat(api, session_id=sid, message="again")
    attempts_after = sum(1 for c in fake_llm.calls if c["model"] == primary)
    assert attempts_after == attempts_before, "cooling model should not be retried immediately"


def test_dead_model_is_disabled(api, fake_llm):
    dead = config.UTILITY_CHAIN[0]
    fake_llm.fail = {dead: denied()}
    sid = new_session()
    stream_chat(api, session_id=sid, message="one")
    first = sum(1 for c in fake_llm.calls if c["model"] == dead)
    stream_chat(api, session_id=sid, message="two")
    assert sum(1 for c in fake_llm.calls if c["model"] == dead) == first


def test_midstream_failure_resets_partial_text(api, fake_llm):
    primary = config.ROUTE_CHAINS["general"][0]
    fake_llm.midstream_fail = {primary: rate_limit()}
    events, text = stream_chat(api, session_id=new_session(), message="explain photosynthesis")
    assert event(events, "reset") is not None
    assert text.startswith("answer")   # only the fallback's full answer survives


# ── Routing with context ─────────────────────────────────────────────────────

def test_router_sees_conversation_for_follow_ups(api, fake_llm):
    sid = new_session()
    fake_llm.route_tasks = [{"category": "dsa", "query": "Explain binary search"}]
    events, _ = stream_chat(api, session_id=sid, message="Explain binary search")
    chat_id = event(events, "metadata")["chat_id"]

    fake_llm.route_tasks = [{"category": "dsa", "query": "Time complexity of binary search"}]
    events, _ = stream_chat(api, session_id=sid, chat_id=chat_id, message="What is its time complexity?")
    router_prompt = fake_llm.router_calls()[-1]["messages"][0]["content"]
    assert "Explain binary search" in router_prompt
    # The DSA specialist also receives the history now.
    dsa_prompt = fake_llm.answer_calls()[-1]["messages"]
    assert roles(dsa_prompt) == ["system", "user", "assistant", "user"]
    assert event(events, "metadata")["categories_used"] == ["dsa"]


def test_compound_query_streams_without_fallback(api, fake_llm):
    fake_llm.route_tasks = [
        {"category": "dsa", "query": "Explain quicksort"},
        {"category": "math", "query": "Compute 17*23"},
    ]
    sid = new_session()
    events, text = stream_chat(api, session_id=sid, message="Explain quicksort and compute 17*23")
    meta = event(events, "metadata")
    assert meta["categories_used"] == ["dsa", "math"]
    assert text and event(events, "fallback") is None
    msgs = api.get(f"/api/chats/{meta['chat_id']}/messages", params={"session_id": sid}).json()
    assert [m["role"] for m in msgs] == ["user", "assistant"]   # user message saved once


def test_web_search_without_results_still_answers(api, fake_llm, monkeypatch):
    from backend.subagents import web_search_agent

    async def no_results(query, max_results=6):
        return []
    monkeypatch.setattr(web_search_agent, "fetch_tavily_results", no_results)
    fake_llm.route_tasks = [{"category": "web_search", "query": "latest OpenAI news"}]
    events, text = stream_chat(api, session_id=new_session(), message="latest OpenAI news?")
    assert text and event(events, "metadata")["categories_used"] == ["web_search"]
    assert "unavailable" in fake_llm.answer_calls()[-1]["messages"][0]["content"]


def test_evaluate_last_answer_uses_history(api, fake_llm):
    sid = new_session()
    events, _ = stream_chat(api, session_id=sid, message="What is a heap?")
    chat_id = event(events, "metadata")["chat_id"]
    fake_llm.route_tasks = [{"category": "evaluate", "query": "evaluate your last answer"}]
    events, text = stream_chat(api, session_id=sid, chat_id=chat_id, message="evaluate your last answer")
    assert "Evaluation Report" in text and "8.0 / 10" in text
    judge_prompt = next(c for c in fake_llm.calls if c["kind"] == "evaluator")["messages"][1]["content"]
    assert "What is a heap?" in judge_prompt


# ── Chat lifecycle ───────────────────────────────────────────────────────────

def test_new_chat_gets_generated_title(api, fake_llm):
    sid = new_session()
    events, _ = stream_chat(api, session_id=sid, message="tell me about rust ownership")
    assert event(events, "metadata")["title"] == "Test Title"
    assert api.get("/api/chats", params={"session_id": sid}).json()[0]["title"] == "Test Title"


def test_stale_chat_id_is_recreated(api, fake_llm):
    sid, stale = new_session(), str(uuid.uuid4())
    events, _ = stream_chat(api, session_id=sid, chat_id=stale, message="hello after a server reset")
    assert event(events, "metadata")["chat_id"] == stale


def test_cannot_write_into_or_read_another_sessions_chat(api, fake_llm):
    owner, intruder = new_session(), new_session()
    chat_id = event(stream_chat(api, session_id=owner, message="secret")[0], "metadata")["chat_id"]

    events, _ = stream_chat(api, session_id=intruder, chat_id=chat_id, message="hijack")
    assert event(events, "metadata")["chat_id"] != chat_id
    assert api.get(f"/api/chats/{chat_id}/messages", params={"session_id": intruder}).status_code == 404
    assert api.delete(f"/api/chats/{chat_id}", params={"session_id": intruder}).status_code == 404
    assert len(api.get(f"/api/chats/{chat_id}/messages", params={"session_id": owner}).json()) == 2


def test_regenerate_replaces_last_turn(api, fake_llm):
    sid = new_session()
    chat_id = event(stream_chat(api, session_id=sid, message="q1")[0], "metadata")["chat_id"]
    stream_chat(api, session_id=sid, chat_id=chat_id, message="q2")
    events, text = stream_chat(api, session_id=sid, chat_id=chat_id, message="q2", regenerate=True)
    msgs = api.get(f"/api/chats/{chat_id}/messages", params={"session_id": sid}).json()
    assert [m["content"] for m in msgs if m["role"] == "user"] == ["q1", "q2"]
    assert msgs[-1]["content"] == text


def test_rename_search_delete(api, fake_llm):
    sid = new_session()
    chat_id = event(stream_chat(api, session_id=sid, message="graphs and 100% coverage")[0], "metadata")["chat_id"]
    r = api.patch(f"/api/chats/{chat_id}", params={"session_id": sid}, json={"title": "  My   chat "})
    assert r.json()["title"] == "My chat"
    assert len(api.get("/api/chats/search", params={"session_id": sid, "q": "100%"}).json()) == 1
    assert api.get("/api/chats/search", params={"session_id": sid, "q": "_"}).json() == []
    assert api.delete(f"/api/chats/{chat_id}", params={"session_id": sid}).status_code == 200
    assert api.get("/api/chats", params={"session_id": sid}).json() == []


def test_unknown_model_preference_falls_back_to_auto(api, fake_llm):
    events, _ = stream_chat(api, session_id=new_session(), message="hi", model_preference="groq/qwen/qwen3.6-27b")
    assert event(events, "route") is not None   # router ran → auto mode


def test_manual_model_is_used(api, fake_llm):
    events, _ = stream_chat(api, session_id=new_session(), message="hi", model_preference=config.FAST)
    meta = event(events, "metadata")
    assert meta["route_category"] == "manual" and meta["models_used"] == [config.FAST]


def test_non_streaming_endpoint(api, fake_llm):
    r = api.post("/api/chat", json={"session_id": new_session(), "message": "hello"})
    assert r.status_code == 200 and r.json()["response"].startswith("answer")


def test_input_validation(api):
    assert api.post("/api/chat/stream", json={"session_id": "nope", "message": "x"}).status_code == 400
    assert api.post("/api/chat/stream", json={"session_id": new_session(), "message": "   "}).status_code == 400
    too_long = "x" * (config.MAX_MESSAGE_CHARS + 1)
    assert api.post("/api/chat/stream", json={"session_id": new_session(), "message": too_long}).status_code == 422


def test_model_status_and_models(api):
    models = api.get("/api/models").json()["models"]
    assert [m["value"] for m in models] == config.ALL_MODELS
    status = api.get("/api/model-status").json()["models"]
    assert {s["model"] for s in status} == set(config.ALL_MODELS)


# ── Cancellation ─────────────────────────────────────────────────────────────

def test_stopping_a_stream_keeps_the_partial_answer(fake_llm, db_path):
    from backend.database.connection import initialize_database
    from backend.pipeline import orchestrator

    fake_llm.answer = lambda call: "a long answer " * 20

    async def scenario():
        await initialize_database()
        sid = new_session()
        gen = orchestrator.Orchestrator().stream(session_id=sid, chat_id=None, message="hello")
        chat_id = None
        async for ev in gen:
            if ev["type"] == "start":
                chat_id = ev["chat_id"]
            if ev["type"] == "token":
                break
        await gen.aclose()
        await asyncio.gather(*list(orchestrator._background))
        return await queries.get_chat_messages(chat_id)

    msgs = asyncio.run(scenario())
    assert [m.role for m in msgs] == ["user", "assistant"]
    assert msgs[-1].content.startswith("a lon")


def test_when_every_model_is_cooling_the_soonest_is_tried_not_the_dead_one(api, fake_llm):
    import time
    from backend.llm import client as llm_client
    now = time.monotonic()
    for m in config.ALL_MODELS:
        llm_client.get_health(m).cooldown_until = now + 600      # all "dead"…
    llm_client.get_health(config.FAST).cooldown_until = now + 0.3  # …except one that frees up soon
    events, text = stream_chat(api, session_id=new_session(), message="explain photosynthesis")
    assert event(events, "error") is None
    assert event(events, "metadata")["models_used"] == [config.FAST]


def test_long_chats_get_summarized_and_keep_recent_turns(fake_llm, db_path):
    from backend.database.connection import initialize_database
    from backend.pipeline.memory import MemoryInjector

    fake_llm.answer = lambda call: "SUMMARY: user is learning graphs"

    async def scenario():
        await initialize_database()
        chat_id = str(uuid.uuid4())
        await queries.save_chat(chat_id, new_session(), "long")
        for i in range(12):
            await queries.save_message(str(uuid.uuid4()), chat_id, "user" if i % 2 == 0 else "assistant",
                                       f"message {i} " + "word " * 600, token_count=700)
        memory = MemoryInjector()
        await memory.maybe_summarize(chat_id)
        return await memory.load_history(chat_id)

    history = asyncio.run(scenario())
    assert history[0]["role"] == "system" and "SUMMARY" in history[0]["content"]
    turns = history[1:]
    assert turns and turns[0]["role"] == "user" and turns[-1]["role"] == "assistant"
    assert turns[-1]["content"].startswith("message 11")
