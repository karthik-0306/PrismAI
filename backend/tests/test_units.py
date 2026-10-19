"""Unit tests for pure helpers: think-tag filtering, history cleaning, router parsing."""

from backend.database.models import Message
from backend.llm.client import _ThinkFilter, _cooldown_for, strip_think_tags
from backend.pipeline.memory import MemoryInjector, _clean_turns
from backend.pipeline.router import SmartRouter
from backend.tests.conftest import denied, rate_limit


def test_think_filter_handles_tags_split_across_chunks():
    f = _ThinkFilter()
    chunks = ["Hel", "lo <thi", "nk>secret reasoning</th", "ink> world", "!"]
    out = "".join(f.feed(c) for c in chunks) + f.flush()
    assert out == "Hello world!"


def test_think_filter_drops_unterminated_block():
    f = _ThinkFilter()
    out = f.feed("Answer. <think>never closed") + f.flush()
    assert out == "Answer. "


def test_strip_think_tags_unterminated():
    assert strip_think_tags("<think>a</think>Hi <think>dangling") == "Hi"


def test_cooldowns():
    assert 2.5 < _cooldown_for(rate_limit()) < 4
    assert _cooldown_for(denied()) >= 600


def _msg(role, content):
    return Message(message_id=content, chat_id="c", role=role, content=content, model_used=None,
                   route_category=None, token_count=1, is_summarized=False, created_at="")


def test_clean_turns_drops_orphans_and_empties():
    turns = [_msg("user", "q1"), _msg("assistant", "a1"),
             _msg("user", "orphan"), _msg("user", "q2"), _msg("assistant", ""), _msg("assistant", "a2"),
             _msg("user", "trailing")]
    assert [(m.role, m.content) for m in _clean_turns(turns)] == [
        ("user", "q1"), ("assistant", "a1"), ("user", "q2"), ("assistant", "a2")]


def test_build_prompt_folds_summary_into_system():
    history = [{"role": "system", "content": "SUMMARY"}, {"role": "user", "content": "q"},
               {"role": "assistant", "content": "a"}]
    prompt = MemoryInjector.build_prompt(history, "now")
    assert [m["role"] for m in prompt] == ["system", "user", "assistant", "user"]
    assert "SUMMARY" in prompt[0]["content"] and "{today}" not in prompt[0]["content"]


def test_router_parsing_variants():
    r = SmartRouter()
    assert r._parse('```json\n{"tasks":[{"category":"DSA","query":"x"}]}\n```', "orig") == \
        [{"category": "dsa", "sub_query": "x"}]
    assert r._parse('[{"category":"weird","sub_query":"y"}]', "orig") == [{"category": "general", "sub_query": "y"}]
    assert r._parse("garbage", "orig") == [{"category": "general", "sub_query": "orig"}]
    # evaluate payloads are never split
    assert r._parse('{"tasks":[{"category":"evaluate","query":"a"},{"category":"dsa","query":"b"}]}', "orig") == \
        [{"category": "evaluate", "sub_query": "orig"}]
    # same-category tasks merge
    merged = r._parse('{"tasks":[{"category":"math","query":"1+1"},{"category":"math","query":"2+2"}]}', "o")
    assert len(merged) == 1 and "1+1" in merged[0]["sub_query"] and "2+2" in merged[0]["sub_query"]


def test_fit_to_budget_shrinks_reservation():
    from backend.llm.client import _fit_to_budget
    exc = Exception("Limit 8000, Used 3000, Requested 5096. Please try again in 9s")
    # prompt = 5096 - 4096 = 1000 → remaining for output = 8000 - 3000 - 1000 - 64
    assert _fit_to_budget(exc, 4096) == 3936
    assert _fit_to_budget(Exception("Limit 8000, Used 7900, Requested 5096"), 4096) is None
    assert _fit_to_budget(Exception("no numbers here"), 4096) is None


def test_retry_after_parsing_handles_minutes():
    from backend.llm.client import _retry_after_seconds
    assert _retry_after_seconds("Please try again in 1m3.4s.") == 63.4
    assert _retry_after_seconds("Please try again in 750ms") == 0.75
    assert _retry_after_seconds("Please try again in 12m0s") == 720
    assert _retry_after_seconds("Please try again in 2.5s") == 2.5
    assert _retry_after_seconds("no hint") is None


def test_rewriter_guard_catches_dropped_negations_numbers_and_code():
    from backend.pipeline.rewriter import _lost_critical_tokens
    assert _lost_critical_tokens("sort it without the built-in sort", "sort it using built-in sort") == ["a negation"]
    assert _lost_critical_tokens("give me 5 examples from 2024", "give me examples") == ["numbers ['2024', '5']"]
    code = "fix this ```py\nx = 1/0\n``` please"
    assert "a code block" in _lost_critical_tokens(code, "fix this code")
    assert _lost_critical_tokens(code, "Fix: ```py\nx = 1/0\n```") == []


def test_rewriter_accepts_and_rejects_by_similarity(monkeypatch):
    import asyncio
    from types import SimpleNamespace
    from backend.pipeline import rewriter

    long_msg = ("Hey there, could you please write me a Python script that reads a CSV file and then "
                "plots a really nice looking bar chart using the matplotlib library? I would really appreciate it, "
                "that would be super helpful for my project, thanks so much in advance!")
    short = "Write a Python script that reads a CSV file and plots a bar chart with matplotlib."

    async def fake_complete(**kwargs):
        return SimpleNamespace(content=short, model_used="m")
    monkeypatch.setattr(rewriter._llm, "complete", fake_complete)
    monkeypatch.setattr(rewriter.embeddings, "is_ready", lambda: True)
    monkeypatch.setattr(rewriter.embeddings, "similarity_threshold", lambda: 0.9)

    async def sim(score):
        async def _s(a, b):
            return score
        monkeypatch.setattr(rewriter.embeddings, "get_similarity", _s)
        return await rewriter.QueryRewriter().rewrite(long_msg)

    ok = asyncio.run(sim(0.95))
    assert not ok.fallback_used and ok.rewritten_query == short and ok.reduction_pct > 0
    assert asyncio.run(sim(0.85)).fallback_used
    assert asyncio.run(sim(None)).fallback_used   # backend unavailable → never trust the rewrite
