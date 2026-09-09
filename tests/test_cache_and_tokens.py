from __future__ import annotations

import pytest

from aa_agent.llm.cache import CacheRecord, JsonlCache, cache_key
from aa_agent.llm.tokens import ContextBudgetExceeded, assert_fits, estimate_prompt_tokens

MSGS = [{"role": "user", "content": "my bag never arrived at DFW"}]


def _rec(key: str, model: str = "gpt-oss-120b") -> CacheRecord:
    return CacheRecord(
        key=key,
        provider="cerebras",
        model=model,
        text="ok",
        prompt_tokens=10,
        completion_tokens=2,
        created_at="2026-09-09T00:00:00+00:00",
        latency_ms=120,
    )


def test_cache_key_is_stable():
    a = cache_key("groq", "m", MSGS, 0.0, 100)
    b = cache_key("groq", "m", MSGS, 0.0, 100)
    assert a == b


@pytest.mark.parametrize(
    "kwargs",
    [
        {"provider": "cerebras"},
        {"model": "other"},
        {"temperature": 0.7},
        {"max_tokens": 200},
    ],
)
def test_cache_key_varies_with_every_call_parameter(kwargs):
    base = dict(provider="groq", model="m", messages=MSGS, temperature=0.0, max_tokens=100)
    assert cache_key(**base) != cache_key(**{**base, **kwargs})


def test_roundtrip_and_reload(tmp_path):
    path = tmp_path / "cache.jsonl"
    c = JsonlCache(path)
    c.put(_rec("k1"))
    assert c.get("k1").text == "ok"

    reloaded = JsonlCache(path)
    assert len(reloaded) == 1
    assert reloaded.get("k1").model == "gpt-oss-120b"


def test_torn_final_line_is_skipped(tmp_path):
    """A run killed mid-write must not poison the cache on reload."""
    path = tmp_path / "cache.jsonl"
    JsonlCache(path).put(_rec("k1"))
    with path.open("a") as fh:
        fh.write('{"key": "k2", "provi')
    reloaded = JsonlCache(path)
    assert len(reloaded) == 1
    assert reloaded.get("k1") is not None


def test_read_only_cache_does_not_write(tmp_path):
    path = tmp_path / "cache.jsonl"
    c = JsonlCache(path, read_only=True)
    c.put(_rec("k1"))
    assert c.get("k1") is not None
    assert not path.exists()


def test_stats_group_by_resolved_model(tmp_path):
    c = JsonlCache(tmp_path / "c.jsonl")
    c.put(_rec("k1", model="gpt-oss-120b"))
    c.put(_rec("k2", model="gpt-oss-120b"))
    c.put(_rec("k3", model="mistral-small-2506"))
    assert c.stats() == {"cerebras/gpt-oss-120b": 2, "cerebras/mistral-small-2506": 1}


def test_estimate_scales_with_length():
    short = estimate_prompt_tokens([{"role": "user", "content": "hi"}])
    long = estimate_prompt_tokens([{"role": "user", "content": "hi" * 500}])
    assert long > short * 10


def test_assert_fits_passes_under_ceiling():
    assert assert_fits(MSGS, max_tokens=600, ceiling=8192, where="judge_a") > 0


def test_assert_fits_raises_rather_than_truncating():
    """Cerebras free tier caps context at 8K.

    A silently truncated judge prompt still returns a well-formed score, which
    corrupts every downstream metric without erroring. Fail loudly instead.
    """
    huge = [{"role": "user", "content": "x" * 100_000}]
    with pytest.raises(ContextBudgetExceeded):
        assert_fits(huge, max_tokens=600, ceiling=8192, where="judge_a")
