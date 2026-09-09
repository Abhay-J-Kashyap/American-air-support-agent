from __future__ import annotations

from pathlib import Path

import pytest

from aa_agent.config import load_config
from aa_agent.llm.cache import CacheRecord
from aa_agent.llm.client import CacheMiss, LLMClient, LLMError

CONFIG = Path("config/config.yaml")


@pytest.fixture()
def cfg(tmp_path, monkeypatch):
    c = load_config(CONFIG)
    c.project.cache_path = tmp_path / "cache.jsonl"
    for var in ("GROQ_API_KEY", "CEREBRAS_API_KEY", "MISTRAL_API_KEY", "OPENROUTER_API_KEY"):
        monkeypatch.delenv(var, raising=False)
    return c


def test_config_loads_and_names_providers(cfg):
    assert cfg.project.brand == "AmericanAir"
    assert cfg.providers["groq"].name == "groq"
    assert cfg.provider("cerebras").max_context_tokens == 8192


def _family(ref) -> str:
    return ref.model.split("-")[0]


def test_every_judge_differs_in_model_family_from_the_drafter(cfg):
    """Minimum bar, non-negotiable for every judge.

    If a judge shared a model family with the drafter, the self-preference
    measurement would be meaningless regardless of which provider hosts it.
    """
    drafter = cfg.role("drafter").primary
    for judge in ("judge_a", "judge_b"):
        ref = cfg.role(judge).primary
        assert _family(ref) != _family(drafter), f"{judge} shares a model family with drafter"


def test_at_least_one_judge_is_fully_independent_of_the_drafter(cfg):
    """At least one judge must differ in BOTH provider and family.

    Originally both judges met this bar. Cerebras is disabled for this
    account (DECISIONS.md #14), which forced judge_a onto Groq alongside
    the drafter -- family-independent but not provider-independent. judge_b
    (Mistral) is what still gives a clean, fully independent reading; this
    test guards that at least one such judge always exists, so a future
    config edit can't silently lose the last clean signal.
    """
    drafter = cfg.role("drafter").primary
    fully_independent = [
        judge
        for judge in ("judge_a", "judge_b")
        if cfg.role(judge).primary.provider != drafter.provider
        and _family(cfg.role(judge).primary) != _family(drafter)
    ]
    assert fully_independent, "no judge is fully independent (provider + family) of the drafter"


def test_openrouter_is_never_a_primary(cfg):
    """50 requests/day on an unfunded account. Fallback only."""
    for name, rcfg in cfg.roles.items():
        assert rcfg.primary.provider != "openrouter", f"{name} has openrouter as primary"


def test_cache_hit_short_circuits_before_any_network(cfg):
    client = LLMClient(cfg, offline=False)
    msgs = [{"role": "user", "content": "where is my bag"}]
    rcfg = cfg.role("classifier")
    ref = rcfg.primary

    from aa_agent.llm.cache import cache_key

    key = cache_key(ref.provider, ref.model, msgs, rcfg.temperature, rcfg.max_tokens, None)
    client.cache.put(
        CacheRecord(
            key=key,
            provider=ref.provider,
            model=ref.model,
            text='{"intent": "baggage"}',
            prompt_tokens=8,
            completion_tokens=6,
            created_at="2026-09-09T00:00:00+00:00",
            latency_ms=90,
        )
    )

    # No API keys are set, so a live call would fail. A hit must not try.
    resp = client.complete("classifier", msgs)
    assert resp.cached is True
    assert "baggage" in resp.text


def test_offline_mode_raises_cache_miss(cfg):
    client = LLMClient(cfg, offline=True)
    with pytest.raises(CacheMiss):
        client.complete("drafter", [{"role": "user", "content": "uncached"}])


def test_missing_keys_surface_every_provider_in_the_error(cfg):
    client = LLMClient(cfg, offline=False)
    with pytest.raises(LLMError) as exc:
        client.complete("classifier", [{"role": "user", "content": "hi"}])
    msg = str(exc.value)
    # classifier's chain is groq (primary) -> openrouter (fallback).
    # Cerebras is disabled account-wide (DECISIONS.md #14) and skipped
    # silently rather than surfaced as a missing-key error.
    assert "GROQ_API_KEY" in msg and "OPENROUTER_API_KEY" in msg
