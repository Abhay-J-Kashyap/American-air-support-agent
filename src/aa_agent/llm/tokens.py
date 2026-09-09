"""Cheap, tokenizer-agnostic token estimation.

Llama, GPT-OSS and Mistral use different tokenizers, so an exact count would
mean shipping three vocabularies to save a few percent of accuracy on a
budgeting heuristic. We estimate, then reconcile against the `usage` block the
API returns (RateLimiter.settle). Documented as decision #7.

The 3.6 chars/token divisor is deliberately conservative for this corpus:
tweets are short, hashtag- and handle-dense, and emoji-bearing, all of which
tokenize worse than prose. Over-estimating costs a little throughput;
under-estimating costs 429s and a dead overnight run.
"""

from __future__ import annotations

CHARS_PER_TOKEN = 3.6
PER_MESSAGE_OVERHEAD = 4


def estimate_text_tokens(text: str) -> int:
    return int(len(text) / CHARS_PER_TOKEN) + 1


def estimate_prompt_tokens(messages: list[dict[str, str]]) -> int:
    total = 0
    for m in messages:
        total += estimate_text_tokens(m.get("content", "")) + PER_MESSAGE_OVERHEAD
    return total


def estimate_call_tokens(messages: list[dict[str, str]], max_tokens: int) -> int:
    """Worst-case budget for one call: prompt plus the full completion."""
    return estimate_prompt_tokens(messages) + max_tokens


class ContextBudgetExceeded(ValueError):
    """Prompt would exceed the provider's context ceiling.

    Raised loudly and never silently truncated. A truncated judge prompt that
    still returns a well-formed score is the worst failure mode available to
    this project: it corrupts every downstream metric without erroring.
    """


def assert_fits(messages: list[dict[str, str]], max_tokens: int, ceiling: int, where: str) -> int:
    est = estimate_call_tokens(messages, max_tokens)
    if est > ceiling:
        raise ContextBudgetExceeded(
            f"{where}: estimated {est} tokens exceeds context ceiling {ceiling}. "
            "Shorten the prompt or route to a provider with a larger window."
        )
    return est
