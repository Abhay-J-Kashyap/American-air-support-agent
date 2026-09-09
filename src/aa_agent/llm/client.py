"""One client, four providers.

Groq, Cerebras, Mistral and OpenRouter all expose an OpenAI-compatible
/chat/completions endpoint, so the entire provider abstraction is a base_url
and an API key. We call it with httpx rather than a vendor SDK: fewer
dependencies, no hidden retry behaviour, and it is short enough to read aloud
in a code review.

Callers ask for a ROLE ("drafter", "judge_a"), never a model. Swapping
providers is a config edit.
"""

from __future__ import annotations

import time
from datetime import datetime, timezone

import httpx

from aa_agent.config import Config, ModelRef, RoleCfg
from aa_agent.llm.cache import CacheRecord, JsonlCache, cache_key
from aa_agent.llm.ratelimit import DailyQuotaExhausted, RateLimiter
from aa_agent.llm.tokens import ContextBudgetExceeded, assert_fits

RETRY_STATUS = {408, 409, 429, 500, 502, 503, 504}


class LLMError(RuntimeError):
    pass


class CacheMiss(LLMError):
    """Offline mode was requested but this call is not in the cache."""


class LLMResponse:
    def __init__(self, text: str, provider: str, model: str, cached: bool) -> None:
        self.text = text
        self.provider = provider
        self.model = model
        self.cached = cached

    def __repr__(self) -> str:
        tag = "cache" if self.cached else "live"
        return f"<LLMResponse {self.provider}/{self.model} [{tag}] {len(self.text)} chars>"


class LLMClient:
    def __init__(
        self,
        cfg: Config,
        offline: bool = False,
        max_retries: int = 5,
        timeout: float = 120.0,
    ) -> None:
        self.cfg = cfg
        self.offline = offline
        self.max_retries = max_retries
        self.cache = JsonlCache(cfg.project.cache_path, read_only=offline)
        self._limiters: dict[str, RateLimiter] = {
            name: RateLimiter(rpm=p.rpm, tpm=p.tpm, rpd=p.rpd) for name, p in cfg.providers.items()
        }
        self._http = httpx.Client(timeout=timeout)

    def close(self) -> None:
        self._http.close()

    def __enter__(self) -> LLMClient:
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    # ------------------------------------------------------------------ #

    def complete(
        self,
        role: str,
        messages: list[dict[str, str]],
        json_schema: dict | None = None,
    ) -> LLMResponse:
        """Run one completion for `role`, walking its fallback chain."""
        rcfg = self.cfg.role(role)
        errors: list[str] = []

        for ref in rcfg.chain:
            provider = self.cfg.provider(ref.provider)
            if not provider.enabled:
                continue

            resp_fmt = {"type": "json_object"} if json_schema else None
            key = cache_key(
                ref.provider, ref.model, messages, rcfg.temperature, rcfg.max_tokens, resp_fmt
            )
            hit = self.cache.get(key)
            if hit is not None:
                return LLMResponse(hit.text, hit.provider, hit.model, cached=True)

            if self.offline:
                errors.append(f"{ref}: cache miss in offline mode")
                continue

            if provider.api_key is None:
                errors.append(f"{ref}: {provider.api_key_env} not set")
                continue

            try:
                return self._call_live(ref, rcfg, messages, resp_fmt, key)
            except ContextBudgetExceeded:
                raise  # a prompt bug, not a provider problem -- never fall through
            except (DailyQuotaExhausted, LLMError) as exc:
                errors.append(f"{ref}: {exc}")
                continue

        if self.offline:
            raise CacheMiss(
                f"role={role!r} not cached. Run `make eval-live` to regenerate. "
                + "; ".join(errors)
            )
        raise LLMError(f"all providers failed for role={role!r}: " + "; ".join(errors))

    # ------------------------------------------------------------------ #

    def _call_live(
        self,
        ref: ModelRef,
        rcfg: RoleCfg,
        messages: list[dict[str, str]],
        resp_fmt: dict | None,
        key: str,
    ) -> LLMResponse:
        provider = self.cfg.provider(ref.provider)
        limiter = self._limiters[ref.provider]

        est = assert_fits(messages, rcfg.max_tokens, provider.max_context_tokens, where=str(ref))

        body: dict = {
            "model": ref.model,
            "messages": messages,
            "temperature": rcfg.temperature,
            "max_tokens": rcfg.max_tokens,
        }
        if resp_fmt:
            body["response_format"] = resp_fmt

        headers = {"Authorization": f"Bearer {provider.api_key}"}
        url = provider.base_url.rstrip("/") + "/chat/completions"

        last: str = "no attempt made"
        for attempt in range(self.max_retries):
            limiter.acquire(est)
            started = time.monotonic()
            try:
                r = self._http.post(url, json=body, headers=headers)
            except httpx.HTTPError as exc:
                last = f"transport error: {exc}"
                time.sleep(min(2**attempt, 30))
                continue

            if r.status_code in RETRY_STATUS:
                # Honour Retry-After when the provider sends one; free tiers
                # usually do, and guessing is worse than being told.
                delay = _retry_after(r) or min(2**attempt, 60)
                last = f"HTTP {r.status_code}: {r.text[:200]}"
                time.sleep(delay)
                continue
            if r.status_code >= 400:
                raise LLMError(f"HTTP {r.status_code}: {r.text[:300]}")

            payload = r.json()
            latency_ms = int((time.monotonic() - started) * 1000)
            try:
                text = payload["choices"][0]["message"]["content"] or ""
            except (KeyError, IndexError) as exc:
                raise LLMError(f"malformed response from {ref}: {exc}") from exc

            usage = payload.get("usage") or {}
            prompt_tokens = int(usage.get("prompt_tokens", est))
            completion_tokens = int(usage.get("completion_tokens", 0))
            limiter.settle(est, prompt_tokens + completion_tokens)

            rec = CacheRecord(
                key=key,
                provider=ref.provider,
                # Prefer the id the API echoes back: it pins the exact
                # snapshot, which config aliases like "-latest" do not.
                model=str(payload.get("model") or ref.model),
                text=text,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                created_at=datetime.now(timezone.utc).isoformat(),
                latency_ms=latency_ms,
            )
            self.cache.put(rec)
            return LLMResponse(text, rec.provider, rec.model, cached=False)

        raise LLMError(f"exhausted {self.max_retries} retries: {last}")


def _retry_after(r: httpx.Response) -> float | None:
    raw = r.headers.get("retry-after")
    if not raw:
        return None
    try:
        return float(raw)
    except ValueError:
        return None
