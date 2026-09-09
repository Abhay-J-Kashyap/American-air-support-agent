"""Client-side rate limiting.

Two buckets per provider because free tiers cap both dimensions and the
binding one is usually TOKENS, not requests. Groq's free tier is 30 RPM but
only 6,000 TPM -- a ~1,350-token drafting call means the real ceiling is
about 4 requests/minute, not 30. Budgeting on RPM alone gets you 429s.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field


class Bucket:
    """Continuously-refilling token bucket. Thread-safe."""

    def __init__(self, capacity: float, per_seconds: float, clock=time.monotonic) -> None:
        if capacity <= 0:
            raise ValueError("capacity must be positive")
        self.capacity = float(capacity)
        self.rate = capacity / per_seconds
        self._tokens = float(capacity)
        self._clock = clock
        self._last = clock()
        self._lock = threading.Lock()

    def _refill_locked(self) -> None:
        now = self._clock()
        elapsed = now - self._last
        if elapsed > 0:
            self._tokens = min(self.capacity, self._tokens + elapsed * self.rate)
            self._last = now

    def available(self) -> float:
        with self._lock:
            self._refill_locked()
            return self._tokens

    def time_until(self, amount: float) -> float:
        """Seconds to wait before `amount` tokens would be available."""
        amount = min(float(amount), self.capacity)
        with self._lock:
            self._refill_locked()
            if self._tokens >= amount:
                return 0.0
            return (amount - self._tokens) / self.rate

    def consume(self, amount: float) -> None:
        """Take `amount` tokens, allowing the balance to go negative.

        Negative balances are how we pay back an underestimate: if a call used
        more tokens than we predicted, the next caller simply waits longer.
        """
        with self._lock:
            self._refill_locked()
            self._tokens -= min(float(amount), self.capacity)


@dataclass
class RateLimiter:
    """Combined request-rate and token-rate limiter for one provider."""

    rpm: int
    tpm: int
    rpd: int | None = None
    _requests: Bucket = field(init=False)
    _tokens: Bucket = field(init=False)
    _day_used: int = field(default=0, init=False)
    _day_lock: threading.Lock = field(default_factory=threading.Lock, init=False)

    def __post_init__(self) -> None:
        self._requests = Bucket(self.rpm, 60.0)
        self._tokens = Bucket(self.tpm, 60.0)

    def wait_time(self, est_tokens: int) -> float:
        return max(self._requests.time_until(1), self._tokens.time_until(est_tokens))

    def acquire(self, est_tokens: int, sleep=time.sleep) -> float:
        """Block until a call of ~est_tokens may proceed. Returns seconds slept."""
        if self.rpd is not None:
            with self._day_lock:
                if self._day_used >= self.rpd:
                    raise DailyQuotaExhausted(
                        f"daily request cap of {self.rpd} reached for this provider"
                    )
                self._day_used += 1
        waited = 0.0
        while (delay := self.wait_time(est_tokens)) > 0:
            sleep(delay)
            waited += delay
        self._requests.consume(1)
        self._tokens.consume(est_tokens)
        return waited

    def settle(self, estimated: int, actual: int) -> None:
        """Reconcile an estimate against the provider's reported usage.

        We estimate prompt size with a cheap heuristic (see tokens.py) because
        every provider here uses a different tokenizer. When the response comes
        back with real counts, charge the difference so the drift does not
        compound over a long run.
        """
        delta = actual - estimated
        if delta > 0:
            self._tokens.consume(delta)


class DailyQuotaExhausted(RuntimeError):
    """Raised when a provider's requests-per-day ceiling is hit."""
