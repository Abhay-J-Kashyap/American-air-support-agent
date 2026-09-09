"""Rate limiter tests run on a fake clock -- no sleeping in CI."""

from __future__ import annotations

import pytest

from aa_agent.llm.ratelimit import Bucket, DailyQuotaExhausted, RateLimiter


class FakeClock:
    def __init__(self) -> None:
        self.t = 0.0

    def __call__(self) -> float:
        return self.t

    def advance(self, seconds: float) -> None:
        self.t += seconds


def test_bucket_starts_full():
    clock = FakeClock()
    b = Bucket(60, 60.0, clock)
    assert b.available() == pytest.approx(60)


def test_bucket_refills_at_rate():
    clock = FakeClock()
    b = Bucket(60, 60.0, clock)
    b.consume(60)
    assert b.available() == pytest.approx(0)
    clock.advance(30)
    assert b.available() == pytest.approx(30)


def test_bucket_never_exceeds_capacity():
    clock = FakeClock()
    b = Bucket(10, 60.0, clock)
    clock.advance(10_000)
    assert b.available() == pytest.approx(10)


def test_time_until_reports_wait():
    clock = FakeClock()
    b = Bucket(60, 60.0, clock)
    b.consume(60)
    assert b.time_until(30) == pytest.approx(30.0)


def test_tpm_binds_before_rpm_on_groq_shaped_limits():
    """The whole reason the limiter tracks two dimensions.

    Groq free tier: 30 RPM but only 6,000 TPM. A ~1,350-token drafting call
    means the real ceiling is ~4 requests/minute. Budgeting on RPM alone is
    how you earn 429s at 3am.
    """
    limiter = RateLimiter(rpm=30, tpm=6000)
    slept: list[float] = []

    for _ in range(4):
        limiter.acquire(1350, sleep=slept.append)
    assert sum(slept) == 0, "first four calls fit inside the token budget"

    limiter.acquire(1350, sleep=slept.append)
    assert sum(slept) > 0, "fifth call must wait on TPM, not RPM"


def test_settle_charges_underestimates():
    limiter = RateLimiter(rpm=1000, tpm=6000)
    limiter.acquire(100, sleep=lambda _: None)
    before = limiter._tokens.available()
    limiter.settle(estimated=100, actual=1100)
    assert limiter._tokens.available() == pytest.approx(before - 1000, abs=1)


def test_settle_ignores_overestimates():
    limiter = RateLimiter(rpm=1000, tpm=6000)
    limiter.acquire(1000, sleep=lambda _: None)
    before = limiter._tokens.available()
    limiter.settle(estimated=1000, actual=200)
    assert limiter._tokens.available() == pytest.approx(before, abs=1)


def test_daily_quota_raises():
    limiter = RateLimiter(rpm=1000, tpm=1_000_000, rpd=2)
    limiter.acquire(10, sleep=lambda _: None)
    limiter.acquire(10, sleep=lambda _: None)
    with pytest.raises(DailyQuotaExhausted):
        limiter.acquire(10, sleep=lambda _: None)
