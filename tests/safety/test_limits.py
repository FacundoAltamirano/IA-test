import pytest
from ia_test.safety.limits import RateLimiter, RateLimitExceeded


async def test_allows_calls_within_limit():
    limiter = RateLimiter(max_per_hour=10, max_per_day=100)
    for _ in range(5):
        await limiter.check_and_record()
    assert limiter.calls_last_hour == 5


async def test_raises_when_hourly_limit_exceeded():
    limiter = RateLimiter(max_per_hour=3, max_per_day=100)
    for _ in range(3):
        await limiter.check_and_record()

    with pytest.raises(RateLimitExceeded, match="Hourly"):
        await limiter.check_and_record()


async def test_raises_when_daily_limit_exceeded():
    limiter = RateLimiter(max_per_hour=100, max_per_day=3)
    for _ in range(3):
        await limiter.check_and_record()

    with pytest.raises(RateLimitExceeded, match="Daily"):
        await limiter.check_and_record()


async def test_calls_last_day_counts_all_calls():
    limiter = RateLimiter(max_per_hour=100, max_per_day=100)
    for _ in range(7):
        await limiter.check_and_record()
    assert limiter.calls_last_day == 7
