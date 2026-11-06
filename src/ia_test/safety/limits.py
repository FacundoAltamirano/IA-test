import asyncio
from collections import deque
from datetime import datetime, timedelta, timezone


class RateLimitExceeded(Exception):
    pass


class RateLimiter:
    """Thread-safe async rate limiter tracking calls per hour and per day."""

    def __init__(self, max_per_hour: int, max_per_day: int) -> None:
        self._max_per_hour = max_per_hour
        self._max_per_day = max_per_day
        self._calls: deque[datetime] = deque()
        self._lock = asyncio.Lock()

    async def check_and_record(self) -> None:
        async with self._lock:
            now = datetime.now(timezone.utc)
            self._evict_old(now)

            hourly = sum(1 for t in self._calls if (now - t) < timedelta(hours=1))
            if hourly >= self._max_per_hour:
                raise RateLimitExceeded(
                    f"Hourly API limit of {self._max_per_hour} calls reached"
                )

            if len(self._calls) >= self._max_per_day:
                raise RateLimitExceeded(
                    f"Daily API limit of {self._max_per_day} calls reached"
                )

            self._calls.append(now)

    def _evict_old(self, now: datetime) -> None:
        cutoff = now - timedelta(days=1)
        while self._calls and self._calls[0] < cutoff:
            self._calls.popleft()

    @property
    def calls_last_hour(self) -> int:
        now = datetime.now(timezone.utc)
        return sum(1 for t in self._calls if (now - t) < timedelta(hours=1))

    @property
    def calls_last_day(self) -> int:
        return len(self._calls)
