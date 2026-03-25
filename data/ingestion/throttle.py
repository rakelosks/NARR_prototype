"""
Async request throttle for external API calls.
Provides concurrency limiting, rate limiting, and retry with backoff.
"""

import asyncio
import logging
import time
from typing import Optional

logger = logging.getLogger(__name__)


class Throttle:
    """
    Async throttle that limits concurrency and enforces a minimum interval
    between requests.  Use as an async context manager around each request.

    Usage:
        throttle = Throttle(max_concurrent=3, min_interval=0.2)

        async with throttle:
            await make_request()
    """

    def __init__(
        self,
        max_concurrent: int = 5,
        min_interval: float = 0.0,
    ):
        """
        Args:
            max_concurrent: Max number of in-flight requests at once.
            min_interval: Minimum seconds between successive requests.
        """
        self._semaphore = asyncio.Semaphore(max_concurrent)
        self._min_interval = min_interval
        self._last_request: float = 0.0
        self._lock = asyncio.Lock()

    async def __aenter__(self):
        await self._semaphore.acquire()
        if self._min_interval > 0:
            async with self._lock:
                now = time.monotonic()
                wait = self._last_request + self._min_interval - now
                if wait > 0:
                    await asyncio.sleep(wait)
                self._last_request = time.monotonic()
        return self

    async def __aexit__(self, *exc):
        self._semaphore.release()


async def retry_with_backoff(
    coro_factory,
    max_retries: int = 3,
    base_delay: float = 1.0,
    retryable: Optional[tuple] = None,
):
    """
    Retry an async call with exponential backoff.

    Args:
        coro_factory: A zero-arg callable that returns a new coroutine each call.
        max_retries: Total attempts (including the first).
        base_delay: Initial backoff delay in seconds (doubles each retry).
        retryable: Tuple of exception types to retry on.
            Defaults to common transient errors.

    Returns:
        The result of the coroutine on success.

    Raises:
        The last exception if all retries are exhausted.
    """
    import httpx

    if retryable is None:
        retryable = (
            httpx.TimeoutException,
            httpx.ConnectError,
            ConnectionError,
            TimeoutError,
        )

    last_exc = None
    for attempt in range(1, max_retries + 1):
        try:
            return await coro_factory()
        except retryable as e:
            last_exc = e
            if attempt == max_retries:
                break
            delay = base_delay * (2 ** (attempt - 1))
            logger.warning(
                f"Retry {attempt}/{max_retries} after {type(e).__name__}: "
                f"{e} — waiting {delay:.1f}s"
            )
            await asyncio.sleep(delay)

    raise last_exc
