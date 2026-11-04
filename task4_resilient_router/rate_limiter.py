"""
Task 4: Token-Aware Sliding Window Rate Limiter
Persists token usage transactions in an on-disk SQLite database.
Enforces per-tenant quotas (e.g. 50,000 tokens/minute).
Handles accurate state eviction and high async concurrency via SQLite WAL mode.
"""

import os
import time
import asyncio
import logging
import sys
from typing import Optional, Tuple
import aiosqlite

logger = logging.getLogger("rate_limiter")
handler = logging.StreamHandler(sys.stderr)
handler.setFormatter(logging.Formatter("[%(asctime)s] [RATE-LIMITER] %(message)s"))
logger.addHandler(handler)
logger.setLevel(logging.INFO)


class RateLimitExceeded(Exception):
    """Exception raised when tenant exceeds sliding window token limit."""

    def __init__(self, tenant_id: str, requested_tokens: int, current_tokens: int, limit: int, retry_after: float):
        self.tenant_id = tenant_id
        self.requested_tokens = requested_tokens
        self.current_tokens = current_tokens
        self.limit = limit
        self.retry_after = round(retry_after, 2)
        super().__init__(
            f"Rate limit exceeded for tenant '{tenant_id}'. "
            f"Current window usage: {current_tokens}/{limit} tokens. "
            f"Requested: {requested_tokens}. Retry after {self.retry_after}s."
        )


class TokenSlidingWindowRateLimiter:
    """
    On-disk SQLite sliding window token rate limiter.
    """

    def __init__(
        self,
        db_path: str = "data/rate_limit.db",
        limit_tokens: int = 50000,
        window_seconds: float = 60.0,
    ):
        self.db_path = db_path
        self.limit_tokens = limit_tokens
        self.window_seconds = window_seconds
        self._initialized = False
        self._lock = asyncio.Lock()

    async def initialize(self):
        """Ensures database directory exists and creates schema with WAL mode."""
        if self._initialized:
            return

        db_dir = os.path.dirname(self.db_path)
        if db_dir:
            os.makedirs(db_dir, exist_ok=True)

        async with aiosqlite.connect(self.db_path) as db:
            # WAL mode enables concurrent readers and writers without database locking
            await db.execute("PRAGMA journal_mode = WAL;")
            await db.execute("PRAGMA synchronous = NORMAL;")
            await db.execute("PRAGMA busy_timeout = 5000;")

            # Usage ledger table
            await db.execute(
                """
                CREATE TABLE IF NOT EXISTS token_usage (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    tenant_id TEXT NOT NULL,
                    timestamp REAL NOT NULL,
                    tokens INTEGER NOT NULL
                );
                """
            )
            # Index for fast range queries on (tenant_id, timestamp)
            await db.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_tenant_timestamp 
                ON token_usage (tenant_id, timestamp);
                """
            )
            await db.commit()

        self._initialized = True
        logger.info(f"Initialized SQLite rate limiter on-disk database at '{self.db_path}' (WAL mode).")

    async def acquire(self, tenant_id: str, tokens: int) -> Tuple[bool, int, float]:
        """
        Attempts to acquire 'tokens' for 'tenant_id'.
        Returns (success: bool, current_usage: int, retry_after: float).
        If tokens would exceed limit, raises RateLimitExceeded.
        """
        if not self._initialized:
            await self.initialize()

        now = time.time()
        window_start = now - self.window_seconds

        async with aiosqlite.connect(self.db_path) as db:
            await db.execute("PRAGMA busy_timeout = 5000;")

            # 1. State Eviction: Purge old records older than sliding window across all tenants
            await db.execute(
                "DELETE FROM token_usage WHERE timestamp < ?;",
                (window_start,),
            )

            # 2. Query current token sum for this tenant within active window
            async with db.execute(
                "SELECT COALESCE(SUM(tokens), 0) FROM token_usage WHERE tenant_id = ? AND timestamp >= ?;",
                (tenant_id, window_start),
            ) as cursor:
                row = await cursor.fetchone()
                current_usage = row[0] if row else 0

            # 3. Check quota
            if current_usage + tokens > self.limit_tokens:
                # Calculate retry-after based on oldest entry in window
                async with db.execute(
                    "SELECT MIN(timestamp) FROM token_usage WHERE tenant_id = ? AND timestamp >= ?;",
                    (tenant_id, window_start),
                ) as cursor:
                    oldest_row = await cursor.fetchone()
                    oldest_ts = oldest_row[0] if oldest_row and oldest_row[0] is not None else now
                    retry_after = max(1.0, (oldest_ts + self.window_seconds) - now)

                logger.warning(
                    f"Rate limit exceeded for tenant '{tenant_id}': "
                    f"{current_usage + tokens} > {self.limit_tokens} tokens. Retry-After: {retry_after:.1f}s"
                )
                raise RateLimitExceeded(
                    tenant_id=tenant_id,
                    requested_tokens=tokens,
                    current_tokens=current_usage,
                    limit=self.limit_tokens,
                    retry_after=retry_after,
                )

            # 4. Quota approved: Record usage event
            await db.execute(
                "INSERT INTO token_usage (tenant_id, timestamp, tokens) VALUES (?, ?, ?);",
                (tenant_id, now, tokens),
            )
            await db.commit()

            new_total = current_usage + tokens
            logger.info(f"Tenant '{tenant_id}' acquired {tokens} tokens (New window total: {new_total}/{self.limit_tokens})")
            return True, new_total, 0.0

    async def get_current_usage(self, tenant_id: str) -> int:
        """Returns the current token consumption for a tenant in the active sliding window."""
        if not self._initialized:
            await self.initialize()

        window_start = time.time() - self.window_seconds
        async with aiosqlite.connect(self.db_path) as db:
            async with db.execute(
                "SELECT COALESCE(SUM(tokens), 0) FROM token_usage WHERE tenant_id = ? AND timestamp >= ?;",
                (tenant_id, window_start),
            ) as cursor:
                row = await cursor.fetchone()
                return row[0] if row else 0

    async def reset(self):
        """Clears all records for testing."""
        if not self._initialized:
            await self.initialize()
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute("DELETE FROM token_usage;")
            await db.commit()
