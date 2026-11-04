"""
Unit & Integration Test Suite for Task 4: Rate-Limiting & Model Fallback Router
Tests On-Disk SQLite Rate Limiter, Eviction, 3000ms Timeout Fallback, 429 Fallback,
Concurrency, and Standardized Error Sanitization.
"""

import os
import time
import asyncio
import pytest
import aiosqlite
import httpx
from fastapi.testclient import TestClient

from task4_resilient_router.rate_limiter import (
    TokenSlidingWindowRateLimiter,
    RateLimitExceeded,
)
from task4_resilient_router.mock_providers import (
    primary_app,
    backup_app,
    config as mock_config,
)
from task4_resilient_router.router import ModelFallbackRouter
from task4_resilient_router.gateway_app import create_resilient_gateway

TEST_DB_PATH = "data/test_rate_limit.db"


@pytest.fixture(autouse=True)
async def cleanup_db():
    mock_config.reset()
    if os.path.exists(TEST_DB_PATH):
        try:
            os.remove(TEST_DB_PATH)
        except OSError:
            pass
    for ext in ["-wal", "-shm"]:
        if os.path.exists(TEST_DB_PATH + ext):
            try:
                os.remove(TEST_DB_PATH + ext)
            except OSError:
                pass
    yield
    mock_config.reset()
    if os.path.exists(TEST_DB_PATH):
        try:
            os.remove(TEST_DB_PATH)
        except OSError:
            pass
    for ext in ["-wal", "-shm"]:
        if os.path.exists(TEST_DB_PATH + ext):
            try:
                os.remove(TEST_DB_PATH + ext)
            except OSError:
                pass


# ===========================================================================
# 1. On-Disk SQLite & Sliding Window Rate Limiter Tests
# ===========================================================================
@pytest.mark.asyncio
class TestTokenRateLimiter:
    async def test_sqlite_on_disk_and_wal_mode(self):
        """Verifies SQLite database is created on disk with WAL journal mode."""
        limiter = TokenSlidingWindowRateLimiter(db_path=TEST_DB_PATH, limit_tokens=50000)
        await limiter.initialize()

        assert os.path.exists(TEST_DB_PATH), "SQLite database file was not created on disk!"

        async with aiosqlite.connect(TEST_DB_PATH) as db:
            async with db.execute("PRAGMA journal_mode;") as cursor:
                row = await cursor.fetchone()
                assert row[0].lower() == "wal", f"Expected WAL mode, got {row[0]}"

    async def test_token_limit_enforcement(self):
        """
        Enforce max 50,000 tokens/minute per tenant.
        Allows up to limit, rejects calls exceeding limit with Retry-After.
        """
        limiter = TokenSlidingWindowRateLimiter(db_path=TEST_DB_PATH, limit_tokens=50000, window_seconds=60.0)
        await limiter.initialize()

        tenant = "tenant-alpha"

        # 1. Acquire 30,000 tokens -> Success
        success, usage, _ = await limiter.acquire(tenant, 30000)
        assert success is True
        assert usage == 30000

        # 2. Acquire 15,000 tokens -> Success (Total 45,000)
        success, usage, _ = await limiter.acquire(tenant, 15000)
        assert success is True
        assert usage == 45000

        # 3. Request 10,000 tokens -> Exceeds 50,000 -> Raises RateLimitExceeded
        with pytest.raises(RateLimitExceeded) as exc_info:
            await limiter.acquire(tenant, 10000)

        err = exc_info.value
        assert err.tenant_id == tenant
        assert err.current_tokens == 45000
        assert err.requested_tokens == 10000
        assert err.limit == 50000
        assert err.retry_after > 0

    async def test_tenant_isolation(self):
        """Tokens consumed by tenant A must not block tenant B."""
        limiter = TokenSlidingWindowRateLimiter(db_path=TEST_DB_PATH, limit_tokens=50000)
        await limiter.initialize()

        # Tenant A consumes all 50,000 tokens
        await limiter.acquire("tenant-A", 50000)

        # Tenant A is now blocked
        with pytest.raises(RateLimitExceeded):
            await limiter.acquire("tenant-A", 100)

        # Tenant B still has full quota
        success, usage, _ = await limiter.acquire("tenant-B", 25000)
        assert success is True
        assert usage == 25000

    async def test_sliding_window_state_eviction(self):
        """
        Evaluation Criterion: Accurate rate-limiter state eviction.
        Tokens outside the active window are purged and quota is restored.
        """
        # Create limiter with short 1.0 second sliding window for fast eviction testing
        limiter = TokenSlidingWindowRateLimiter(db_path=TEST_DB_PATH, limit_tokens=1000, window_seconds=1.0)
        await limiter.initialize()

        tenant = "tenant-fast-evict"

        # Fill quota
        await limiter.acquire(tenant, 1000)

        # Should now be blocked
        with pytest.raises(RateLimitExceeded):
            await limiter.acquire(tenant, 50)

        # Wait for sliding window to expire (1.1 seconds)
        await asyncio.sleep(1.1)

        # Eviction occurs on next check, quota restored
        success, usage, _ = await limiter.acquire(tenant, 600)
        assert success is True
        assert usage == 600

    async def test_async_concurrency_handling(self):
        """
        Evaluation Criterion: Async concurrency handling without SQLite locking errors.
        Runs 20 concurrent acquisitions across different tenants.
        """
        limiter = TokenSlidingWindowRateLimiter(db_path=TEST_DB_PATH, limit_tokens=100000)
        await limiter.initialize()

        async def worker(tenant_id: str, tokens: int):
            return await limiter.acquire(tenant_id, tokens)

        tasks = [worker(f"tenant-{i % 4}", 500) for i in range(20)]
        results = await asyncio.gather(*tasks)

        assert len(results) == 20
        assert all(res[0] is True for res in results)


# ===========================================================================
# 2. Model Fallback Router Tests (Timeout & 429 Failover)
# ===========================================================================
@pytest.mark.asyncio
class TestModelFallbackRouter:
    async def test_primary_success_path(self):
        """When primary model responds normally, router returns primary response without fallback."""
        primary_transport = httpx.ASGITransport(app=primary_app)
        backup_transport = httpx.ASGITransport(app=backup_app)

        async with httpx.AsyncClient(transport=primary_transport, base_url="http://primary") as p_client:
            async with httpx.AsyncClient(transport=backup_transport, base_url="http://backup") as b_client:
                router = ModelFallbackRouter(
                    primary_url="http://primary/v1/chat/completions",
                    backup_url="http://backup/v1/chat/completions",
                    primary_client=p_client,
                    backup_client=b_client,
                )

                result = await router.route_completion({"prompt": "Hello world"})
                assert result.success is True
                assert result.provider_used == "primary"
                assert result.failover_occurred is False
                assert mock_config.primary_call_count == 1
                assert mock_config.backup_call_count == 0

    async def test_primary_timeout_failover_at_3000ms(self):
        """
        Critical Requirement:
        If primary model times out after 3000ms, automatically failover to secondary backup model.
        """
        mock_config.primary_mode = "timeout"  # Causes primary to take > 3000ms

        primary_transport = httpx.ASGITransport(app=primary_app)
        backup_transport = httpx.ASGITransport(app=backup_app)

        async with httpx.AsyncClient(transport=primary_transport, base_url="http://primary") as p_client:
            async with httpx.AsyncClient(transport=backup_transport, base_url="http://backup") as b_client:
                router = ModelFallbackRouter(
                    primary_url="http://primary/v1/chat/completions",
                    backup_url="http://backup/v1/chat/completions",
                    primary_client=p_client,
                    backup_client=b_client,
                    primary_timeout=3.0,  # Strict 3000ms timeout
                )

                start_t = time.monotonic()
                result = await router.route_completion({"prompt": "Time out test"})
                elapsed = time.monotonic() - start_t

                # Assert fallback occurred
                assert result.success is True
                assert result.provider_used == "secondary_backup"
                assert result.failover_occurred is True
                assert "TIMEOUT" in result.failover_reason
                assert "3000MS" in result.failover_reason

                # Verify secondary response data received
                assert result.data["provider"] == "secondary-backup-model"

                # Verify timeout occurred around ~3.0s
                assert elapsed >= 2.9, f"Elapsed time was too fast: {elapsed:.2f}s"
                assert mock_config.backup_call_count == 1

    async def test_primary_429_failover(self):
        """
        Critical Requirement:
        If primary model returns 429 Too Many Requests, automatically failover to secondary backup model.
        """
        mock_config.primary_mode = "429"  # Primary returns HTTP 429

        primary_transport = httpx.ASGITransport(app=primary_app)
        backup_transport = httpx.ASGITransport(app=backup_app)

        async with httpx.AsyncClient(transport=primary_transport, base_url="http://primary") as p_client:
            async with httpx.AsyncClient(transport=backup_transport, base_url="http://backup") as b_client:
                router = ModelFallbackRouter(
                    primary_url="http://primary/v1/chat/completions",
                    backup_url="http://backup/v1/chat/completions",
                    primary_client=p_client,
                    backup_client=b_client,
                )

                result = await router.route_completion({"prompt": "429 rate limit test"})

                # Failover triggered immediately
                assert result.success is True
                assert result.provider_used == "secondary_backup"
                assert result.failover_occurred is True
                assert result.failover_reason == "PRIMARY_HTTP_429_TOO_MANY_REQUESTS"
                assert result.data["provider"] == "secondary-backup-model"
                assert mock_config.primary_call_count == 1
                assert mock_config.backup_call_count == 1

    async def test_standardized_error_sanitization_when_all_fail(self):
        """
        Evaluation Criterion: Standardized error sanitization without leaking stack traces or internals.
        """
        mock_config.primary_mode = "500"
        mock_config.backup_mode = "500"  # Both fail

        primary_transport = httpx.ASGITransport(app=primary_app)
        backup_transport = httpx.ASGITransport(app=backup_app)

        async with httpx.AsyncClient(transport=primary_transport, base_url="http://primary") as p_client:
            async with httpx.AsyncClient(transport=backup_transport, base_url="http://backup") as b_client:
                router = ModelFallbackRouter(
                    primary_url="http://primary/v1/chat/completions",
                    backup_url="http://backup/v1/chat/completions",
                    primary_client=p_client,
                    backup_client=b_client,
                )

                result = await router.route_completion({"prompt": "Failure test"})

                assert result.success is False
                assert result.error_payload is not None
                err = result.error_payload["error"]

                # Standardized gateway error payload checks
                assert err["code"] in ["GATEWAY_UPSTREAM_FAILURE", "GATEWAY_SERVICE_UNAVAILABLE"]
                assert "request_id" in err
                assert "timestamp" in err

                # Verification of ZERO leak of upstream stack traces
                err_text = str(err)
                assert "Traceback" not in err_text
                assert "crash dump" not in err_text
                assert "internal engine" not in err_text


# ===========================================================================
# 3. Full Gateway Integration Test
# ===========================================================================
@pytest.mark.asyncio
class TestResilientGatewayApp:
    async def test_gateway_rate_limit_and_fallback_e2e(self):
        """Full end-to-end gateway test combining rate limiter and router."""
        limiter = TokenSlidingWindowRateLimiter(db_path=TEST_DB_PATH, limit_tokens=5000)
        await limiter.initialize()

        primary_transport = httpx.ASGITransport(app=primary_app)
        backup_transport = httpx.ASGITransport(app=backup_app)

        p_client = httpx.AsyncClient(transport=primary_transport, base_url="http://primary")
        b_client = httpx.AsyncClient(transport=backup_transport, base_url="http://backup")

        router = ModelFallbackRouter(
            primary_url="http://primary/v1/chat/completions",
            backup_url="http://backup/v1/chat/completions",
            primary_client=p_client,
            backup_client=b_client,
        )

        gateway_app = create_resilient_gateway(
            db_path=TEST_DB_PATH,
            rate_limiter=limiter,
            router=router,
        )

        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=gateway_app), base_url="http://gateway") as client:
            # 1. Normal Request (prompt=40 chars -> ~10 tokens + max_tokens=1000 = 1010 tokens)
            resp1 = await client.post(
                "/v1/chat/completions",
                headers={"X-Tenant-Key": "tenant-corp-1"},
                json={"prompt": "Summarize quarterly report", "max_tokens": 1000},
            )
            assert resp1.status_code == 200
            assert resp1.headers["x-provider-used"] == "primary"
            assert resp1.headers["x-failover-occurred"] == "false"

            # 2. Trigger Primary 429 -> Should failover to secondary backup
            mock_config.primary_mode = "429"
            resp2 = await client.post(
                "/v1/chat/completions",
                headers={"X-Tenant-Key": "tenant-corp-1"},
                json={"prompt": "Generate sales forecast", "max_tokens": 1000},
            )
            assert resp2.status_code == 200
            assert resp2.headers["x-provider-used"] == "secondary_backup"
            assert resp2.headers["x-failover-occurred"] == "true"

            # 3. Exceed Token Rate Limit (Limit is 5000, already used ~2020 tokens, request 4000 tokens)
            resp3 = await client.post(
                "/v1/chat/completions",
                headers={"X-Tenant-Key": "tenant-corp-1"},
                json={"prompt": "Large extraction task", "max_tokens": 4000},
            )
            assert resp3.status_code == 429
            assert "Retry-After" in resp3.headers
            err_data = resp3.json()
            assert err_data["error"]["code"] == "TENANT_RATE_LIMIT_EXCEEDED"

        await p_client.aclose()
        await b_client.aclose()
