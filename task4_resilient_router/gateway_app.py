"""
Task 4: Complete Resilient LLM Gateway Application
Combines SQLite on-disk sliding window rate limiting with model fallback routing.
"""

import sys
import uuid
import logging
from contextlib import asynccontextmanager
from typing import Optional
from fastapi import FastAPI, Request, Response
from fastapi.responses import JSONResponse

from task4_resilient_router.rate_limiter import (
    TokenSlidingWindowRateLimiter,
    RateLimitExceeded,
)
from task4_resilient_router.router import ModelFallbackRouter

logger = logging.getLogger("gateway_app")
handler = logging.StreamHandler(sys.stderr)
handler.setFormatter(logging.Formatter("[%(asctime)s] [GATEWAY-APP] %(message)s"))
logger.addHandler(handler)
logger.setLevel(logging.INFO)


def create_resilient_gateway(
    db_path: str = "data/rate_limit.db",
    primary_url: str = "http://127.0.0.1:8003/v1/chat/completions",
    backup_url: str = "http://127.0.0.1:8004/v1/chat/completions",
    rate_limiter: Optional[TokenSlidingWindowRateLimiter] = None,
    router: Optional[ModelFallbackRouter] = None,
) -> FastAPI:
    """
    Factory creating fully configured resilient LLM Gateway FastAPI instance.
    """
    limiter = rate_limiter or TokenSlidingWindowRateLimiter(db_path=db_path)
    model_router = router or ModelFallbackRouter(primary_url=primary_url, backup_url=backup_url)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        await limiter.initialize()
        yield

    app = FastAPI(title="Resilient LLM Gateway", lifespan=lifespan)

    @app.post("/v1/chat/completions")
    async def handle_completion(request: Request):
        req_id = request.headers.get("X-Request-ID") or f"gw-{uuid.uuid4().hex[:10]}"

        # 1. Tenant Authentication & Identification
        tenant_id = request.headers.get("X-Tenant-Key")
        if not tenant_id:
            auth_header = request.headers.get("Authorization", "")
            if auth_header.startswith("Bearer "):
                tenant_id = auth_header[7:].strip()

        if not tenant_id:
            tenant_id = "default-tenant"

        try:
            body = await request.json()
        except Exception:
            return JSONResponse(
                status_code=400,
                content={
                    "error": {
                        "code": "MALFORMED_JSON_REQUEST",
                        "message": "Request body could not be parsed as valid JSON.",
                        "request_id": req_id,
                    }
                },
            )

        # 2. Estimate requested tokens
        # Typically prompt tokens + max_tokens requested
        prompt_text = body.get("prompt") or str(body.get("messages", ""))
        estimated_prompt_tokens = max(1, len(prompt_text) // 4)
        max_tokens = body.get("max_tokens", 500)
        requested_tokens = estimated_prompt_tokens + max_tokens

        # 3. Token-Aware Sliding Window Rate Limiting (On-Disk SQLite)
        try:
            await limiter.acquire(tenant_id, requested_tokens)
        except RateLimitExceeded as rle:
            logger.warning(f"[{req_id}] Rate limit rejected for tenant '{tenant_id}': {rle}")
            return JSONResponse(
                status_code=429,
                headers={"Retry-After": str(int(rle.retry_after))},
                content={
                    "error": {
                        "code": "TENANT_RATE_LIMIT_EXCEEDED",
                        "message": (
                            f"Token rate limit exceeded for tenant '{tenant_id}'. "
                            f"Current usage: {rle.current_tokens}/{rle.limit} tokens. "
                            f"Requested: {rle.requested_tokens} tokens."
                        ),
                        "request_id": req_id,
                        "status": 429,
                        "retry_after_seconds": rle.retry_after,
                    }
                },
            )

        # 4. Route Completion with Fallback & Timeout Handling
        result = await model_router.route_completion(body, request_id=req_id)

        if result.success and result.data is not None:
            return JSONResponse(
                status_code=200,
                headers={
                    "X-Request-ID": req_id,
                    "X-Provider-Used": result.provider_used or "unknown",
                    "X-Failover-Occurred": str(result.failover_occurred).lower(),
                    "X-Latency-Ms": str(result.latency_ms),
                },
                content=result.data,
            )
        else:
            # Standardized sanitized error payload
            status_code = result.error_payload.get("error", {}).get("status", 503) if result.error_payload else 503
            return JSONResponse(
                status_code=status_code,
                headers={"X-Request-ID": req_id},
                content=result.error_payload or {
                    "error": {
                        "code": "GATEWAY_SERVICE_UNAVAILABLE",
                        "message": "Gateway was unable to process completion across available providers.",
                        "request_id": req_id,
                    }
                },
            )

    return app
