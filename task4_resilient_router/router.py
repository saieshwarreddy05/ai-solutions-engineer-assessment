"""
Task 4: Model Fallback Router
Dispatches completion requests with strict 3000ms timeout and HTTP 429 failover.
Sanitizes all errors into standardized enterprise gateway payloads.
"""

import sys
import time
import uuid
import asyncio
import logging
from typing import Any, Dict, Optional, Tuple
import httpx

logger = logging.getLogger("model_router")
handler = logging.StreamHandler(sys.stderr)
handler.setFormatter(logging.Formatter("[%(asctime)s] [MODEL-ROUTER] %(message)s"))
logger.addHandler(handler)
logger.setLevel(logging.INFO)

PRIMARY_TIMEOUT_SECONDS = 3.0  # 3000ms timeout requirement


class RouterResult:
    """Holds routing execution outcome and metadata."""

    def __init__(
        self,
        success: bool,
        provider_used: Optional[str] = None,
        data: Optional[Dict[str, Any]] = None,
        error_payload: Optional[Dict[str, Any]] = None,
        failover_occurred: bool = False,
        failover_reason: Optional[str] = None,
        latency_ms: float = 0.0,
    ):
        self.success = success
        self.provider_used = provider_used
        self.data = data
        self.error_payload = error_payload
        self.failover_occurred = failover_occurred
        self.failover_reason = failover_reason
        self.latency_ms = round(latency_ms, 2)


class ModelFallbackRouter:
    """
    Resilient router handling primary model execution with automatic failover to backup.
    """

    def __init__(
        self,
        primary_url: str = "http://127.0.0.1:8003/v1/chat/completions",
        backup_url: str = "http://127.0.0.1:8004/v1/chat/completions",
        primary_client: Optional[httpx.AsyncClient] = None,
        backup_client: Optional[httpx.AsyncClient] = None,
        primary_timeout: float = PRIMARY_TIMEOUT_SECONDS,
    ):
        self.primary_url = primary_url
        self.backup_url = backup_url
        self._primary_client = primary_client
        self._backup_client = backup_client
        self.primary_timeout = primary_timeout

    def _sanitize_error(self, code: str, message: str, request_id: str, status_code: int = 503) -> Dict[str, Any]:
        """
        Builds standardized gateway error payload.
        Guarantees that raw upstream stack traces or internal implementation details are never leaked.
        """
        return {
            "error": {
                "code": code,
                "message": message,
                "request_id": request_id,
                "status": status_code,
                "timestamp": round(time.time(), 3),
            }
        }

    async def route_completion(
        self,
        payload: Dict[str, Any],
        request_id: Optional[str] = None,
    ) -> RouterResult:
        """
        Execute completion with 3000ms primary timeout and 429 failover to secondary.
        """
        req_id = request_id or f"req-{uuid.uuid4().hex[:12]}"
        start_time = time.monotonic()

        failover_triggered = False
        failover_reason = None

        logger.info(f"[{req_id}] Routing completion request. Attempting primary provider ({self.primary_url})...")

        # -------------------------------------------------------------------
        # 1. Attempt Primary Provider with 3000ms Strict Timeout
        # -------------------------------------------------------------------
        primary_client = self._primary_client or httpx.AsyncClient(timeout=self.primary_timeout + 1.0)
        should_close_primary = self._primary_client is None

        primary_success = False
        primary_result_data = None

        try:
            # Wrap primary call in asyncio.wait_for to guarantee 3000ms race condition handling
            resp = await asyncio.wait_for(
                primary_client.post(
                    self.primary_url,
                    json=payload,
                    headers={"Content-Type": "application/json", "X-Request-ID": req_id},
                    timeout=self.primary_timeout,
                ),
                timeout=self.primary_timeout,
            )

            if resp.status_code == 200:
                primary_success = True
                primary_result_data = resp.json()
                logger.info(f"[{req_id}] Primary provider responded successfully.")
            elif resp.status_code == 429:
                failover_triggered = True
                failover_reason = "PRIMARY_HTTP_429_TOO_MANY_REQUESTS"
                logger.warning(
                    f"[{req_id}] Primary returned HTTP 429 Too Many Requests. "
                    "Initiating immediate failover to secondary backup provider..."
                )
            else:
                failover_triggered = True
                failover_reason = f"PRIMARY_HTTP_STATUS_{resp.status_code}"
                logger.warning(f"[{req_id}] Primary returned error status {resp.status_code}. Initiating failover...")

        except (asyncio.TimeoutError, httpx.TimeoutException):
            failover_triggered = True
            failover_reason = f"PRIMARY_TIMEOUT_EXCEEDED_{int(self.primary_timeout * 1000)}MS"
            logger.warning(
                f"[{req_id}] Primary model provider timed out after {self.primary_timeout * 1000}ms! "
                "Initiating immediate failover to secondary backup provider..."
            )
        except Exception as exc:
            failover_triggered = True
            failover_reason = f"PRIMARY_CONNECTION_ERROR_{type(exc).__name__}"
            logger.warning(
                f"[{req_id}] Primary provider connection failed: {exc}. "
                "Initiating immediate failover to secondary backup provider..."
            )
        finally:
            if should_close_primary:
                await primary_client.aclose()

        if primary_success and primary_result_data is not None:
            total_latency = (time.monotonic() - start_time) * 1000
            return RouterResult(
                success=True,
                provider_used="primary",
                data=primary_result_data,
                failover_occurred=False,
                latency_ms=total_latency,
            )

        # -------------------------------------------------------------------
        # 2. Failover: Attempt Secondary Backup Provider
        # -------------------------------------------------------------------
        logger.info(f"[{req_id}] Calling secondary backup provider ({self.backup_url}). Reason: {failover_reason}")
        backup_client = self._backup_client or httpx.AsyncClient(timeout=10.0)
        should_close_backup = self._backup_client is None

        try:
            backup_resp = await backup_client.post(
                self.backup_url,
                json=payload,
                headers={"Content-Type": "application/json", "X-Request-ID": req_id},
                timeout=10.0,
            )

            total_latency = (time.monotonic() - start_time) * 1000

            if backup_resp.status_code == 200:
                logger.info(f"[{req_id}] Secondary backup provider succeeded. Failover completed successfully.")
                return RouterResult(
                    success=True,
                    provider_used="secondary_backup",
                    data=backup_resp.json(),
                    failover_occurred=True,
                    failover_reason=failover_reason,
                    latency_ms=total_latency,
                )
            else:
                logger.error(
                    f"[{req_id}] Secondary backup provider also failed with status {backup_resp.status_code}."
                )
                error_payload = self._sanitize_error(
                    code="GATEWAY_UPSTREAM_FAILURE",
                    message="Both primary and secondary backup model providers failed to satisfy the request.",
                    request_id=req_id,
                    status_code=502,
                )
                return RouterResult(
                    success=False,
                    error_payload=error_payload,
                    failover_occurred=True,
                    failover_reason=failover_reason,
                    latency_ms=total_latency,
                )

        except Exception as backup_exc:
            total_latency = (time.monotonic() - start_time) * 1000
            logger.error(f"[{req_id}] Secondary backup connection exception: {backup_exc}", exc_info=True)
            error_payload = self._sanitize_error(
                code="GATEWAY_SERVICE_UNAVAILABLE",
                message="Model providers are currently unavailable. Please try again shortly.",
                request_id=req_id,
                status_code=503,
            )
            return RouterResult(
                success=False,
                error_payload=error_payload,
                failover_occurred=True,
                failover_reason=failover_reason,
                latency_ms=total_latency,
            )
        finally:
            if should_close_backup:
                await backup_client.aclose()
