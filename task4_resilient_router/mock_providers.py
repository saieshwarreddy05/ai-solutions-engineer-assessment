"""
Task 4: Mock Upstream Model Providers
Provides primary and secondary backup model endpoints with configurable behavior
(latency, HTTP 429 status, server errors, successful generation).
"""

import asyncio
from typing import Any, Dict, List, Optional
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

primary_app = FastAPI(title="Primary Model Provider (e.g., GPT-4)")
backup_app = FastAPI(title="Secondary Backup Model Provider (e.g., Claude)")

# State configuration for testing
class ProviderConfig:
    def __init__(self):
        self.primary_mode = "success"  # "success", "timeout", "429", "500"
        self.primary_delay = 0.05
        self.backup_mode = "success"
        self.backup_delay = 0.05
        self.primary_call_count = 0
        self.backup_call_count = 0

    def reset(self):
        self.primary_mode = "success"
        self.primary_delay = 0.05
        self.backup_mode = "success"
        self.backup_delay = 0.05
        self.primary_call_count = 0
        self.backup_call_count = 0

config = ProviderConfig()


@primary_app.post("/v1/chat/completions")
async def primary_completions(request: Request):
    config.primary_call_count += 1
    body = await request.json()

    if config.primary_mode == "timeout":
        # Simulate delay exceeding 3000ms
        await asyncio.sleep(3.5)
        return {"choices": [{"message": {"role": "assistant", "content": "Primary late response"}}]}

    if config.primary_mode == "429":
        await asyncio.sleep(config.primary_delay)
        return JSONResponse(
            status_code=429,
            content={"error": {"code": "rate_limit_exceeded", "message": "Primary model capacity exceeded"}},
        )

    if config.primary_mode == "500":
        await asyncio.sleep(config.primary_delay)
        return JSONResponse(
            status_code=500,
            content={"error": {"code": "internal_server_error", "message": "Raw internal engine crash dump: Traceback (most recent call last)..."}},
        )

    # Success mode
    await asyncio.sleep(config.primary_delay)
    return {
        "id": "cmpl-primary-1",
        "provider": "primary-model",
        "choices": [{"message": {"role": "assistant", "content": "Response from Primary Model Provider."}}],
        "usage": {"total_tokens": 120},
    }


@backup_app.post("/v1/chat/completions")
async def backup_completions(request: Request):
    config.backup_call_count += 1
    body = await request.json()

    if config.backup_mode == "500":
        await asyncio.sleep(config.backup_delay)
        return JSONResponse(
            status_code=500,
            content={"error": {"code": "backup_failure", "message": "Secondary provider also down"}},
        )

    await asyncio.sleep(config.backup_delay)
    return {
        "id": "cmpl-backup-1",
        "provider": "secondary-backup-model",
        "choices": [{"message": {"role": "assistant", "content": "Response from Secondary Backup Model Provider."}}],
        "usage": {"total_tokens": 115},
    }
