"""
Task 3: Mock LLM Upstream Provider
Streams completion tokens using standard Server-Sent Events (SSE).
Intentionally splits sensitive PII tokens across chunk boundaries to test guardrails.
"""

import asyncio
import json
from typing import AsyncGenerator, List, Optional
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse

app = FastAPI(title="Mock LLM Streaming Provider")

# Default token sequence with cross-chunk PII splits
DEFAULT_SPLIT_TOKENS: List[str] = [
    "Hello! ",
    "Here is the account ",
    "information you requested:\n",
    "Customer contact: ",
    "alice.",           # Email split part 1
    "smith",            # Email split part 2
    "@cyber",           # Email split part 3
    "dyne-systems",     # Email split part 4
    ".org. ",           # Email split part 5
    "Verification SSN on file: ",
    "987-",             # SSN split part 1
    "65-",              # SSN split part 2
    "4321. ",           # SSN split part 3
    "Primary billing card: ",
    "4532 ",            # Credit card split part 1
    "0150 ",            # Credit card split part 2
    "8899 ",            # Credit card split part 3
    "2311. ",           # Credit card split part 4
    "Please let us know if you need anything else."
]


async def generate_token_stream(
    tokens: Optional[List[str]] = None,
    delay_seconds: float = 0.01,
) -> AsyncGenerator[str, None]:
    """Yield raw string tokens with simulated network latency."""
    token_list = tokens if tokens is not None else DEFAULT_SPLIT_TOKENS
    for tok in token_list:
        if delay_seconds > 0:
            await asyncio.sleep(delay_seconds)
        yield tok


async def generate_openai_sse_stream(
    tokens: Optional[List[str]] = None,
    delay_seconds: float = 0.01,
) -> AsyncGenerator[str, None]:
    """Yield OpenAI-compatible SSE events formatted as data: {...}."""
    token_list = tokens if tokens is not None else DEFAULT_SPLIT_TOKENS
    for idx, tok in enumerate(token_list):
        if delay_seconds > 0:
            await asyncio.sleep(delay_seconds)
        payload = {
            "id": f"chatcmpl-mock-{idx}",
            "object": "chat.completion.chunk",
            "choices": [{"index": 0, "delta": {"content": tok}, "finish_reason": None}],
        }
        yield f"data: {json.dumps(payload)}\n\n"

    # End of stream event
    yield "data: [DONE]\n\n"


@app.post("/v1/chat/completions")
async def mock_chat_completions(request: Request):
    """OpenAI-compatible streaming chat completion endpoint."""
    body = await request.json()
    stream = body.get("stream", True)

    if stream:
        return StreamingResponse(
            generate_openai_sse_stream(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            },
        )
    else:
        # Non-streaming response
        full_text = "".join(DEFAULT_SPLIT_TOKENS)
        return {
            "id": "chatcmpl-mock-full",
            "object": "chat.completion",
            "choices": [{"index": 0, "message": {"role": "assistant", "content": full_text}}],
        }
