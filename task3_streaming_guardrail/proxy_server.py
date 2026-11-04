"""
Task 3: LLM Gateway Proxy Endpoint with Real-Time PII Streaming Guardrail
Routes text generation requests to upstream LLM providers, intercepts token deltas in real-time,
redacts sensitive PII (Emails, SSNs, Credit Cards) across chunk boundaries,
and streams the sanitized response to the client with ultra-low TTFT.
"""

import json
import logging
import sys
from typing import AsyncGenerator, Optional
import httpx
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse, JSONResponse

from task3_streaming_guardrail.guardrail import StreamingPIIGuardrail

logger = logging.getLogger("llm_gateway_proxy")
handler = logging.StreamHandler(sys.stderr)
handler.setFormatter(logging.Formatter("[%(asctime)s] [GUARDRAIL-PROXY] %(message)s"))
logger.addHandler(handler)
logger.setLevel(logging.INFO)


def create_proxy_app(
    upstream_url: str = "http://127.0.0.1:8002/v1/chat/completions",
    http_client: Optional[httpx.AsyncClient] = None,
) -> FastAPI:
    """Creates the FastAPI proxy application for streaming LLM generation."""
    app = FastAPI(title="LLM Gateway Streaming Guardrail Proxy")

    @app.post("/v1/chat/completions")
    async def chat_completions_proxy(request: Request):
        """
        Streaming reverse proxy with inline PII redaction guardrail.
        """
        try:
            body = await request.json()
        except Exception:
            return JSONResponse(status_code=400, content={"error": "Malformed JSON payload"})

        stream_requested = body.get("stream", True)
        logger.info(f"Received completion request (stream={stream_requested})")

        # Create fresh guardrail state for this stream
        guardrail = StreamingPIIGuardrail()

        # Connect to upstream streaming provider
        async def event_generator() -> AsyncGenerator[str, None]:
            client = http_client or httpx.AsyncClient(timeout=30.0)
            should_close = http_client is None
            try:
                async with client.stream(
                    "POST",
                    upstream_url,
                    json=body,
                    headers={"Content-Type": "application/json"},
                ) as upstream_response:
                    if upstream_response.status_code != 200:
                        err_content = await upstream_response.aread()
                        logger.error(f"Upstream returned error {upstream_response.status_code}: {err_content}")
                        yield f"data: {json.dumps({'error': 'Upstream provider failure'})}\n\n"
                        return

                    async for line in upstream_response.aiter_lines():
                        if not line:
                            continue

                        # Standard SSE protocol handling
                        if line.startswith("data: "):
                            data_str = line[6:].strip()
                            if data_str == "[DONE]":
                                # Stream EOF reached: flush remaining guardrail buffer
                                remaining = guardrail.flush()
                                if remaining:
                                    payload = {
                                        "choices": [{"delta": {"content": remaining}, "index": 0}],
                                    }
                                    yield f"data: {json.dumps(payload)}\n\n"
                                yield "data: [DONE]\n\n"
                                break

                            try:
                                parsed = json.loads(data_str)
                                choices = parsed.get("choices", [])
                                if choices and "delta" in choices[0]:
                                    delta_content = choices[0]["delta"].get("content", "")
                                    if delta_content:
                                        # Redact delta through sliding window guardrail
                                        safe_chunk = guardrail.process_delta(delta_content)
                                        if safe_chunk:
                                            out_payload = {
                                                "id": parsed.get("id"),
                                                "object": parsed.get("object"),
                                                "choices": [
                                                    {
                                                        "index": 0,
                                                        "delta": {"content": safe_chunk},
                                                        "finish_reason": None,
                                                    }
                                                ],
                                            }
                                            yield f"data: {json.dumps(out_payload)}\n\n"
                            except json.JSONDecodeError:
                                continue
                        else:
                            # Non-SSE raw line: pass directly through guardrail
                            safe_chunk = guardrail.process_delta(line + "\n")
                            if safe_chunk:
                                yield safe_chunk

                    # Flush any remaining buffer if [DONE] was omitted
                    final_flush = guardrail.flush()
                    if final_flush:
                        payload = {"choices": [{"delta": {"content": final_flush}, "index": 0}]}
                        yield f"data: {json.dumps(payload)}\n\n"
            except Exception as exc:
                logger.error(f"Streaming error occurred: {exc}", exc_info=True)
                yield f"data: {json.dumps({'error': f'Streaming exception: {str(exc)}'})}\n\n"
            finally:
                if should_close:
                    await client.aclose()

        return StreamingResponse(
            event_generator(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            },
        )

    return app
