"""
Unit & Integration Test Suite for Task 3: LLM Gateway Streaming Guardrail
Verifies cross-chunk boundary PII redaction, real-time chunk streaming,
memory bounds, and ultra-low TTFT.
"""

import json
import asyncio
import time
import pytest
import httpx

from task3_streaming_guardrail.guardrail import StreamingPIIGuardrail, REDACTION_PLACEHOLDER
from task3_streaming_guardrail.mock_llm_upstream import (
    app as mock_upstream_app,
    generate_token_stream,
    DEFAULT_SPLIT_TOKENS,
)
from task3_streaming_guardrail.proxy_server import create_proxy_app


@pytest.mark.asyncio
class TestStreamingPIIGuardrail:
    async def test_cross_chunk_email_redaction(self):
        """Verifies an email split across multiple chunk boundaries is fully redacted."""
        chunks = [
            "Please send the contract to ",
            "john.",
            "doe",
            "@acme-",
            "corp.com",
            " by tomorrow noon.",
        ]

        guardrail = StreamingPIIGuardrail()
        output_chunks = []

        async def stream():
            for c in chunks:
                yield c

        async for safe_text in guardrail.stream_guardrail(stream()):
            output_chunks.append(safe_text)

        full_output = "".join(output_chunks)
        assert "john.doe@acme-corp.com" not in full_output
        assert REDACTION_PLACEHOLDER in full_output
        assert "Please send the contract to [REDACTED] by tomorrow noon." == full_output
        assert guardrail.redactions_count >= 1

    async def test_cross_chunk_ssn_redaction(self):
        """Verifies SSN split across multiple chunks is caught and redacted."""
        chunks = [
            "Verification record: SSN is ",
            "123-",
            "45-",
            "6789",
            " for customer file.",
        ]

        guardrail = StreamingPIIGuardrail()
        output_chunks = []

        async def stream():
            for c in chunks:
                yield c

        async for safe_text in guardrail.stream_guardrail(stream()):
            output_chunks.append(safe_text)

        full_output = "".join(output_chunks)
        assert "123-45-6789" not in full_output
        assert "[REDACTED]" in full_output
        assert "Verification record: SSN is [REDACTED] for customer file." == full_output

    async def test_cross_chunk_credit_card_redaction(self):
        """Verifies 16-digit credit card split across chunks is caught and redacted."""
        chunks = [
            "Payment processed with card: ",
            "4111 ",
            "2222 ",
            "3333 ",
            "4444",
            " expiration 12/28.",
        ]

        guardrail = StreamingPIIGuardrail()
        output_chunks = []

        async def stream():
            for c in chunks:
                yield c

        async for safe_text in guardrail.stream_guardrail(stream()):
            output_chunks.append(safe_text)

        full_output = "".join(output_chunks)
        assert "4111 2222 3333 4444" not in full_output
        assert "[REDACTED]" in full_output
        assert "Payment processed with card: [REDACTED] expiration 12/28." == full_output

    async def test_clean_text_preservation(self):
        """Regular non-PII text must pass through intact without unintended modifications."""
        text = "Antigravity models optimize latency and deliver resilient proxy architectures across clouds."
        guardrail = StreamingPIIGuardrail()
        output_chunks = []

        async def stream():
            for word in text.split(" "):
                yield word + " "

        async for chunk in guardrail.stream_guardrail(stream()):
            output_chunks.append(chunk)

        reconstructed = "".join(output_chunks).strip()
        assert reconstructed == text
        assert guardrail.redactions_count == 0

    async def test_bounded_memory_efficiency(self):
        """
        Evaluation Criterion: Memory efficiency.
        Stream thousands of tokens and verify buffer size is strictly O(1) bounded (peak <= 65 chars).
        """
        guardrail = StreamingPIIGuardrail(window_size=35)

        async def huge_token_stream():
            # Emit 1,000 chunks of safe text with occasional emails
            for i in range(1000):
                if i % 100 == 50:
                    yield f"user_{i}@domain{i}.com "
                else:
                    yield f"token_{i}_data "

        async for _ in guardrail.stream_guardrail(huge_token_stream()):
            # Assert that buffer at no point grows unbounded
            assert len(guardrail.buffer) <= 100, f"Buffer exceeded safe bounds: {len(guardrail.buffer)}"

        # Verify peak buffer size remained tightly bounded despite 10,000+ characters processed
        assert guardrail.peak_buffer_size <= 70
        assert guardrail.total_input_chars > 10000

    async def test_real_time_streaming_ttft(self):
        """
        Evaluation Criterion: Minimize Time To First Token (TTFT).
        Verify that safe tokens are emitted incrementally before the slow stream completes.
        """
        guardrail = StreamingPIIGuardrail()
        emission_timestamps = []

        async def slow_stream():
            words = [
                "The", " quick", " brown", " fox", " jumps", " over", " the", " lazy", " dog.",
                " Contact:", " admin", "@cloud", ".net", " today."
            ]
            for w in words:
                await asyncio.sleep(0.02)  # 20ms delay per token
                yield w

        start_time = time.monotonic()
        first_token_time = None

        async for safe_chunk in guardrail.stream_guardrail(slow_stream()):
            now = time.monotonic()
            if first_token_time is None:
                first_token_time = now - start_time
            emission_timestamps.append(now)

        total_duration = time.monotonic() - start_time

        # TTFT must be significantly smaller than total completion duration!
        assert first_token_time is not None
        assert first_token_time < (total_duration * 0.6), (
            f"TTFT was too slow: {first_token_time:.3f}s vs total duration {total_duration:.3f}s"
        )
        # Verify multiple discrete emissions occurred over time (not buffered all at once at the end)
        assert len(emission_timestamps) >= 3


@pytest.mark.asyncio
class TestLLMStreamingProxyEndpoint:
    async def test_proxy_end_to_end_streaming_guardrail(self):
        """
        Full integration test connecting LLM Gateway proxy to mock upstream provider.
        Verifies Server-Sent Events (SSE) streaming and real-time redaction.
        """
        # Set up in-memory upstream transport
        upstream_transport = httpx.ASGITransport(app=mock_upstream_app)
        async with httpx.AsyncClient(transport=upstream_transport, base_url="http://mock-llm") as upstream_client:
            proxy_app = create_proxy_app(
                upstream_url="http://mock-llm/v1/chat/completions",
                http_client=upstream_client,
            )

            # Route proxy calls through in-memory transport
            async with httpx.AsyncClient(transport=httpx.ASGITransport(app=proxy_app), base_url="http://proxy") as client:
                # Stream response from proxy
                received_contents = []
                async with client.stream(
                    "POST",
                    "/v1/chat/completions",
                    json={"stream": True, "prompt": "Give me account details"},
                ) as resp:
                    assert resp.status_code == 200
                    assert "text/event-stream" in resp.headers.get("content-type", "")

                    async for line in resp.aiter_lines():
                        if line.startswith("data: "):
                            data_str = line[6:].strip()
                            if data_str == "[DONE]":
                                break
                            try:
                                payload = json.loads(data_str)
                                choices = payload.get("choices", [])
                                if choices and "delta" in choices[0]:
                                    content = choices[0]["delta"].get("content", "")
                                    if content:
                                        received_contents.append(content)
                            except json.JSONDecodeError:
                                pass

                full_text = "".join(received_contents)

                # Assert that all PII was redacted
                assert "alice.smith@cyberdyne-systems.org" not in full_text
                assert "987-65-4321" not in full_text
                assert "4532 0150 8899 2311" not in full_text

                # Assert redaction placeholders are present
                assert "[REDACTED]" in full_text

                # Assert safe content remains intact
                assert "Hello!" in full_text
                assert "Here is the account information you requested:" in full_text
                assert "Please let us know if you need anything else." in full_text
