# AI Field Developer & Solutions Engineer Assessment Suite

This repository implements all 4 technical assessment tasks evaluating candidates on building Model Context Protocol (MCP) servers, implementing LLM & MCP security proxy gateways, handling stream guardrails, managing model routing resilience, and zero-trust proxy design.

---

## Quick Start & Verification

### Prerequisites
- Python 3.12+ (or activated local virtual environment `.venv`)

### Run All 35 Tests
```bash
./run_all_tests.sh
```
Or directly via pytest:
```bash
source .venv/bin/activate
PYTHONPATH=. pytest -v
```

---

## Architecture & Implementation Overview

```
Quilr HA/
├── task1_mcp_server/              # Task 1: Custom MCP Server with Strict Validation & Transport
│   ├── models.py                  # Pydantic schemas (CUST-XXXXX, positive amounts, min_length=10)
│   ├── server.py                  # MCPServer with stdio isolation & JSON-RPC error codes
│   ├── stdio_runner.py            # Standalone stdio runner ensuring stdout has zero non-JSON pollution
│   └── test_task1.py              # Pytest suite (11 tests: isolation, protocol, validation)
│
├── task2_mcp_gateway/             # Task 2: MCP Security Gateway Proxy (Tool Filtering & Auth)
│   ├── auth.py                    # Bearer token parsing and role extraction (admin vs viewer)
│   ├── mock_mcp_server.py         # Mock downstream MCP server tracking call history
│   ├── gateway.py                 # Reverse proxy intercepting admin_* tools with -32001
│   └── test_task2.py              # Pytest suite (7 tests: token auth, forwarding, interception)
│
├── task3_streaming_guardrail/     # Task 3: LLM Gateway Streaming Guardrail (PII Redaction)
│   ├── guardrail.py               # Sliding window streaming redaction engine (Emails, SSNs, CCs)
│   ├── mock_llm_upstream.py       # Mock upstream LLM streaming SSE with cross-chunk PII splits
│   ├── proxy_server.py            # FastAPI streaming proxy with ultra-low TTFT and O(1) memory
│   └── test_task3.py              # Pytest suite (7 tests: cross-chunk regex, TTFT, memory bound)
│
├── task4_resilient_router/        # Task 4: Rate-Limiting & Model Fallback Router
│   ├── rate_limiter.py            # On-disk SQLite token sliding window rate limiter (WAL mode)
│   ├── router.py                  # Fallback router: 3000ms timeout race condition & 429 failover
│   ├── mock_providers.py          # Primary & Secondary mock model providers
│   ├── gateway_app.py             # Full FastAPI Gateway endpoint integrating limiter & router
│   └── test_task4.py              # Pytest suite (10 tests: WAL persistence, eviction, failover)
│
├── pytest.ini                     # Pytest configuration
├── run_all_tests.sh               # Execution script
└── README.md                      # Complete system documentation
```

---

## Detailed Task Breakdown

### Task 1: Custom MCP Server with Strict Validation & Transport Handling
- **Problem**: Build a runnable MCP server using the official SDK (`mcp`) exposing `get_customer_record` and `trigger_refund`.
- **Validation**: Enforce strict Pydantic schemas:
  - `get_customer_record`: `customer_id` validated strictly against pattern `^CUST-[A-Z0-9]{5}$`.
  - `trigger_refund`: `customer_id` (`CUST-XXXXX`), `amount` (strictly positive float `> 0`), and `reason` (string with `min_length=10`).
- **STDIO Transport Isolation**:
  - `sys.stdout` is strictly reserved for valid JSON-RPC frames (`jsonrpc: "2.0"`).
  - All application, debug, and trace logs are directed exclusively to `sys.stderr`.
- **Standard Protocol Error Codes**:
  - Invalid parameters return `-32602: Invalid params` with structured field validation diagnostics.
  - Unknown tools return `-32601: Method not found`.
  - Malformed payloads return `-32700: Parse error` or `-32600: Invalid Request`.

### Task 2: MCP Security Gateway Proxy (Tool Filtering & Auth)
- **Problem**: Build a lightweight HTTP/JSON-RPC reverse proxy sitting between an AI agent and downstream mock MCP servers.
- **Authentication**:
  - Extracts `Authorization: Bearer <token>` and maps credentials to user roles (`admin` or `viewer`).
  - Missing or unrecognized tokens immediately return `401 Unauthorized`.
- **Method-Level Tool Filtering**:
  - `tools/list`: Transparently forwarded to downstream MCP server.
  - `tools/call`: Inspects `params.name`. If `params.name.startswith("admin_")`:
    - If user role is NOT `admin`: Intercepts and directly returns JSON-RPC error:
      ```json
      {
        "jsonrpc": "2.0",
        "id": req_id,
        "error": {
          "code": -32001,
          "message": "Unauthorized Tool Call: Tool requires admin privileges",
          "data": {"tool": "admin_reset_key", "role": "viewer", "required_role": "admin"}
        }
      }
      ```
    - **Zero Downstream Invocations**: Downstream server receives 0 calls for unauthorized attempts.
  - Non-restricted tools (e.g. `get_system_health`) and authorized admin calls forward transparently.

### Task 3: LLM Gateway Streaming Guardrail (PII Redaction)
- **Problem**: Intercept streaming LLM generation chunks in real-time, redact sensitive PII (Emails, SSNs, Credit Cards) into `[REDACTED]`, while maintaining ultra-low Time To First Token (TTFT) and $O(1)$ memory.
- **Why Naive Approaches Fail**:
  - Buffering the entire stream ruins TTFT and requires $O(N)$ memory.
  - Chunk-by-chunk regex leaks PII split across chunk boundaries (e.g., chunk 1: `alice.`, chunk 2: `smith@`, chunk 3: `cyberdyne.org`).
- **The Solution: Lookahead Sliding Window Engine**:
  - Maintains a small lookahead buffer (35 characters, sized to the maximum possible PII prefix).
  - Emits safe text immediately as incoming deltas arrive.
  - Retains only trailing candidate prefixes across chunk boundaries.
  - When stream terminates, performs final flush of the lookahead buffer.
  - **Memory Bound**: Buffer peak size strictly $\le 70$ characters even over a 10,000+ character stream.

### Task 4: Rate-Limiting & Model Fallback Router for LLM Gateways
- **Problem**: Build a resilient model-routing module for incoming completion requests with rate limiting and automated failover.
- **Token-Aware Rate Limiting (On-Disk SQLite)**:
  - Backed by an on-disk SQLite database (`data/rate_limit.db`) operating in **WAL mode** (`PRAGMA journal_mode=WAL; PRAGMA synchronous=NORMAL;`) for concurrent async transactions.
  - Enforces sliding 60-second window quotas (e.g., 50,000 tokens/minute per tenant key).
  - Automatically evicts stale state (`DELETE FROM token_usage WHERE timestamp < window_start;`).
  - Rejections return HTTP 429 with `Retry-After` header.
- **Model Fallback Router (3000ms Timeout & 429 Failover)**:
  - Dispatches to primary model with strict 3000ms timeout race handling (`asyncio.wait_for(timeout=3.0)`).
  - If primary model times out (> 3000ms) or returns HTTP 429 Too Many Requests, automatically triggers failover to secondary backup model.
  - Fallback request succeeds seamlessly, returning secondary provider output.
- **Standardized Error Sanitization**:
  - If all providers fail, returns standardized gateway error payload:
    ```json
    {
      "error": {
        "code": "GATEWAY_UPSTREAM_FAILURE",
        "message": "Both primary and secondary backup model providers failed to satisfy the request.",
        "request_id": "req-...",
        "status": 502,
        "timestamp": 1741000000.123
      }
    }
    ```
  - Upstream stack traces, Python tracebacks, internal URLs, and credentials are completely sanitized.
