"""
Task 2: Downstream Mock MCP Server
Simulates an enterprise downstream MCP server exposing administrative and read-only tools.
Tracks request counts and call history so tests can assert zero downstream invocations on blocked calls.
"""

from typing import Any, Dict, List, Optional
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

app = FastAPI(title="Downstream Mock MCP Server")

# Downstream state & telemetry
call_history: List[Dict[str, Any]] = []

DOWNSTREAM_TOOLS = [
    {
        "name": "admin_reset_key",
        "description": "Administrative tool to revoke and re-issue a tenant API key.",
        "inputSchema": {
            "type": "object",
            "properties": {"tenant_id": {"type": "string"}},
            "required": ["tenant_id"],
        },
    },
    {
        "name": "admin_purge_cache",
        "description": "Administrative tool to invalidate all caching tiers.",
        "inputSchema": {
            "type": "object",
            "properties": {"tier": {"type": "string"}},
            "required": ["tier"],
        },
    },
    {
        "name": "get_system_health",
        "description": "Public diagnostic tool to inspect system health metrics.",
        "inputSchema": {
            "type": "object",
            "properties": {"service": {"type": "string"}},
        },
    },
    {
        "name": "query_customer_metrics",
        "description": "Read-only tool to inspect aggregated customer traffic.",
        "inputSchema": {
            "type": "object",
            "properties": {"window_hours": {"type": "integer"}},
        },
    },
]


@app.post("/mcp")
async def handle_mcp_jsonrpc(request: Request):
    """Downstream JSON-RPC 2.0 processor."""
    try:
        payload = await request.json()
    except Exception:
        return JSONResponse(
            status_code=400,
            content={
                "jsonrpc": "2.0",
                "id": None,
                "error": {"code": -32700, "message": "Parse error"},
            },
        )

    req_id = payload.get("id")
    method = payload.get("method")
    params = payload.get("params", {})

    # Log downstream invocation
    call_history.append({
        "method": method,
        "params": params,
        "id": req_id,
        "headers": dict(request.headers),
    })

    if method == "tools/list":
        return JSONResponse(
            content={
                "jsonrpc": "2.0",
                "id": req_id,
                "result": {"tools": DOWNSTREAM_TOOLS},
            }
        )

    if method == "tools/call":
        tool_name = params.get("name")
        args = params.get("arguments", {})

        if tool_name == "admin_reset_key":
            return JSONResponse(
                content={
                    "jsonrpc": "2.0",
                    "id": req_id,
                    "result": {
                        "content": [
                            {
                                "type": "text",
                                "text": f"Successfully rotated key for tenant {args.get('tenant_id', 'default')}. New key: key_live_rotated_{req_id}",
                            }
                        ],
                        "isError": False,
                    },
                }
            )

        if tool_name == "admin_purge_cache":
            return JSONResponse(
                content={
                    "jsonrpc": "2.0",
                    "id": req_id,
                    "result": {
                        "content": [
                            {
                                "type": "text",
                                "text": f"Cache tier {args.get('tier', 'all')} invalidated.",
                            }
                        ],
                        "isError": False,
                    },
                }
            )

        if tool_name == "get_system_health":
            return JSONResponse(
                content={
                    "jsonrpc": "2.0",
                    "id": req_id,
                    "result": {
                        "content": [
                            {
                                "type": "text",
                                "text": "System status: Healthy. Memory: 42%, CPU: 12%, Latency: 14ms.",
                            }
                        ],
                        "isError": False,
                    },
                }
            )

        if tool_name == "query_customer_metrics":
            return JSONResponse(
                content={
                    "jsonrpc": "2.0",
                    "id": req_id,
                    "result": {
                        "content": [
                            {
                                "type": "text",
                                "text": "Metrics: 1,240 requests/min, 99.98% availability.",
                            }
                        ],
                        "isError": False,
                    },
                }
            )

        # Unrecognized tool on downstream
        return JSONResponse(
            content={
                "jsonrpc": "2.0",
                "id": req_id,
                "error": {
                    "code": -32601,
                    "message": f"Downstream tool '{tool_name}' not found",
                },
            }
        )

    return JSONResponse(
        content={
            "jsonrpc": "2.0",
            "id": req_id,
            "error": {"code": -32601, "message": f"Method '{method}' not found"},
        }
    )


def reset_mock_state():
    """Helper to reset call telemetry between tests."""
    call_history.clear()
