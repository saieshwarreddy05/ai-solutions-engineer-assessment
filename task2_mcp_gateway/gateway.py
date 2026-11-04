"""
Task 2: MCP Security Gateway Reverse Proxy
Implements fine-grained method-level tool filtering and Bearer token authorization.
Intersects incoming AI agent requests before they reach downstream MCP servers.
"""

import sys
import json
import logging
from typing import Any, Dict, Optional
import httpx
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request, Response
from fastapi.responses import JSONResponse

from task2_mcp_gateway.auth import (
    UserRole,
    extract_bearer_token,
    authenticate_token,
)

# ---------------------------------------------------------------------------
# Logging Setup (stderr-only audit trail)
# ---------------------------------------------------------------------------
logger = logging.getLogger("mcp_gateway")
handler = logging.StreamHandler(sys.stderr)
handler.setFormatter(
    logging.Formatter("[%(asctime)s] [GATEWAY-%(levelname)s] %(message)s")
)
logger.addHandler(handler)
logger.setLevel(logging.INFO)

# Error Code Constant
ERR_UNAUTHORIZED_TOOL_CALL = -32001


def create_gateway_app(downstream_url: str = "http://127.0.0.1:8001/mcp", http_client: Optional[httpx.AsyncClient] = None) -> FastAPI:
    """
    Factory creating configured MCP Gateway proxy instance pointing to downstream MCP endpoint.
    """
    client = http_client or httpx.AsyncClient(timeout=10.0)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        yield
        if not http_client:
            await client.aclose()

    app = FastAPI(title="MCP Security Gateway Proxy", lifespan=lifespan)

    @app.post("/mcp")
    @app.post("/rpc")
    async def proxy_mcp_request(request: Request):
        """
        Main reverse proxy entrypoint.
        1. Validates Authorization Bearer token and extracts role.
        2. Validates JSON-RPC wire payload.
        3. Enforces authorization policies:
           - tools/list: transparently forward.
           - tools/call with admin_*: requires role == 'admin'. Block with -32001 if unauthorized.
           - tools/call without admin_*: forward.
        """
        # 1. Extract and Verify Bearer Token
        auth_header = request.headers.get("Authorization")
        raw_token = extract_bearer_token(auth_header)

        if not raw_token:
            logger.warning("Rejected request: Missing or malformed Authorization header.")
            return JSONResponse(
                status_code=401,
                content={
                    "jsonrpc": "2.0",
                    "id": None,
                    "error": {
                        "code": -32000,
                        "message": "Unauthorized: Bearer token is required in Authorization header.",
                    },
                },
            )

        auth_res = authenticate_token(raw_token)
        if not auth_res.is_authenticated or not auth_res.role:
            logger.warning(f"Rejected request: Invalid Bearer token. Details: {auth_res.error_message}")
            return JSONResponse(
                status_code=401,
                content={
                    "jsonrpc": "2.0",
                    "id": None,
                    "error": {
                        "code": -32000,
                        "message": f"Unauthorized: {auth_res.error_message}",
                    },
                },
            )

        user_role = auth_res.role
        logger.info(f"Authenticated request from role='{user_role.value}'")

        # 2. Parse JSON-RPC wire format
        try:
            body_bytes = await request.body()
            payload = json.loads(body_bytes.decode("utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError) as e:
            logger.error(f"Failed to parse incoming JSON-RPC wire format: {e}")
            return JSONResponse(
                status_code=400,
                content={
                    "jsonrpc": "2.0",
                    "id": None,
                    "error": {"code": -32700, "message": "Parse error: Invalid JSON payload"},
                },
            )

        if not isinstance(payload, dict):
            return JSONResponse(
                status_code=400,
                content={
                    "jsonrpc": "2.0",
                    "id": None,
                    "error": {"code": -32600, "message": "Invalid Request: root payload must be an object"},
                },
            )

        req_id = payload.get("id")
        method = payload.get("method")
        params = payload.get("params", {})

        if payload.get("jsonrpc") != "2.0" or not method:
            return JSONResponse(
                status_code=400,
                content={
                    "jsonrpc": "2.0",
                    "id": req_id,
                    "error": {
                        "code": -32600,
                        "message": "Invalid Request: missing jsonrpc='2.0' or method attribute",
                    },
                },
            )

        # 3. Policy Inspection & Method Filtering
        if method == "tools/list":
            logger.info("Forwarding 'tools/list' transparently to downstream server.")
            return await _forward_to_downstream(client, downstream_url, body_bytes, request.headers)

        if method == "tools/call":
            if not isinstance(params, dict):
                return JSONResponse(
                    content={
                        "jsonrpc": "2.0",
                        "id": req_id,
                        "error": {
                            "code": -32602,
                            "message": "Invalid params: params must be an object",
                        },
                    }
                )

            tool_name = params.get("name", "")

            # Security Guardrail: Check admin_ prefix
            if tool_name.startswith("admin_"):
                logger.info(f"Evaluating access for restricted tool: '{tool_name}' against role: '{user_role.value}'")
                if user_role != UserRole.ADMIN:
                    # INTERCEPT: Return JSON-RPC Error -32001 directly without downstream call!
                    logger.warning(
                        f"ACCESS DENIED: Role '{user_role.value}' attempted to execute admin tool '{tool_name}'. "
                        "Intercepting and returning JSON-RPC -32001 without contacting downstream."
                    )
                    return JSONResponse(
                        content={
                            "jsonrpc": "2.0",
                            "id": req_id,
                            "error": {
                                "code": ERR_UNAUTHORIZED_TOOL_CALL,
                                "message": "Unauthorized Tool Call: Tool requires admin privileges",
                                "data": {
                                    "tool": tool_name,
                                    "role": user_role.value,
                                    "required_role": "admin",
                                },
                            },
                        }
                    )
                else:
                    logger.info(f"Access granted: Admin user executing '{tool_name}'. Forwarding downstream.")

            # Authorized tool call: Forward to downstream server
            return await _forward_to_downstream(client, downstream_url, body_bytes, request.headers)

        # Any other method (e.g. initialize, ping, resources/*) forward transparently
        logger.info(f"Forwarding general method '{method}' to downstream server.")
        return await _forward_to_downstream(client, downstream_url, body_bytes, request.headers)

    return app


async def _forward_to_downstream(
    client: httpx.AsyncClient,
    downstream_url: str,
    body: bytes,
    original_headers: Any,
) -> Response:
    """Forward payload to downstream MCP HTTP server and return verbatim response."""
    # Filter out hop-by-hop headers
    forward_headers = {
        "Content-Type": "application/json",
        "Accept": "application/json",
    }
    # Pass along incoming Authorization header if present
    if "Authorization" in original_headers:
        forward_headers["Authorization"] = original_headers["Authorization"]

    try:
        downstream_resp = await client.post(
            downstream_url,
            content=body,
            headers=forward_headers,
        )
        return Response(
            content=downstream_resp.content,
            status_code=downstream_resp.status_code,
            media_type="application/json",
        )
    except httpx.RequestError as exc:
        logger.error(f"Downstream connection failure at '{downstream_url}': {exc}")
        return JSONResponse(
            status_code=502,
            content={
                "jsonrpc": "2.0",
                "id": None,
                "error": {
                    "code": -32603,
                    "message": "Gateway Bad Gateway: Unable to connect to downstream MCP server.",
                },
            },
        )
