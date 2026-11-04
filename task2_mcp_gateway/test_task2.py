"""
Unit & Integration Test Suite for Task 2: MCP Security Gateway Proxy
Tests Bearer Token Auth, Method Filtering, Tool Interception, and Downstream Forwarding.
"""

import json
import pytest
import httpx
from fastapi.testclient import TestClient

from task2_mcp_gateway.mock_mcp_server import app as downstream_app, call_history, reset_mock_state
from task2_mcp_gateway.gateway import create_gateway_app, ERR_UNAUTHORIZED_TOOL_CALL
from task2_mcp_gateway.auth import TOKEN_REGISTRY, UserRole


ADMIN_TOKEN = "admin-secret-token-xyz"
VIEWER_TOKEN = "viewer-readonly-token-abc"


@pytest.fixture(autouse=True)
def setup_teardown():
    reset_mock_state()
    yield
    reset_mock_state()


@pytest.mark.asyncio
class TestMCPSecurityGateway:
    async def test_missing_auth_header_rejected(self):
        gateway_app = create_gateway_app(downstream_url="http://mock-downstream/mcp")
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=gateway_app), base_url="http://gateway") as client:
            resp = await client.post(
                "/mcp",
                json={"jsonrpc": "2.0", "id": 1, "method": "tools/list", "params": {}},
            )
            assert resp.status_code == 401
            data = resp.json()
            assert data["error"]["code"] == -32000
            assert "Bearer token is required" in data["error"]["message"]
            assert len(call_history) == 0

    async def test_invalid_bearer_token_rejected(self):
        gateway_app = create_gateway_app(downstream_url="http://mock-downstream/mcp")
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=gateway_app), base_url="http://gateway") as client:
            resp = await client.post(
                "/mcp",
                headers={"Authorization": "Bearer totally-fake-token-999"},
                json={"jsonrpc": "2.0", "id": 2, "method": "tools/list", "params": {}},
            )
            assert resp.status_code == 401
            assert len(call_history) == 0

    async def test_tools_list_forwarded_transparently_for_viewer(self):
        """tools/list must be forwarded transparently without filtering."""
        downstream_transport = httpx.ASGITransport(app=downstream_app)
        async with httpx.AsyncClient(transport=downstream_transport, base_url="http://mock-downstream") as ds_client:
            gateway_app = create_gateway_app(
                downstream_url="http://mock-downstream/mcp",
                http_client=ds_client
            )
            async with httpx.AsyncClient(transport=httpx.ASGITransport(app=gateway_app), base_url="http://gateway") as client:
                resp = await client.post(
                    "/mcp",
                    headers={"Authorization": f"Bearer {VIEWER_TOKEN}"},
                    json={"jsonrpc": "2.0", "id": 10, "method": "tools/list", "params": {}},
                )
                assert resp.status_code == 200
                data = resp.json()
                assert data["id"] == 10
                assert "tools" in data["result"]
                # Downstream was invoked
                assert len(call_history) == 1
                assert call_history[0]["method"] == "tools/list"

    async def test_admin_tool_blocked_for_viewer_without_downstream_call(self):
        """
        Critical Requirement:
        If role is not admin, admin_* tools must be intercepted with -32001
        directly WITHOUT calling the downstream server.
        """
        downstream_transport = httpx.ASGITransport(app=downstream_app)
        async with httpx.AsyncClient(transport=downstream_transport, base_url="http://mock-downstream") as ds_client:
            gateway_app = create_gateway_app(
                downstream_url="http://mock-downstream/mcp",
                http_client=ds_client
            )
            async with httpx.AsyncClient(transport=httpx.ASGITransport(app=gateway_app), base_url="http://gateway") as client:
                resp = await client.post(
                    "/mcp",
                    headers={"Authorization": f"Bearer {VIEWER_TOKEN}"},
                    json={
                        "jsonrpc": "2.0",
                        "id": 20,
                        "method": "tools/call",
                        "params": {
                            "name": "admin_reset_key",
                            "arguments": {"tenant_id": "tenant-abc"},
                        },
                    },
                )

                assert resp.status_code == 200
                data = resp.json()
                assert data["id"] == 20
                assert "error" in data
                assert data["error"]["code"] == ERR_UNAUTHORIZED_TOOL_CALL
                assert ERR_UNAUTHORIZED_TOOL_CALL == -32001
                assert "Unauthorized Tool Call" in data["error"]["message"]
                assert data["error"]["data"]["tool"] == "admin_reset_key"
                assert data["error"]["data"]["role"] == "viewer"

                # Crucial assertion: Downstream call history MUST be 0!
                assert len(call_history) == 0, "Downstream server was illegally called for unauthorized tool!"

    async def test_admin_tool_allowed_for_admin_role(self):
        """Admin role can invoke admin_* tools and request is forwarded."""
        downstream_transport = httpx.ASGITransport(app=downstream_app)
        async with httpx.AsyncClient(transport=downstream_transport, base_url="http://mock-downstream") as ds_client:
            gateway_app = create_gateway_app(
                downstream_url="http://mock-downstream/mcp",
                http_client=ds_client
            )
            async with httpx.AsyncClient(transport=httpx.ASGITransport(app=gateway_app), base_url="http://gateway") as client:
                resp = await client.post(
                    "/mcp",
                    headers={"Authorization": f"Bearer {ADMIN_TOKEN}"},
                    json={
                        "jsonrpc": "2.0",
                        "id": 30,
                        "method": "tools/call",
                        "params": {
                            "name": "admin_reset_key",
                            "arguments": {"tenant_id": "tenant-corp"},
                        },
                    },
                )

                assert resp.status_code == 200
                data = resp.json()
                assert data["id"] == 30
                assert "result" in data
                assert "Successfully rotated key" in data["result"]["content"][0]["text"]

                # Downstream was called exactly once
                assert len(call_history) == 1
                assert call_history[0]["params"]["name"] == "admin_reset_key"

    async def test_non_admin_tool_allowed_for_viewer(self):
        """Viewer role can invoke non-admin tools (e.g. get_system_health)."""
        downstream_transport = httpx.ASGITransport(app=downstream_app)
        async with httpx.AsyncClient(transport=downstream_transport, base_url="http://mock-downstream") as ds_client:
            gateway_app = create_gateway_app(
                downstream_url="http://mock-downstream/mcp",
                http_client=ds_client
            )
            async with httpx.AsyncClient(transport=httpx.ASGITransport(app=gateway_app), base_url="http://gateway") as client:
                resp = await client.post(
                    "/mcp",
                    headers={"Authorization": f"Bearer {VIEWER_TOKEN}"},
                    json={
                        "jsonrpc": "2.0",
                        "id": 40,
                        "method": "tools/call",
                        "params": {
                            "name": "get_system_health",
                            "arguments": {"service": "database"},
                        },
                    },
                )

                assert resp.status_code == 200
                data = resp.json()
                assert data["id"] == 40
                assert "result" in data
                assert "Healthy" in data["result"]["content"][0]["text"]
                assert len(call_history) == 1

    async def test_malformed_json_rpc_structure(self):
        """Malformed JSON-RPC returns standard JSON-RPC parse/structure error."""
        gateway_app = create_gateway_app(downstream_url="http://mock-downstream/mcp")
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=gateway_app), base_url="http://gateway") as client:
            resp = await client.post(
                "/mcp",
                headers={"Authorization": f"Bearer {ADMIN_TOKEN}"},
                content="BROKEN_JSON_RPC",
            )
            assert resp.status_code == 400
            data = resp.json()
            assert data["error"]["code"] == -32700
