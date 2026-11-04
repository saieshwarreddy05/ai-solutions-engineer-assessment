"""
Unit & Integration Test Suite for Task 1: MCP Server
Tests STDIO Isolation, Protocol Compliance, and Pydantic Schema Validation.
"""

import sys
import json
import subprocess
import pytest
from pydantic import ValidationError

from task1_mcp_server.models import (
    GetCustomerRecordInput,
    TriggerRefundInput,
    JSONRPC_INVALID_PARAMS,
    JSONRPC_METHOD_NOT_FOUND,
    JSONRPC_INVALID_REQUEST,
    JSONRPC_PARSE_ERROR,
)
from task1_mcp_server.server import create_customer_mcp_server, DirectMCPProtocolHandler


# ===========================================================================
# 1. Schema Validation Unit Tests
# ===========================================================================
class TestSchemaValidation:
    def test_valid_customer_id_formats(self):
        valid_ids = ["CUST-10001", "CUST-99999", "CUST-ABCDE", "CUST-1A2B3", "CUST-00000"]
        for cid in valid_ids:
            model = GetCustomerRecordInput(customer_id=cid)
            assert model.customer_id == cid

    def test_invalid_customer_id_formats(self):
        invalid_ids = [
            "10001",           # Missing prefix
            "CUST-1234",       # Too short (4 chars)
            "CUST-123456",     # Too long (6 chars)
            "cust-10001",      # Lowercase
            "CUST_10001",      # Underscore instead of hyphen
            "CUST-12@45",      # Special character
            "",                # Empty
            "CUSTOMER-10001",  # Wrong prefix
        ]
        for cid in invalid_ids:
            with pytest.raises(ValidationError):
                GetCustomerRecordInput(customer_id=cid)

    def test_valid_trigger_refund(self):
        refund = TriggerRefundInput(
            customer_id="CUST-10001",
            amount=50.25,
            reason="Customer received defective shipment",
        )
        assert refund.customer_id == "CUST-10001"
        assert refund.amount == 50.25
        assert refund.reason == "Customer received defective shipment"

    def test_refund_rejects_non_positive_amount(self):
        # 0.0 amount
        with pytest.raises(ValidationError) as exc_0:
            TriggerRefundInput(
                customer_id="CUST-10001",
                amount=0.0,
                reason="Customer requested exchange instead",
            )
        assert "greater than 0" in str(exc_0.value) or "strictly positive" in str(exc_0.value)

        # Negative amount
        with pytest.raises(ValidationError) as exc_neg:
            TriggerRefundInput(
                customer_id="CUST-10001",
                amount=-25.50,
                reason="Customer requested exchange instead",
            )
        assert "greater than 0" in str(exc_neg.value) or "strictly positive" in str(exc_neg.value)

    def test_refund_rejects_short_reason(self):
        # Reason shorter than 10 chars
        with pytest.raises(ValidationError) as exc:
            TriggerRefundInput(
                customer_id="CUST-10001",
                amount=10.0,
                reason="Broken",  # Only 6 characters
            )
        assert "at least 10" in str(exc.value)

        # Whitespace padded short reason
        with pytest.raises(ValidationError):
            TriggerRefundInput(
                customer_id="CUST-10001",
                amount=10.0,
                reason="   Short   ",
            )


# ===========================================================================
# 2. Protocol Compliance & Error Code Mapping
# ===========================================================================
@pytest.mark.asyncio
class TestProtocolCompliance:
    async def test_initialize_handshake(self):
        server = create_customer_mcp_server()
        handler = DirectMCPProtocolHandler(server)

        req = json.dumps({
            "jsonrpc": "2.0",
            "id": 1,
            "method": "initialize",
            "params": {"protocolVersion": "2024-11-05"}
        })
        resp = await handler.handle_message(req)
        assert resp is not None
        data = json.loads(resp)
        assert data["id"] == 1
        assert "result" in data
        assert data["result"]["serverInfo"]["name"] == "CustomerOperationsServer"

    async def test_tools_list_schema(self):
        server = create_customer_mcp_server()
        handler = DirectMCPProtocolHandler(server)

        req = json.dumps({"jsonrpc": "2.0", "id": 2, "method": "tools/list", "params": {}})
        resp = await handler.handle_message(req)
        data = json.loads(resp)
        tools = data["result"]["tools"]
        tool_names = [t["name"] for t in tools]
        assert "get_customer_record" in tool_names
        assert "trigger_refund" in tool_names

    async def test_invalid_params_error_code(self):
        server = create_customer_mcp_server()
        handler = DirectMCPProtocolHandler(server)

        # Call with malformed customer_id
        req = json.dumps({
            "jsonrpc": "2.0",
            "id": 10,
            "method": "tools/call",
            "params": {
                "name": "get_customer_record",
                "arguments": {"customer_id": "MALFORMED-1"}
            }
        })
        resp = await handler.handle_message(req)
        data = json.loads(resp)
        assert data["id"] == 10
        assert "error" in data
        assert data["error"]["code"] == JSONRPC_INVALID_PARAMS
        assert "pattern" in data["error"]["message"] or "CUST" in data["error"]["message"]

    async def test_method_not_found_error_code(self):
        server = create_customer_mcp_server()
        handler = DirectMCPProtocolHandler(server)

        req = json.dumps({
            "jsonrpc": "2.0",
            "id": 11,
            "method": "tools/call",
            "params": {"name": "non_existent_tool", "arguments": {}}
        })
        resp = await handler.handle_message(req)
        data = json.loads(resp)
        assert data["id"] == 11
        assert "error" in data
        assert data["error"]["code"] == JSONRPC_METHOD_NOT_FOUND

    async def test_malformed_json_parse_error(self):
        server = create_customer_mcp_server()
        handler = DirectMCPProtocolHandler(server)

        resp = await handler.handle_message("NOT A VALID JSON STRING")
        data = json.loads(resp)
        assert data["error"]["code"] == JSONRPC_PARSE_ERROR


# ===========================================================================
# 3. STDIO Transport Isolation & Integration Test
# ===========================================================================
class TestSTDIOTransportIsolation:
    def test_stdio_isolation_and_flow(self):
        """Run server in subprocess. Ensure STDOUT is strictly valid JSON-RPC frames and logs go to STDERR."""
        python_bin = sys.executable
        proc = subprocess.Popen(
            [python_bin, "-m", "task1_mcp_server.stdio_runner"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
        )

        def send_and_receive(request_dict):
            proc.stdin.write(json.dumps(request_dict) + "\n")
            proc.stdin.flush()
            line = proc.stdout.readline()
            assert line, "Process unexpectedly closed stdout!"
            # Crucial assertion: STDOUT line must be strictly JSON
            parsed = json.loads(line.strip())
            assert parsed.get("jsonrpc") == "2.0"
            return parsed

        # 1. Handshake
        init_res = send_and_receive({
            "jsonrpc": "2.0",
            "id": 100,
            "method": "initialize",
            "params": {}
        })
        assert init_res["id"] == 100

        # 2. Successful Tool Call
        get_res = send_and_receive({
            "jsonrpc": "2.0",
            "id": 101,
            "method": "tools/call",
            "params": {
                "name": "get_customer_record",
                "arguments": {"customer_id": "CUST-10001"}
            }
        })
        assert get_res["id"] == 101
        assert get_res.get("result", {}).get("isError") is False
        content_text = get_res["result"]["content"][0]["text"]
        assert "Sarah Connor" in content_text

        # 3. Invalid Customer ID -> Must return code -32602
        err_res = send_and_receive({
            "jsonrpc": "2.0",
            "id": 102,
            "method": "tools/call",
            "params": {
                "name": "get_customer_record",
                "arguments": {"customer_id": "BAD-ID"}
            }
        })
        assert err_res["id"] == 102
        assert "error" in err_res
        assert err_res["error"]["code"] == JSONRPC_INVALID_PARAMS

        # 4. Trigger Refund Successful
        refund_res = send_and_receive({
            "jsonrpc": "2.0",
            "id": 103,
            "method": "tools/call",
            "params": {
                "name": "trigger_refund",
                "arguments": {
                    "customer_id": "CUST-10001",
                    "amount": 75.50,
                    "reason": "Billing adjustment for server maintenance"
                }
            }
        })
        assert refund_res["id"] == 103
        assert "COMPLETED" in refund_res["result"]["content"][0]["text"]

        # 5. Invalid Refund (amount <= 0)
        invalid_refund = send_and_receive({
            "jsonrpc": "2.0",
            "id": 104,
            "method": "tools/call",
            "params": {
                "name": "trigger_refund",
                "arguments": {
                    "customer_id": "CUST-10001",
                    "amount": -20.0,
                    "reason": "Negative amount test"
                }
            }
        })
        assert invalid_refund["id"] == 104
        assert invalid_refund["error"]["code"] == JSONRPC_INVALID_PARAMS

        # Close stdin and terminate cleanly
        proc.stdin.close()
        proc.wait(timeout=5)

        # Verify stderr captured diagnostic logs
        stderr_logs = proc.stderr.read()
        assert len(stderr_logs) > 0
        assert "[RUNNER]" in stderr_logs or "get_customer_record" in stderr_logs
