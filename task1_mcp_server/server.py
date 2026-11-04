"""
Task 1: Production MCP Server with Strict Validation & Transport Handling
Standard: Official Model Context Protocol (MCP) SDK with stdio transport.
Guarantees:
1. STDIO Isolation: stdout is exclusively reserved for JSON-RPC messages; all debug/system logs go to stderr.
2. Protocol Compliance: JSON-RPC 2.0 error mapping with standard error codes (-32602, -32601, etc.).
3. Input Validation: Pydantic schemas enforcing regex CUST-XXXXX, positive amounts, min_length=10.
"""

import sys
import json
import logging
import uuid
from datetime import datetime, timezone
from typing import Annotated, Any, Dict, List, Optional
from pydantic import Field, ValidationError

from mcp.server.mcpserver import MCPServer
import mcp.types as types

from task1_mcp_server.models import (
    GetCustomerRecordInput,
    TriggerRefundInput,
    JSONRPC_INVALID_PARAMS,
    JSONRPC_METHOD_NOT_FOUND,
    CUSTOMER_ID_REGEX,
)

# ---------------------------------------------------------------------------
# Strict STDIO Isolation: Configure logging exclusively to sys.stderr
# ---------------------------------------------------------------------------
stderr_handler = logging.StreamHandler(stream=sys.stderr)
stderr_handler.setFormatter(
    logging.Formatter("[%(asctime)s] [%(levelname)s] [Task1-MCPServer] %(message)s")
)
logger = logging.getLogger("mcp_server")
logger.setLevel(logging.DEBUG)
logger.handlers.clear()
logger.addHandler(stderr_handler)
logger.propagate = False

logger.info("Initializing MCP Server with stderr-only logging isolation...")

# ---------------------------------------------------------------------------
# In-Memory Mock Database
# ---------------------------------------------------------------------------
MOCK_CUSTOMERS: Dict[str, Dict[str, Any]] = {
    "CUST-10001": {
        "customer_id": "CUST-10001",
        "name": "Sarah Connor",
        "email": "sarah.connor@sky-shield.io",
        "tier": "Enterprise",
        "balance": 1500.00,
        "status": "active",
        "created_at": "2024-01-15T08:30:00Z",
    },
    "CUST-10002": {
        "customer_id": "CUST-10002",
        "name": "Miles Dyson",
        "email": "m.dyson@cyberdyne.org",
        "tier": "Pro",
        "balance": 350.25,
        "status": "active",
        "created_at": "2024-03-22T14:15:00Z",
    },
    "CUST-A1B2C": {
        "customer_id": "CUST-A1B2C",
        "name": "John Anderton",
        "email": "j.anderton@precrime.gov",
        "tier": "Standard",
        "balance": 75.50,
        "status": "active",
        "created_at": "2024-06-10T11:00:00Z",
    },
}

REFUND_LOG: List[Dict[str, Any]] = []


# ---------------------------------------------------------------------------
# Server Factory
# ---------------------------------------------------------------------------
def create_customer_mcp_server() -> MCPServer:
    """Build and configure the MCP server instance with strictly validated tools."""
    server = MCPServer(name="CustomerOperationsServer", version="1.0.0")

    @server.tool()
    def get_customer_record(
        customer_id: Annotated[
            str,
            Field(
                pattern=CUSTOMER_ID_REGEX,
                description="Unique customer identifier formatted strictly as CUST-XXXXX (e.g., CUST-10001)",
            ),
        ]
    ) -> str:
        """
        Retrieve full customer profile and billing record by customer_id.
        Validates customer_id strictly against format CUST-XXXXX.
        """
        logger.info(f"get_customer_record invoked for customer_id='{customer_id}'")

        # Explicit Pydantic validation pass to capture any edge-case formatting issues
        try:
            validated = GetCustomerRecordInput(customer_id=customer_id)
        except ValidationError as err:
            logger.warning(f"Schema validation failed for get_customer_record: {err}")
            error_details = [
                {"field": e["loc"][0] if e["loc"] else "customer_id", "message": e["msg"]}
                for e in err.errors()
            ]
            raise ValueError(
                json.dumps({
                    "error_code": JSONRPC_INVALID_PARAMS,
                    "error": "Invalid params: customer_id format must match CUST-XXXXX",
                    "details": error_details,
                })
            )

        cid = validated.customer_id
        customer = MOCK_CUSTOMERS.get(cid)
        if not customer:
            logger.warning(f"Customer '{cid}' not found in database.")
            return json.dumps({
                "found": False,
                "customer_id": cid,
                "message": f"No customer found with identifier {cid}",
            })

        logger.info(f"Customer '{cid}' record retrieved successfully.")
        return json.dumps({"found": True, "record": customer}, indent=2)

    @server.tool()
    def trigger_refund(
        customer_id: Annotated[
            str,
            Field(
                pattern=CUSTOMER_ID_REGEX,
                description="Unique customer identifier formatted strictly as CUST-XXXXX",
            ),
        ],
        amount: Annotated[
            float,
            Field(
                gt=0,
                description="Positive refund amount in currency units (must be > 0.00)",
            ),
        ],
        reason: Annotated[
            str,
            Field(
                min_length=10,
                description="Detailed reason for refund (minimum length 10 characters)",
            ),
        ],
    ) -> str:
        """
        Trigger a financial refund for a customer.
        Strictly enforces positive amount, valid CUST-XXXXX customer_id, and reason length >= 10.
        """
        logger.info(
            f"trigger_refund invoked for customer_id='{customer_id}', amount={amount}, reason='{reason[:20]}...'"
        )

        # Enforce strict Pydantic model validation
        try:
            validated = TriggerRefundInput(
                customer_id=customer_id,
                amount=amount,
                reason=reason,
            )
        except ValidationError as err:
            logger.warning(f"Schema validation failed for trigger_refund: {err}")
            error_details = [
                {
                    "field": ".".join(str(loc) for loc in e["loc"]),
                    "message": e["msg"],
                    "type": e["type"],
                }
                for e in err.errors()
            ]
            raise ValueError(
                json.dumps({
                    "error_code": JSONRPC_INVALID_PARAMS,
                    "error": "Invalid params for trigger_refund",
                    "details": error_details,
                })
            )

        cid = validated.customer_id
        ref_amount = validated.amount
        ref_reason = validated.reason

        customer = MOCK_CUSTOMERS.get(cid)
        if not customer:
            logger.warning(f"Refund rejected: customer '{cid}' not found.")
            return json.dumps({
                "status": "failed",
                "customer_id": cid,
                "message": f"Customer '{cid}' does not exist.",
            })

        # Process refund
        refund_tx_id = f"REF-{datetime.now(timezone.utc).strftime('%Y%m%d')}-{uuid.uuid4().hex[:8].upper()}"
        previous_balance = customer["balance"]
        customer["balance"] = round(previous_balance + ref_amount, 2)

        record = {
            "transaction_id": refund_tx_id,
            "customer_id": cid,
            "amount": ref_amount,
            "reason": ref_reason,
            "previous_balance": previous_balance,
            "new_balance": customer["balance"],
            "processed_at": datetime.now(timezone.utc).isoformat(),
            "status": "COMPLETED",
        }
        REFUND_LOG.append(record)

        logger.info(
            f"Refund {refund_tx_id} completed for customer {cid}. New balance: {customer['balance']}"
        )
        return json.dumps({"status": "success", "refund": record}, indent=2)

    return server


# ---------------------------------------------------------------------------
# Direct Protocol Processor (for standalone / test JSON-RPC dispatch)
# ---------------------------------------------------------------------------
class DirectMCPProtocolHandler:
    """
    Pure JSON-RPC 2.0 processor providing strict compliance with error codes
    -32700 (Parse error), -32600 (Invalid Request), -32601 (Method not found),
    -32602 (Invalid params).
    Ensures stdout writes are exclusively valid JSON-RPC frames.
    """

    def __init__(self, server: MCPServer):
        self.server = server

    async def handle_message(self, raw_line: str) -> Optional[str]:
        """Parse raw line and return JSON-RPC response string if request is non-notification."""
        # Check empty or whitespace
        stripped = raw_line.strip()
        if not stripped:
            return None

        # 1. Parse JSON
        try:
            payload = json.loads(stripped)
        except json.JSONDecodeError as e:
            logger.error(f"JSON Parse Error: {e}")
            return json.dumps({
                "jsonrpc": "2.0",
                "id": None,
                "error": {
                    "code": -32700,
                    "message": f"Parse error: {str(e)}",
                },
            })

        if not isinstance(payload, dict):
            return json.dumps({
                "jsonrpc": "2.0",
                "id": None,
                "error": {
                    "code": -32600,
                    "message": "Invalid Request: root payload must be a JSON object",
                },
            })

        req_id = payload.get("id")
        method = payload.get("method")
        params = payload.get("params", {})

        # Handle notifications (no id)
        is_notification = req_id is None

        # Check standard JSON-RPC header
        if payload.get("jsonrpc") != "2.0" or not method:
            if not is_notification:
                return json.dumps({
                    "jsonrpc": "2.0",
                    "id": req_id,
                    "error": {
                        "code": -32600,
                        "message": "Invalid Request: missing jsonrpc='2.0' or method attribute",
                    },
                })
            return None

        logger.debug(f"Received JSON-RPC method='{method}' (id={req_id})")

        # 2. Method Dispatching
        if method == "initialize":
            return json.dumps({
                "jsonrpc": "2.0",
                "id": req_id,
                "result": {
                    "protocolVersion": "2024-11-05",
                    "capabilities": {"tools": {"listChanged": False}},
                    "serverInfo": {
                        "name": self.server.name,
                        "version": self.server.version or "1.0.0",
                    },
                },
            })

        if method == "notifications/initialized":
            logger.info("Client completed initialization handshake.")
            return None

        if method == "ping":
            return json.dumps({"jsonrpc": "2.0", "id": req_id, "result": {}})

        if method == "tools/list":
            tools_list = await self.server.list_tools()
            tools_data = []
            for t in tools_list:
                tools_data.append({
                    "name": t.name,
                    "description": t.description or "",
                    "inputSchema": t.input_schema,
                })
            return json.dumps({
                "jsonrpc": "2.0",
                "id": req_id,
                "result": {"tools": tools_data},
            })

        if method == "tools/call":
            if not isinstance(params, dict):
                return json.dumps({
                    "jsonrpc": "2.0",
                    "id": req_id,
                    "error": {
                        "code": JSONRPC_INVALID_PARAMS,
                        "message": "Invalid params: params must be an object",
                    },
                })

            tool_name = params.get("name")
            arguments = params.get("arguments", {})

            if not tool_name:
                return json.dumps({
                    "jsonrpc": "2.0",
                    "id": req_id,
                    "error": {
                        "code": JSONRPC_INVALID_PARAMS,
                        "message": "Invalid params: missing tool 'name'",
                    },
                })

            # Check tool existence
            tools = await self.server.list_tools()
            known_tools = {t.name for t in tools}
            if tool_name not in known_tools:
                logger.warning(f"Unknown tool requested: '{tool_name}'")
                return json.dumps({
                    "jsonrpc": "2.0",
                    "id": req_id,
                    "error": {
                        "code": JSONRPC_METHOD_NOT_FOUND,
                        "message": f"Method not found: tool '{tool_name}' is not registered on this server",
                    },
                })

            # Execute tool with strict validation
            try:
                # Direct validation step before invoking
                if tool_name == "get_customer_record":
                    GetCustomerRecordInput.model_validate(arguments)
                elif tool_name == "trigger_refund":
                    TriggerRefundInput.model_validate(arguments)

                res = await self.server.call_tool(tool_name, arguments)

                # Format MCP Content Result
                if isinstance(res, list):
                    content = [{"type": item.type, "text": item.text} if hasattr(item, "text") else item for item in res]
                else:
                    content = [{"type": "text", "text": str(res)}]

                return json.dumps({
                    "jsonrpc": "2.0",
                    "id": req_id,
                    "result": {
                        "content": content,
                        "isError": False,
                    },
                })
            except ValidationError as val_err:
                logger.warning(f"Input validation rejected for '{tool_name}': {val_err}")
                error_details = [
                    {
                        "field": ".".join(str(loc) for loc in e["loc"]),
                        "message": e["msg"],
                        "type": e["type"],
                    }
                    for e in val_err.errors()
                ]
                return json.dumps({
                    "jsonrpc": "2.0",
                    "id": req_id,
                    "error": {
                        "code": JSONRPC_INVALID_PARAMS,
                        "message": f"Invalid params for tool '{tool_name}': {val_err.errors()[0]['msg']}",
                        "data": {"validation_errors": error_details},
                    },
                })
            except Exception as e:
                logger.error(f"Error executing tool '{tool_name}': {e}", exc_info=True)
                return json.dumps({
                    "jsonrpc": "2.0",
                    "id": req_id,
                    "error": {
                        "code": JSONRPC_INVALID_PARAMS,
                        "message": str(e),
                    },
                })

        # Method not found
        logger.warning(f"Unsupported JSON-RPC method received: '{method}'")
        return json.dumps({
            "jsonrpc": "2.0",
            "id": req_id,
            "error": {
                "code": JSONRPC_METHOD_NOT_FOUND,
                "message": f"Method not found: '{method}'",
            },
        })


def run_stdio_server():
    """Main execution loop for stdio transport with strict stdout/stderr isolation."""
    server = create_customer_mcp_server()
    logger.info("Starting Customer Operations MCP Server over STDIO...")
    # Run the official MCP stdio runner
    server.run(transport="stdio")


if __name__ == "__main__":
    run_stdio_server()
