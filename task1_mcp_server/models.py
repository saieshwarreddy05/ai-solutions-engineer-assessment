"""
Task 1: Pydantic Validation Models and JSON-RPC Protocol Constants
"""

import re
from typing import Annotated, Any, Dict, Optional
from pydantic import BaseModel, Field, field_validator


# Standard JSON-RPC 2.0 and MCP Error Codes
JSONRPC_PARSE_ERROR = -32700
JSONRPC_INVALID_REQUEST = -32600
JSONRPC_METHOD_NOT_FOUND = -32601
JSONRPC_INVALID_PARAMS = -32602
JSONRPC_INTERNAL_ERROR = -32603

# Custom application error codes
ERR_CUSTOMER_NOT_FOUND = -32004
ERR_INSUFFICIENT_FUNDS = -32005

# Strict regex for customer ID: CUST- followed by exactly 5 alphanumeric uppercase characters/digits
CUSTOMER_ID_REGEX = r"^CUST-[A-Z0-9]{5}$"
CUSTOMER_ID_PATTERN = re.compile(CUSTOMER_ID_REGEX)


class CustomerRecord(BaseModel):
    """Domain model representing a customer entity."""
    customer_id: str
    name: str
    email: str
    balance: float
    status: str
    created_at: str


class GetCustomerRecordInput(BaseModel):
    """Input validation schema for get_customer_record tool."""
    customer_id: Annotated[
        str,
        Field(
            pattern=CUSTOMER_ID_REGEX,
            description="Unique customer identifier formatted strictly as CUST-XXXXX (e.g. CUST-10001)",
            examples=["CUST-10001", "CUST-A9B8C"]
        )
    ]

    @field_validator("customer_id")
    @classmethod
    def validate_customer_id_format(cls, v: str) -> str:
        if not CUSTOMER_ID_PATTERN.match(v):
            raise ValueError(
                f"Invalid customer_id '{v}'. Must strictly match format 'CUST-XXXXX' (5 uppercase alphanumeric characters after prefix)."
            )
        return v


class TriggerRefundInput(BaseModel):
    """Input validation schema for trigger_refund tool."""
    customer_id: Annotated[
        str,
        Field(
            pattern=CUSTOMER_ID_REGEX,
            description="Unique customer identifier formatted strictly as CUST-XXXXX",
            examples=["CUST-10001"]
        )
    ]
    amount: Annotated[
        float,
        Field(
            gt=0,
            description="Positive refund amount in currency units (must be > 0.00)",
            examples=[49.99, 120.00]
        )
    ]
    reason: Annotated[
        str,
        Field(
            min_length=10,
            description="Detailed business reason for refund (minimum 10 characters required)",
            examples=["Customer returned undamaged item within 30-day window"]
        )
    ]

    @field_validator("customer_id")
    @classmethod
    def validate_customer_id_format(cls, v: str) -> str:
        if not CUSTOMER_ID_PATTERN.match(v):
            raise ValueError(
                f"Invalid customer_id '{v}'. Must strictly match format 'CUST-XXXXX'."
            )
        return v

    @field_validator("amount")
    @classmethod
    def validate_positive_amount(cls, v: float) -> float:
        if v <= 0:
            raise ValueError(f"Refund amount must be strictly positive (> 0.0), received: {v}")
        return round(v, 2)

    @field_validator("reason")
    @classmethod
    def validate_reason_length(cls, v: str) -> str:
        trimmed = v.strip()
        if len(trimmed) < 10:
            raise ValueError(
                f"Refund reason must contain at least 10 non-whitespace characters, received: '{trimmed}' (length {len(trimmed)})"
            )
        return trimmed


class JSONRPCErrorResponse(BaseModel):
    """Standard JSON-RPC 2.0 Error structure."""
    jsonrpc: str = "2.0"
    id: Optional[Any] = None
    error: Dict[str, Any]


class JSONRPCSuccessResponse(BaseModel):
    """Standard JSON-RPC 2.0 Success structure."""
    jsonrpc: str = "2.0"
    id: Optional[Any] = None
    result: Dict[str, Any]
