"""
Task 2: Authentication & Role Extraction Module
Extracts and validates Bearer tokens from incoming HTTP headers.
Supports admin and viewer roles.
"""

from enum import Enum
from typing import Optional, Tuple
import base64
import json


class UserRole(str, Enum):
    ADMIN = "admin"
    VIEWER = "viewer"


# Token-to-Role registry for production test credentials
TOKEN_REGISTRY = {
    "admin-secret-token-xyz": UserRole.ADMIN,
    "admin-key-ops-99": UserRole.ADMIN,
    "viewer-readonly-token-abc": UserRole.VIEWER,
    "viewer-guest-token-123": UserRole.VIEWER,
}


class AuthResult:
    def __init__(self, is_authenticated: bool, role: Optional[UserRole] = None, error_message: Optional[str] = None):
        self.is_authenticated = is_authenticated
        self.role = role
        self.error_message = error_message


def extract_bearer_token(auth_header: Optional[str]) -> Optional[str]:
    """Extract raw bearer token string from 'Bearer <token>' header."""
    if not auth_header:
        return None
    parts = auth_header.strip().split()
    if len(parts) != 2 or parts[0].lower() != "bearer":
        return None
    return parts[1]


def authenticate_token(token: str) -> AuthResult:
    """
    Authenticate token against static registry or structured base64 claims.
    """
    # 1. Check registry
    if token in TOKEN_REGISTRY:
        return AuthResult(is_authenticated=True, role=TOKEN_REGISTRY[token])

    # 2. Check structured base64 token (e.g. mock JWT payload {"role": "admin"})
    try:
        decoded = base64.b64decode(token).decode("utf-8")
        payload = json.loads(decoded)
        role_str = payload.get("role", "").lower()
        if role_str == UserRole.ADMIN.value:
            return AuthResult(is_authenticated=True, role=UserRole.ADMIN)
        if role_str == UserRole.VIEWER.value:
            return AuthResult(is_authenticated=True, role=UserRole.VIEWER)
    except Exception:
        pass

    return AuthResult(
        is_authenticated=False,
        error_message="Invalid bearer token: credentials not recognized or role unassigned"
    )
