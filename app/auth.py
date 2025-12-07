"""
JWT-based authentication for RAGify MVP.

Provides login endpoint, token generation, and authentication middleware.
"""

import os
import jwt
import bcrypt
from datetime import datetime, timedelta
from typing import Optional, Dict
from fastapi import HTTPException, Security, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
import logging

logger = logging.getLogger(__name__)

# JWT Configuration
SECRET_KEY = os.getenv("JWT_SECRET_KEY", "your-secret-key-change-in-production")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 60 * 24  # 24 hours

# HTTP Bearer token scheme
security = HTTPBearer()


# ============================================================================
# In-memory user store (for MVP - replace with database in production)
# ============================================================================

# Format: {username: {password_hash: bytes, tenant_id: str, name: str}}
_USERS: Dict[str, dict] = {}


def _hash_password(password: str) -> bytes:
    """Hash a password using bcrypt."""
    return bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt())


def _verify_password(password: str, hashed: bytes) -> bool:
    """Verify a password against its hash."""
    return bcrypt.checkpw(password.encode('utf-8'), hashed)


def init_users():
    """Initialize default users for MVP demo."""
    global _USERS
    
    _USERS = {
        "demo": {
            "password_hash": _hash_password("demo123"),
            "tenant_id": "default",
            "name": "Demo User"
        },
        "acme_admin": {
            "password_hash": _hash_password("acme123"),
            "tenant_id": "acme",
            "name": "ACME Administrator"
        },
        "finance_user": {
            "password_hash": _hash_password("finance123"),
            "tenant_id": "finance",
            "name": "Finance User"
        },
    }
    
    logger.info("Initialized %d default users", len(_USERS))


# Initialize users on module load
init_users()


def create_access_token(username: str, tenant_id: str) -> str:
    """
    Create a JWT access token for a user.
    
    Args:
        username: Username
        tenant_id: Tenant ID for this user
        
    Returns:
        JWT token string
    """
    expire = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    to_encode = {
        "sub": username,
        "tenant_id": tenant_id,
        "exp": expire
    }
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt


def verify_token(token: str) -> dict:
    """
    Verify and decode a JWT token.
    
    Args:
        token: JWT token string
        
    Returns:
        Decoded token payload
        
    Raises:
        HTTPException: If token is invalid or expired
    """
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        return payload
    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=401, detail="Token has expired")
    except jwt.InvalidTokenError:
        raise HTTPException(status_code=401, detail="Invalid token")


def authenticate_user(username: str, password: str) -> Optional[dict]:
    """
    Authenticate a user with username and password.
    
    Args:
        username: Username
        password: Password
        
    Returns:
        User dict if authenticated, None otherwise
    """
    user = _USERS.get(username)
    if not user:
        return None
    
    if not _verify_password(password, user["password_hash"]):
        return None
    
    return {
        "username": username,
        "tenant_id": user["tenant_id"],
        "name": user["name"]
    }


async def get_current_user(credentials: HTTPAuthorizationCredentials = Security(security)) -> dict:
    """
    FastAPI dependency to get current authenticated user from JWT token.
    
    Usage:
        @app.get("/protected")
        async def protected_route(user: dict = Depends(get_current_user)):
            # user dict contains: username, tenant_id, name
            pass
    
    Raises:
        HTTPException: If token is missing or invalid
    """
    token = credentials.credentials
    payload = verify_token(token)
    
    username = payload.get("sub")
    tenant_id = payload.get("tenant_id")
    
    if not username or not tenant_id:
        raise HTTPException(status_code=401, detail="Invalid token payload")
    
    # Verify user still exists in store
    user = _USERS.get(username)
    if not user:
        raise HTTPException(status_code=401, detail="User not found")
    
    return {
        "username": username,
        "tenant_id": tenant_id,
        "name": user["name"]
    }
