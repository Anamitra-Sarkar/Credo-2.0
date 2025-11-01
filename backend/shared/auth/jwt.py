"""
Authentication utilities for JWT token management.
"""
from datetime import datetime, timedelta
from typing import Optional, Dict, Any
from jose import JWTError, jwt
from passlib.context import CryptContext
from fastapi import HTTPException, Security, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from shared.config import settings


# Password hashing context
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# HTTP Bearer token security
security = HTTPBearer()


class AuthError(HTTPException):
    """Custom authentication error."""
    def __init__(self, detail: str):
        super().__init__(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=detail,
            headers={"WWW-Authenticate": "Bearer"},
        )


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verify a password against its hash."""
    return pwd_context.verify(plain_password, hashed_password)


def get_password_hash(password: str) -> str:
    """Hash a password."""
    return pwd_context.hash(password)


def create_access_token(data: Dict[str, Any], expires_delta: Optional[timedelta] = None) -> str:
    """
    Create a JWT access token.
    
    Args:
        data: Payload data to encode in the token
        expires_delta: Optional expiration time delta
    
    Returns:
        Encoded JWT token string
    """
    to_encode = data.copy()
    
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=settings.jwt_access_token_expire_minutes)
    
    to_encode.update({
        "exp": expire,
        "iat": datetime.utcnow(),
        "type": "access"
    })
    
    encoded_jwt = jwt.encode(
        to_encode,
        settings.jwt_secret_key,
        algorithm=settings.jwt_algorithm
    )
    
    return encoded_jwt


def create_refresh_token(data: Dict[str, Any]) -> str:
    """
    Create a JWT refresh token.
    
    Args:
        data: Payload data to encode in the token
    
    Returns:
        Encoded JWT refresh token string
    """
    to_encode = data.copy()
    expire = datetime.utcnow() + timedelta(days=settings.jwt_refresh_token_expire_days)
    
    to_encode.update({
        "exp": expire,
        "iat": datetime.utcnow(),
        "type": "refresh"
    })
    
    encoded_jwt = jwt.encode(
        to_encode,
        settings.jwt_secret_key,
        algorithm=settings.jwt_algorithm
    )
    
    return encoded_jwt


def decode_token(token: str) -> Dict[str, Any]:
    """
    Decode and validate a JWT token.
    
    Args:
        token: JWT token string
    
    Returns:
        Decoded token payload
    
    Raises:
        AuthError: If token is invalid or expired
    """
    try:
        payload = jwt.decode(
            token,
            settings.jwt_secret_key,
            algorithms=[settings.jwt_algorithm]
        )
        return payload
    except JWTError as e:
        raise AuthError(detail=f"Could not validate credentials: {str(e)}")


async def get_current_user_id(
    credentials: HTTPAuthorizationCredentials = Security(security)
) -> int:
    """
    Dependency to get current authenticated user ID from JWT token.
    
    Args:
        credentials: HTTP Authorization credentials
    
    Returns:
        User ID from token
    
    Raises:
        AuthError: If token is invalid or missing user_id
    """
    token = credentials.credentials
    payload = decode_token(token)
    
    # Verify token type
    if payload.get("type") != "access":
        raise AuthError(detail="Invalid token type")
    
    # Extract user ID
    user_id: Optional[int] = payload.get("sub")
    if user_id is None:
        raise AuthError(detail="Token missing user ID")
    
    try:
        return int(user_id)
    except ValueError:
        raise AuthError(detail="Invalid user ID in token")


async def get_optional_user_id(
    credentials: Optional[HTTPAuthorizationCredentials] = Security(security)
) -> Optional[int]:
    """
    Dependency to optionally get current user ID (for public endpoints).
    Returns None if no valid token is provided.
    """
    if not credentials:
        return None
    
    try:
        return await get_current_user_id(credentials)
    except AuthError:
        return None


def create_token_pair(user_id: int, email: str, role: str) -> Dict[str, str]:
    """
    Create both access and refresh tokens for a user.
    
    Args:
        user_id: User's ID
        email: User's email
        role: User's role
    
    Returns:
        Dictionary containing access_token and refresh_token
    """
    token_data = {
        "sub": str(user_id),
        "email": email,
        "role": role
    }
    
    access_token = create_access_token(token_data)
    refresh_token = create_refresh_token({"sub": str(user_id)})
    
    return {
        "access_token": access_token,
        "refresh_token": refresh_token,
        "token_type": "bearer"
    }


async def verify_refresh_token(token: str) -> int:
    """
    Verify a refresh token and return the user ID.
    
    Args:
        token: Refresh token string
    
    Returns:
        User ID from token
    
    Raises:
        AuthError: If token is invalid or not a refresh token
    """
    payload = decode_token(token)
    
    if payload.get("type") != "refresh":
        raise AuthError(detail="Invalid token type")
    
    user_id: Optional[int] = payload.get("sub")
    if user_id is None:
        raise AuthError(detail="Token missing user ID")
    
    try:
        return int(user_id)
    except ValueError:
        raise AuthError(detail="Invalid user ID in token")
