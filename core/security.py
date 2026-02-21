"""
JWT and password hashing.
"""
from datetime import datetime, timedelta
from typing import Optional
import bcrypt
from jose import JWTError, jwt
from passlib.context import CryptContext
from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from core.config import settings
import logging

logger = logging.getLogger(__name__)

# Password hashing
# Default to pbkdf2_sha256 to avoid passlib/bcrypt backend compatibility issues.
# Keep bcrypt in the context for backward compatibility with existing hashes.
pwd_context = CryptContext(
    schemes=["pbkdf2_sha256", "bcrypt"],
    deprecated="auto",
)

# OAuth2 scheme
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/api/auth/login")
oauth2_scheme_optional = OAuth2PasswordBearer(tokenUrl="/api/auth/login", auto_error=False)

# JWT settings
SECRET_KEY = settings.SECRET_KEY
ALGORITHM = settings.ALGORITHM or "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = settings.ACCESS_TOKEN_EXPIRE_MINUTES or 1440  # 24 hours

def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verify a password against a hash"""
    if not plain_password or not hashed_password:
        return False

    try:
        return pwd_context.verify(plain_password, hashed_password)
    except Exception as e:
        # Fallback path for legacy bcrypt hashes when passlib backend is incompatible.
        if hashed_password.startswith("$2"):
            try:
                return bcrypt.checkpw(
                    plain_password.encode("utf-8"),
                    hashed_password.encode("utf-8"),
                )
            except Exception as bcrypt_error:
                logger.error("bcrypt fallback verify failed: %s", bcrypt_error)
                return False

        logger.error("Password verification failed: %s", e)
        return False

def get_password_hash(password: str) -> str:
    """Hash a password"""
    # Force pbkdf2_sha256 for new hashes to avoid runtime issues with bcrypt backend drift.
    return pwd_context.hash(password, scheme="pbkdf2_sha256")

def create_access_token(data: dict, expires_delta: Optional[timedelta] = None) -> str:
    """
    Create JWT access token
    
    Args:
        data: Payload data (typically user_id, email, etc.)
        expires_delta: Optional expiration time delta
        
    Returns:
        JWT token string
    """
    to_encode = data.copy()
    
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    
    to_encode.update({"exp": expire, "iat": datetime.utcnow()})
    
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

def decode_access_token(token: str) -> dict:
    """
    Decode and verify JWT token
    
    Args:
        token: JWT token string
        
    Returns:
        Decoded payload
        
    Raises:
        HTTPException if token is invalid
    """
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        return payload
    except JWTError as e:
        logger.error(f"JWT decode error: {e}")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Could not validate credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )

def _credentials_exception() -> HTTPException:
    return HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )

def _user_from_payload(payload: dict) -> dict:
    user_id: str = payload.get("sub")
    if user_id is None:
        raise _credentials_exception()

    return {
        "user_id": user_id,
        "email": payload.get("email"),
        "subscription_tier": payload.get("subscription_tier", "free"),
    }

async def get_current_user(token: str = Depends(oauth2_scheme)) -> dict:
    """
    Dependency to get current authenticated user from token
    
    Args:
        token: JWT token from Authorization header
        
    Returns:
        User information from token payload
    """
    payload = decode_access_token(token)
    return _user_from_payload(payload)

async def get_current_active_user(token: Optional[str] = Depends(oauth2_scheme_optional)) -> dict:
    """Dependency to ensure user is active"""
    if not settings.ENABLE_AUTH:
        # Test-mode user when auth is disabled.
        return {
            "user_id": None,
            "email": "test@local",
            "subscription_tier": "enterprise",
            "auth_disabled": True,
        }

    if not token:
        raise _credentials_exception()

    payload = decode_access_token(token)
    return _user_from_payload(payload)
