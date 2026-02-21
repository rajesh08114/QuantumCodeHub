"""
Authentication endpoints.
"""
from fastapi import APIRouter, Depends, Form, HTTPException, status
from pydantic import BaseModel, EmailStr
from datetime import timedelta
from core.security import (
    verify_password,
    get_password_hash,
    create_access_token,
    ACCESS_TOKEN_EXPIRE_MINUTES,
    get_current_active_user
)
from core.config import settings
from core.database import get_db_connection, release_db_connection
import uuid
import logging

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/auth", tags=["Authentication"])

# Pydantic models
class UserRegister(BaseModel):
    email: EmailStr
    username: str
    password: str
    full_name: str = None

class Token(BaseModel):
    access_token: str
    token_type: str
    user: dict

class UserResponse(BaseModel):
    id: str
    email: str
    username: str
    full_name: str = None
    subscription_tier: str
    created_at: str

@router.post("/register", response_model=Token, status_code=status.HTTP_201_CREATED)
async def register(user_data: UserRegister):
    """
    Register a new user
    
    Creates a new user account with hashed password and returns JWT token.
    """
    conn = await get_db_connection()
    
    try:
        # Check if user already exists
        existing_user = await conn.fetchrow(
            "SELECT id FROM users WHERE email = $1 OR username = $2",
            user_data.email,
            user_data.username
        )
        
        if existing_user:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="User with this email or username already exists"
            )
        
        # Hash password
        hashed_password = get_password_hash(user_data.password)
        
        # Create user
        user = await conn.fetchrow(
            """
            INSERT INTO users (email, username, hashed_password, full_name, subscription_tier)
            VALUES ($1, $2, $3, $4, 'free')
            RETURNING id, email, username, full_name, subscription_tier, created_at
            """,
            user_data.email,
            user_data.username,
            hashed_password,
            user_data.full_name
        )
        
        # Create access token
        access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
        access_token = create_access_token(
            data={
                "sub": str(user["id"]),
                "email": user["email"],
                "subscription_tier": user["subscription_tier"]
            },
            expires_delta=access_token_expires
        )
        
        logger.info(f"New user registered: {user['email']}")
        
        return {
            "access_token": access_token,
            "token_type": "bearer",
            "user": {
                "id": str(user["id"]),
                "email": user["email"],
                "username": user["username"],
                "full_name": user["full_name"],
                "subscription_tier": user["subscription_tier"],
                "created_at": user["created_at"].isoformat()
            }
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Registration error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Registration failed"
        )
    finally:
        await release_db_connection(conn)

@router.post("/login", response_model=Token)
async def login(
    username: str = Form(...),
    password: str = Form(...),
    grant_type: str = Form(default="password"),
):
    """
    Login and get JWT token
    
    Validates credentials and returns access token for authenticated requests.
    """
    conn = await get_db_connection()
    
    try:
        if grant_type and grant_type != "password":
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Unsupported grant_type",
            )

        # Username form field is treated as email for login.
        user = await conn.fetchrow(
            """
            SELECT id, email, username, full_name, hashed_password, 
                   subscription_tier, is_active
            FROM users 
            WHERE email = $1
            """,
            username
        )
        
        if not user:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Incorrect email or password"
            )
        
        # Verify password
        if not verify_password(password, user["hashed_password"]):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Incorrect email or password"
            )
        
        # Check if user is active
        if not user["is_active"]:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Account is inactive"
            )
        
        # Update last login
        await conn.execute(
            "UPDATE users SET last_login = NOW() WHERE id = $1",
            user["id"]
        )
        
        # Create access token
        access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
        access_token = create_access_token(
            data={
                "sub": str(user["id"]),
                "email": user["email"],
                "subscription_tier": user["subscription_tier"]
            },
            expires_delta=access_token_expires
        )
        
        logger.info(f"User logged in: {user['email']}")
        
        return {
            "access_token": access_token,
            "token_type": "bearer",
            "user": {
                "id": str(user["id"]),
                "email": user["email"],
                "username": user["username"],
                "full_name": user["full_name"],
                "subscription_tier": user["subscription_tier"]
            }
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Login error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Login failed"
        )
    finally:
        await release_db_connection(conn)

@router.get("/me", response_model=UserResponse)
async def get_current_user_info(current_user: dict = Depends(get_current_active_user)):
    """
    Get current user information
    
    Returns detailed information about the authenticated user.
    """
    if not settings.ENABLE_AUTH and current_user.get("auth_disabled"):
        return {
            "id": "test-user",
            "email": current_user.get("email", "test@local"),
            "username": "test_user",
            "full_name": "Test User",
            "subscription_tier": current_user.get("subscription_tier", "enterprise"),
            "created_at": "1970-01-01T00:00:00Z"
        }

    conn = await get_db_connection()
    
    try:
        user = await conn.fetchrow(
            """
            SELECT id, email, username, full_name, subscription_tier, created_at
            FROM users 
            WHERE id = $1
            """,
            uuid.UUID(current_user["user_id"])
        )
        
        if not user:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="User not found"
            )
        
        return {
            "id": str(user["id"]),
            "email": user["email"],
            "username": user["username"],
            "full_name": user["full_name"],
            "subscription_tier": user["subscription_tier"],
            "created_at": user["created_at"].isoformat()
        }
        
    finally:
        await release_db_connection(conn)


