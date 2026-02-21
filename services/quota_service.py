"""
Rate limiting service.
"""
from datetime import date, datetime
from typing import Dict
import redis.asyncio as redis
from core.config import settings
from core.database import get_db_connection, release_db_connection
import uuid
import logging

logger = logging.getLogger(__name__)

class QuotaService:
    """Service for managing user quotas and rate limiting"""
    
    # Tier limits
    TIER_LIMITS = {
        "free": {
            "daily": 50,
            "monthly": 1000,
            "concurrent": 2
        },
        "pro": {
            "daily": 500,
            "monthly": 10000,
            "concurrent": 10
        },
        "team": {
            "daily": 2000,
            "monthly": 50000,
            "concurrent": 50
        },
        "enterprise": {
            "daily": -1,  # Unlimited
            "monthly": -1,
            "concurrent": 100
        }
    }
    
    def __init__(self):
        self.redis_client = None
    
    async def connect(self):
        """Connect to Redis"""
        try:
            self.redis_client = redis.from_url(
                settings.REDIS_URL,
                encoding="utf-8",
                decode_responses=True
            )
        except Exception as e:
            logger.error(f"Redis connection failed: {e}")
    
    async def check_quota(self, user_id: str, subscription_tier: str) -> Dict:
        """
        Check if user has remaining quota
        
        Returns:
            Dict with quota status and limits
        """
        limits = self.TIER_LIMITS.get(subscription_tier, self.TIER_LIMITS["free"])
        
        # Get current usage from database
        conn = await get_db_connection()
        try:
            user_data = await conn.fetchrow(
                """
                SELECT daily_request_count, monthly_request_count, 
                       last_request_reset
                FROM users 
                WHERE id = $1
                """,
                uuid.UUID(user_id)
            )
            
            if not user_data:
                return {
                    "allowed": False,
                    "reason": "User not found"
                }
            
            # Reset daily counter if needed
            today = date.today()
            if user_data["last_request_reset"] != today:
                await conn.execute(
                    """
                    UPDATE users 
                    SET daily_request_count = 0, 
                        last_request_reset = $1
                    WHERE id = $2
                    """,
                    today,
                    uuid.UUID(user_id)
                )
                daily_count = 0
            else:
                daily_count = user_data["daily_request_count"]
            
            monthly_count = user_data["monthly_request_count"]
            
            # Check limits
            daily_limit = limits["daily"]
            monthly_limit = limits["monthly"]
            
            # -1 means unlimited
            if daily_limit != -1 and daily_count >= daily_limit:
                return {
                    "allowed": False,
                    "reason": "Daily limit exceeded",
                    "daily_used": daily_count,
                    "daily_limit": daily_limit,
                    "resets_at": "midnight UTC"
                }
            
            if monthly_limit != -1 and monthly_count >= monthly_limit:
                return {
                    "allowed": False,
                    "reason": "Monthly limit exceeded",
                    "monthly_used": monthly_count,
                    "monthly_limit": monthly_limit
                }
            
            return {
                "allowed": True,
                "daily_used": daily_count,
                "daily_limit": daily_limit,
                "daily_remaining": daily_limit - daily_count if daily_limit != -1 else -1,
                "monthly_used": monthly_count,
                "monthly_limit": monthly_limit,
                "monthly_remaining": monthly_limit - monthly_count if monthly_limit != -1 else -1
            }
            
        finally:
            await release_db_connection(conn)
    
    async def increment_usage(self, user_id: str):
        """Increment user's request counters"""
        conn = await get_db_connection()
        try:
            await conn.execute(
                """
                UPDATE users 
                SET daily_request_count = daily_request_count + 1,
                    monthly_request_count = monthly_request_count + 1
                WHERE id = $1
                """,
                uuid.UUID(user_id)
            )
        finally:
            await release_db_connection(conn)
    
    async def check_rate_limit(
        self,
        user_id: str,
        endpoint: str,
        max_requests: int = 10,
        window_seconds: int = 60
    ) -> bool:
        """
        Check rate limit for specific endpoint
        
        Uses sliding window algorithm via Redis
        """
        if not self.redis_client:
            return True  # Allow if Redis unavailable
        
        try:
            key = f"ratelimit:{user_id}:{endpoint}"
            now = datetime.utcnow().timestamp()
            window_start = now - window_seconds
            
            # Remove old entries
            await self.redis_client.zremrangebyscore(key, 0, window_start)
            
            # Count requests in window
            count = await self.redis_client.zcard(key)
            
            if count >= max_requests:
                return False
            
            # Add current request
            await self.redis_client.zadd(key, {str(now): now})
            await self.redis_client.expire(key, window_seconds)
            
            return True
            
        except Exception as e:
            logger.error(f"Rate limit check error: {e}")
            return True  # Allow on error

# Singleton instance
quota_service = QuotaService()
