import pytest
import asyncio
from httpx import AsyncClient
from api.main import app

@pytest.fixture(scope="session")
def event_loop():
    """Create event loop for async tests"""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()

@pytest.fixture
async def client():
    """Create test client"""
    async with AsyncClient(app=app, base_url="http://test") as ac:
        yield ac

@pytest.fixture
def test_user_token():
    """Create test user token"""
    from core.security import create_access_token
    return create_access_token(
        data={
            "sub": "test-user-id",
            "email": "test@example.com",
            "subscription_tier": "pro"
        }
    )