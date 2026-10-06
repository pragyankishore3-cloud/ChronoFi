import pytest
from core.security import get_password_hash
from models.database import User

@pytest.fixture
async def test_user(db_session):
    # Create a user in the isolated test database
    user = User(
        name="Test User",
        email="test@chronofi.demo",
        hashed_password=get_password_hash("password123"),
        age=30,
        risk_tolerance="Moderate",
        base_salary=100000.0
    )
    db_session.add(user)
    await db_session.commit()
    await db_session.refresh(user)
    return user

@pytest.mark.asyncio
async def test_login_success(client, test_user):
    response = await client.post(
        "/api/auth/token",
        data={"username": "test@chronofi.demo", "password": "password123"}
    )
    assert response.status_code == 200
    data = response.json()
    assert "access_token" in data
    assert data["token_type"] == "bearer"

@pytest.mark.asyncio
async def test_login_failure(client, test_user):
    response = await client.post(
        "/api/auth/token",
        data={"username": "test@chronofi.demo", "password": "wrongpassword"}
    )
    assert response.status_code == 401

@pytest.mark.asyncio
async def test_protected_profile_route(client, test_user):
    # 1. Get token
    login_res = await client.post(
        "/api/auth/token",
        data={"username": "test@chronofi.demo", "password": "password123"}
    )
    token = login_res.json()["access_token"]
    
    # 2. Access protected route
    response = await client.get(
        "/api/profile/me",
        headers={"Authorization": f"Bearer {token}"}
    )
    assert response.status_code == 200
    assert response.json()["email"] == "test@chronofi.demo"

@pytest.mark.asyncio
async def test_unauthorized_profile_route(client):
    response = await client.get("/api/profile/me")
    assert response.status_code == 401
