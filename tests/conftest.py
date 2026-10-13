import os

os.environ['GEMINI_API_KEY'] = os.environ.get('GEMINI_API_KEY', 'dummy_key')
import pytest
from httpx import ASGITransport, AsyncClient
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.pool import StaticPool

from api.deps import get_db
from main import app
from models.database import Base

# Use in-memory SQLite for blazing fast tests that don't pollute the dev DB
TEST_DATABASE_URL = "sqlite+aiosqlite:///:memory:"

engine = create_async_engine(
    TEST_DATABASE_URL,
    connect_args={"check_same_thread": False},
    poolclass=StaticPool,
)
TestingSessionLocal = async_sessionmaker(
    autocommit=False, autoflush=False, expire_on_commit=False, bind=engine, class_=AsyncSession
)

@pytest.fixture(autouse=True)
async def setup_db():
    import models.database as db_mod
    orig_session = db_mod.AsyncSessionLocal
    db_mod.AsyncSessionLocal = TestingSessionLocal
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    yield
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)
    db_mod.AsyncSessionLocal = orig_session

@pytest.fixture
async def db_session():
    async with TestingSessionLocal() as session:
        yield session

@pytest.fixture
async def client(db_session):
    async def override_get_db():
        yield db_session

    app.dependency_overrides[get_db] = override_get_db
    # Using ASGITransport as recommended by httpx 0.28.0+ for FastAPI async testing
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        yield ac
    app.dependency_overrides.clear()
