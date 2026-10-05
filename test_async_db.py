import asyncio
from models.database import AsyncSessionLocal, User, Asset
from repositories.base import BaseRepository

async def test_db():
    print("Testing Async Database Connection & Repository Pattern...")
    
    async with AsyncSessionLocal() as session:
        user_repo = BaseRepository(User, session)
        
        # Insert a user
        print("Creating User...")
        new_user = await user_repo.create({
            "name": "Enterprise Tester",
            "age": 30,
            "risk_tolerance": "Aggressive",
            "base_salary": 150000.0
        })
        print(f"Created User: {new_user.name} with ID {new_user.id}")
        
        # Read user back
        print("Fetching User...")
        fetched = await user_repo.get_by_id(new_user.id)
        assert fetched.name == "Enterprise Tester"
        print("Success! User fetched correctly via Async Repository.")

if __name__ == "__main__":
    asyncio.run(test_db())
