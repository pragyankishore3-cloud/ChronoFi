import asyncio

from core.security import get_password_hash
import models.database as db_models
from models.database import Asset, Goal, Liability, User


async def seed_database():
    async with db_models.AsyncSessionLocal() as db:
        # Clear existing data
        await db.execute(Goal.__table__.delete())
        await db.execute(Liability.__table__.delete())
        await db.execute(Asset.__table__.delete())
        await db.execute(User.__table__.delete())
        await db.commit()

        dummy_password = get_password_hash("password123")

        # Persona 1: Arjun
        arjun = User(
            name="Arjun", 
            email="arjun@chronofi.demo", 
            hashed_password=dummy_password, 
            age=25, 
            risk_tolerance="Aggressive", 
            base_salary=80000.0
        )
        db.add(arjun)
        await db.commit()
        await db.refresh(arjun)
        
        db.add(Asset(user_id=arjun.id, name="Savings Account", asset_type="Cash", current_value=50000.0, expected_return_rate=0.04))
        db.add(Liability(user_id=arjun.id, name="Education Loan", principal_remaining=800000.0, interest_rate=0.11, monthly_payment=15000.0))
        db.add(Goal(user_id=arjun.id, name="Buy a Car", target_amount=1000000.0, target_years=3, priority="Important"))

        # Persona 2: Priya
        priya = User(
            name="Priya", 
            email="priya@chronofi.demo", 
            hashed_password=dummy_password,
            age=35, 
            risk_tolerance="Moderate", 
            base_salary=250000.0
        )
        db.add(priya)
        await db.commit()
        await db.refresh(priya)
        
        db.add(Asset(user_id=priya.id, name="Mutual Funds", asset_type="Equity", current_value=2500000.0, expected_return_rate=0.12))
        db.add(Asset(user_id=priya.id, name="Fixed Deposit", asset_type="Cash", current_value=1000000.0, expected_return_rate=0.07))
        db.add(Goal(user_id=priya.id, name="House Downpayment", target_amount=5000000.0, target_years=2, priority="Essential"))
        db.add(Goal(user_id=priya.id, name="Retirement", target_amount=50000000.0, target_years=25, priority="Essential"))

        await db.commit()
        print("Database seeded successfully with Arjun and Priya personas!")

if __name__ == "__main__":
    asyncio.run(seed_database())
