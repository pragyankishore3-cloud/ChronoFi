from sqlalchemy.orm import Session
from models.database import SessionLocal, User, Asset, Liability, Goal, init_db

def seed_database():
    init_db()
    db = SessionLocal()
    
    # Clear existing data
    db.query(Goal).delete()
    db.query(Liability).delete()
    db.query(Asset).delete()
    db.query(User).delete()
    db.commit()

    # Persona 1: Arjun (25yo, high debt, low savings)
    arjun = User(name="Arjun", age=25, risk_tolerance="Aggressive", base_salary=80000.0)
    db.add(arjun)
    db.commit()
    db.refresh(arjun)
    
    db.add(Asset(user_id=arjun.id, name="Savings Account", asset_type="Cash", current_value=50000.0, expected_return_rate=0.04))
    db.add(Liability(user_id=arjun.id, name="Education Loan", principal_remaining=800000.0, interest_rate=0.11, monthly_payment=15000.0))
    db.add(Goal(user_id=arjun.id, name="Buy a Car", target_amount=1000000.0, target_years=3, priority="Important"))

    # Persona 2: Priya (35yo, buying a house, good savings)
    priya = User(name="Priya", age=35, risk_tolerance="Moderate", base_salary=250000.0)
    db.add(priya)
    db.commit()
    db.refresh(priya)
    
    db.add(Asset(user_id=priya.id, name="Mutual Funds", asset_type="Equity", current_value=2500000.0, expected_return_rate=0.12))
    db.add(Asset(user_id=priya.id, name="Fixed Deposit", asset_type="Cash", current_value=1000000.0, expected_return_rate=0.07))
    db.add(Goal(user_id=priya.id, name="House Downpayment", target_amount=5000000.0, target_years=2, priority="Essential"))
    db.add(Goal(user_id=priya.id, name="Retirement", target_amount=50000000.0, target_years=25, priority="Essential"))

    db.commit()
    db.close()
    print("Database seeded successfully with Arjun and Priya personas!")

if __name__ == "__main__":
    seed_database()
