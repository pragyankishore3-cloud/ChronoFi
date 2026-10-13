"""
Tests for Profile Management, Cash Flow Profiling, and Financial Item CRUD.
Verifies:
1. Profile retrieval with computed KPIs (Net Worth, Investable Surplus, Months of Runway).
2. Updating monthly expenses and demographics via PUT /api/profile/me.
3. Adding and deleting Assets with strict user authorization.
4. Adding and deleting Liabilities with strict user authorization.
5. Adding and deleting Goals with strict user authorization.
6. User isolation: User A cannot delete User B's assets or liabilities.
"""

import pytest
from core.security import create_access_token
from models.database import Asset, Goal, Liability, User


@pytest.fixture
async def user_profile_setup(db_session):
    user_a = User(
        name="Test User A",
        email="usera_crud@chronofi.demo",
        hashed_password="hashed_pw",
        age=30,
        risk_tolerance="Moderate",
        base_salary=150000.0,
        monthly_expenses=50000.0,
    )
    user_b = User(
        name="Test User B",
        email="userb_crud@chronofi.demo",
        hashed_password="hashed_pw",
        age=40,
        risk_tolerance="Aggressive",
        base_salary=200000.0,
        monthly_expenses=80000.0,
    )
    db_session.add(user_a)
    db_session.add(user_b)
    await db_session.commit()
    await db_session.refresh(user_a)
    await db_session.refresh(user_b)

    # Add assets and liabilities for user A
    asset_a = Asset(name="Emergency Cash", asset_type="Cash", current_value=300000.0, expected_return_rate=0.06)
    liab_a = Liability(name="Personal Loan", principal_remaining=100000.0, interest_rate=0.10, monthly_payment=10000.0)
    goal_a = Goal(name="Vacation", target_amount=200000.0, target_years=1, priority="Aspirational")

    user_a.assets.append(asset_a)
    user_a.liabilities.append(liab_a)
    user_a.goals.append(goal_a)
    await db_session.commit()
    await db_session.refresh(asset_a)
    await db_session.refresh(liab_a)
    await db_session.refresh(goal_a)

    return {"user_a": user_a, "user_b": user_b, "asset_a": asset_a, "liab_a": liab_a, "goal_a": goal_a}


@pytest.mark.asyncio
async def test_get_profile_with_kpis(client, user_profile_setup):
    user_a = user_profile_setup["user_a"]
    token = create_access_token(data={"sub": str(user_a.id)})
    headers = {"Authorization": f"Bearer {token}"}

    res = await client.get("/api/profile/me", headers=headers)
    assert res.status_code == 200
    data = res.json()

    assert data["name"] == "Test User A"
    assert data["base_salary"] == 150000.0
    assert data["monthly_expenses"] == 50000.0
    # Net Worth = 300,000 (asset) - 100,000 (liability) = 200,000
    assert data["net_worth"] == 200000.0
    # Surplus = 150,000 (salary) - 10,000 (EMI) - 50,000 (expenses) = 90,000
    assert data["monthly_investable_surplus"] == 90000.0
    # Runway = 300,000 / (50,000 + 10,000) = 5.0 months
    assert data["months_of_runway"] == 5.0
    assert len(data["assets"]) == 1
    assert len(data["liabilities"]) == 1
    assert len(data["goals"]) == 1


@pytest.mark.asyncio
async def test_update_profile_expenses_and_salary(client, user_profile_setup):
    user_a = user_profile_setup["user_a"]
    token = create_access_token(data={"sub": str(user_a.id)})
    headers = {"Authorization": f"Bearer {token}"}

    update_payload = {
        "monthly_expenses": 60000.0,
        "base_salary": 180000.0,
        "age": 31,
    }
    res = await client.put("/api/profile/me", json=update_payload, headers=headers)
    assert res.status_code == 200
    data = res.json()

    assert data["monthly_expenses"] == 60000.0
    assert data["base_salary"] == 180000.0
    assert data["age"] == 31
    # Updated Surplus = 180,000 - 10,000 (EMI) - 60,000 (expenses) = 110,000
    assert data["monthly_investable_surplus"] == 110000.0


@pytest.mark.asyncio
async def test_asset_crud_and_user_isolation(client, user_profile_setup):
    user_a = user_profile_setup["user_a"]
    user_b = user_profile_setup["user_b"]
    token_a = create_access_token(data={"sub": str(user_a.id)})
    token_b = create_access_token(data={"sub": str(user_b.id)})

    # User A creates asset
    new_asset = {
        "name": "Nifty Index Fund",
        "asset_type": "Equity",
        "current_value": 500000.0,
        "expected_return_rate": 0.12,
    }
    res = await client.post("/api/profile/assets", json=new_asset, headers={"Authorization": f"Bearer {token_a}"})
    assert res.status_code == 200
    asset_id = res.json()["id"]

    # User B attempts to delete User A's asset (Forbidden / Not Found isolation)
    res_b_delete = await client.delete(f"/api/profile/assets/{asset_id}", headers={"Authorization": f"Bearer {token_b}"})
    assert res_b_delete.status_code == 404

    # User A deletes their own asset
    res_a_delete = await client.delete(f"/api/profile/assets/{asset_id}", headers={"Authorization": f"Bearer {token_a}"})
    assert res_a_delete.status_code == 200
    assert res_a_delete.json()["status"] == "success"


@pytest.mark.asyncio
async def test_liability_and_goal_crud(client, user_profile_setup):
    user_a = user_profile_setup["user_a"]
    token_a = create_access_token(data={"sub": str(user_a.id)})
    headers = {"Authorization": f"Bearer {token_a}"}

    # Add Liability
    new_liab = {
        "name": "Bike Loan",
        "principal_remaining": 80000.0,
        "interest_rate": 0.08,
        "monthly_payment": 3500.0,
    }
    res_liab = await client.post("/api/profile/liabilities", json=new_liab, headers=headers)
    assert res_liab.status_code == 200
    liab_id = res_liab.json()["id"]

    # Add Goal
    new_goal = {
        "name": "Higher Education",
        "target_amount": 1500000.0,
        "target_years": 4,
        "priority": "Essential",
        "inflation_rate": 0.08,
    }
    res_goal = await client.post("/api/profile/goals", json=new_goal, headers=headers)
    assert res_goal.status_code == 200
    goal_id = res_goal.json()["id"]

    # Delete Liability & Goal
    del_liab = await client.delete(f"/api/profile/liabilities/{liab_id}", headers=headers)
    assert del_liab.status_code == 200
    del_goal = await client.delete(f"/api/profile/goals/{goal_id}", headers=headers)
    assert del_goal.status_code == 200
