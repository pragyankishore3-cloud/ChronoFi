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


@pytest.mark.asyncio
async def test_profile_contract_validation_and_currency(client, user_profile_setup):
    """Verifies logical data contract compliance: currency defaults and non-negative constraints."""
    user_a = user_profile_setup["user_a"]
    token = create_access_token(data={"sub": str(user_a.id)})
    headers = {"Authorization": f"Bearer {token}"}

    # Profile query contains ISO currency
    res = await client.get("/api/profile/me", headers=headers)
    assert res.status_code == 200
    profile_data = res.json()
    assert profile_data["currency"] == "INR"
    assert profile_data["assets"][0]["currency"] == "INR"
    assert profile_data["liabilities"][0]["currency"] == "INR"
    assert profile_data["goals"][0]["currency"] == "INR"

    # Reject negative asset value
    bad_asset = {
        "name": "Negative Asset",
        "asset_type": "Equity",
        "current_value": -500.0,
        "expected_return_rate": 0.08,
    }
    res_bad_asset = await client.post("/api/profile/assets", json=bad_asset, headers=headers)
    assert res_bad_asset.status_code == 422

    # Reject negative liability principal
    bad_liab = {
        "name": "Negative Debt",
        "principal_remaining": -1000.0,
        "interest_rate": 0.1,
        "monthly_payment": 500.0,
    }
    res_bad_liab = await client.post("/api/profile/liabilities", json=bad_liab, headers=headers)
    assert res_bad_liab.status_code == 422

    # Reject invalid target_years (< 1)
    bad_goal = {
        "name": "Zero Year Goal",
        "target_amount": 100000.0,
        "target_years": 0,
        "priority": "Essential",
    }
    res_bad_goal = await client.post("/api/profile/goals", json=bad_goal, headers=headers)
    assert res_bad_goal.status_code == 422


@pytest.mark.asyncio
async def test_update_financial_records_and_isolation(client, user_profile_setup):
    """Verifies that authenticated users can edit their own assets, liabilities, and goals, with strict isolation."""
    user_a = user_profile_setup["user_a"]
    user_b = user_profile_setup["user_b"]
    token_a = create_access_token(data={"sub": str(user_a.id)})
    token_b = create_access_token(data={"sub": str(user_b.id)})
    headers_a = {"Authorization": f"Bearer {token_a}"}
    headers_b = {"Authorization": f"Bearer {token_b}"}

    asset_a = user_profile_setup["asset_a"]
    liab_a = user_profile_setup["liab_a"]
    goal_a = user_profile_setup["goal_a"]

    # 1. User A updates their asset
    update_asset_payload = {"name": "Updated Emergency Reserve", "current_value": 350000.0}
    res_asset = await client.put(f"/api/profile/assets/{asset_a.id}", json=update_asset_payload, headers=headers_a)
    assert res_asset.status_code == 200
    assert res_asset.json()["name"] == "Updated Emergency Reserve"
    assert res_asset.json()["current_value"] == 350000.0

    # User B cannot update User A's asset
    res_asset_b = await client.put(f"/api/profile/assets/{asset_a.id}", json=update_asset_payload, headers=headers_b)
    assert res_asset_b.status_code == 404

    # 2. User A updates their liability
    update_liab_payload = {"principal_remaining": 85000.0, "monthly_payment": 9000.0}
    res_liab = await client.put(f"/api/profile/liabilities/{liab_a.id}", json=update_liab_payload, headers=headers_a)
    assert res_liab.status_code == 200
    assert res_liab.json()["principal_remaining"] == 85000.0
    assert res_liab.json()["monthly_payment"] == 9000.0

    # User B cannot update User A's liability
    res_liab_b = await client.put(f"/api/profile/liabilities/{liab_a.id}", json=update_liab_payload, headers=headers_b)
    assert res_liab_b.status_code == 404

    # 3. User A updates their goal
    update_goal_payload = {"target_amount": 250000.0, "priority": "Essential", "target_years": 2}
    res_goal = await client.put(f"/api/profile/goals/{goal_a.id}", json=update_goal_payload, headers=headers_a)
    assert res_goal.status_code == 200
    assert res_goal.json()["target_amount"] == 250000.0
    assert res_goal.json()["priority"] == "Essential"
    assert res_goal.json()["target_years"] == 2

    # User B cannot update User A's goal
    res_goal_b = await client.put(f"/api/profile/goals/{goal_a.id}", json=update_goal_payload, headers=headers_b)
    assert res_goal_b.status_code == 404


