"""
Unit and Integration Tests for Liquidity Shock & Emergency Runway Stress Testing Engine.
"""

import pytest
from core.security import create_access_token
from engine.stress_engine import run_liquidity_stress_test
from models.database import Asset, Liability, User


def test_job_loss_liquidity_shock_waterfall():
    """
    Simulates a 6-month job loss for a user with ₹50k expenses and ₹20k EMI (burn rate ₹70k).
    Has ₹200k in Cash (Tier 1), ₹300k in FD (Tier 2), and ₹1M in Equity (Tier 4).
    Expects:
    - Tier 1 drains first (~2.85 months)
    - Tier 2 taps second with 1% friction penalty
    - Stays solvent through 6 months shock with ₹28k left in Tier 2 and ₹1M untouched in Equity.
    """
    assets = [
        {"name": "Savings Account", "asset_type": "Cash", "current_value": 200000.0},
        {"name": "Bank FD", "asset_type": "FD", "current_value": 300000.0},
        {"name": "Nifty Index Fund", "asset_type": "Equity", "current_value": 1000000.0},
    ]

    res = run_liquidity_stress_test(
        monthly_expenses=50000.0,
        base_salary=150000.0,
        existing_liabilities_monthly=20000.0,
        assets=assets,
        shock_type="job_loss",
        shock_duration_months=6,
        income_reduction_pct=100.0,
        lump_sum_cost=0.0,
    )

    assert res.monthly_burn_rate == 70000.0
    assert res.is_solvent_through_shock is True
    assert res.insolvency_month is None
    assert res.resilience_rating in ["Adequate", "Robust"]

    t1 = next(t for t in res.asset_tiers if "Tier 1" in t.tier_name)
    t2 = next(t for t in res.asset_tiers if "Tier 2" in t.tier_name)
    t4 = next(t for t in res.asset_tiers if "Tier 4" in t.tier_name)

    # Cash fully liquidated
    assert t1.liquidated_value == 200000.0
    assert t1.remaining_value == 0.0

    # FD partially liquidated with 1% friction
    assert t2.liquidated_value > 0.0
    assert t2.friction_cost > 0.0

    # Equity remained untouched (not forced to fire-sale)
    assert t4.liquidated_value == 0.0
    assert t4.remaining_value == 1000000.0


def test_insolvency_critical_rating():
    """
    Simulates a user with high burn rate (₹100k/mo) and only ₹150k total liquid assets.
    Job loss of 6 months should cause insolvency at Month 2.
    """
    assets = [
        {"name": "Emergency Cash", "asset_type": "Cash", "current_value": 150000.0},
    ]

    res = run_liquidity_stress_test(
        monthly_expenses=80000.0,
        base_salary=200000.0,
        existing_liabilities_monthly=20000.0,
        assets=assets,
        shock_type="job_loss",
        shock_duration_months=6,
        income_reduction_pct=100.0,
    )

    assert res.is_solvent_through_shock is False
    assert res.insolvency_month == 2
    assert res.resilience_rating == "Critical"
    assert res.survival_runway_months <= 2.0


def test_lump_sum_medical_emergency():
    """
    Simulates a ₹300,000 upfront medical emergency shock with 30% salary cut for 3 months.
    """
    assets = [
        {"name": "Cash Buffer", "asset_type": "Liquid", "current_value": 500000.0},
    ]

    res = run_liquidity_stress_test(
        monthly_expenses=40000.0,
        base_salary=100000.0,
        existing_liabilities_monthly=10000.0,
        assets=assets,
        shock_type="medical_emergency",
        shock_duration_months=3,
        income_reduction_pct=30.0,
        lump_sum_cost=300000.0,
    )

    assert res.is_solvent_through_shock is True
    assert res.total_shock_cost >= 280000.0


def test_invalid_stress_inputs():
    """Test boundary and negative inputs."""
    with pytest.raises(ValueError, match="Monthly expenses cannot be negative"):
        run_liquidity_stress_test(monthly_expenses=-1000, base_salary=100000, existing_liabilities_monthly=0, assets=[])

    with pytest.raises(ValueError, match="Shock duration must be at least 1 month"):
        run_liquidity_stress_test(monthly_expenses=50000, base_salary=100000, existing_liabilities_monthly=0, assets=[], shock_duration_months=0)


@pytest.mark.asyncio
async def test_api_simulate_liquidity_stress(client, db_session):
    """
    Tests POST /api/stress/simulate with authenticated user.
    """
    user = User(
        name="Stress Test User",
        email="stress_test@chronofi.demo",
        hashed_password="pw",
        age=34,
        risk_tolerance="Moderate",
        base_salary=160000.0,
        monthly_expenses=45000.0,
    )
    db_session.add(user)
    await db_session.commit()
    await db_session.refresh(user)

    a1 = Asset(user_id=user.id, name="Savings", asset_type="Cash", current_value=250000.0, expected_return_rate=0.04)
    l1 = Liability(user_id=user.id, name="Auto Loan", principal_remaining=300000.0, interest_rate=0.09, monthly_payment=15000.0)
    user.assets.append(a1)
    user.liabilities.append(l1)
    await db_session.commit()
    await db_session.refresh(a1)
    await db_session.refresh(l1)

    token = create_access_token(data={"sub": str(user.id)})
    headers = {"Authorization": f"Bearer {token}"}

    payload = {
        "shock_type": "job_loss",
        "shock_duration_months": 4,
        "income_reduction_pct": 100.0,
        "lump_sum_cost": 0.0,
    }

    res = await client.post("/api/stress/simulate", json=payload, headers=headers)
    assert res.status_code == 200
    data = res.json()

    assert data["monthly_burn_rate"] == 60000.0  # 45k + 15k
    assert "survival_runway_months" in data
    assert "asset_tiers" in data
    assert len(data["asset_tiers"]) == 4
    assert len(data["monthly_trajectory"]) >= 4
    assert len(data["actionable_advice"]) > 0
