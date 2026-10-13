"""
Tests for ChronoFi Debt Acceleration Engine (Avalanche vs Snowball Optimizer).
"""

import pytest
from core.security import create_access_token
from engine.debt_engine import compare_debt_acceleration
from models.database import Liability, User


def test_empty_liabilities_graceful():
    """Verifies that an empty liabilities list is handled gracefully."""
    res = compare_debt_acceleration(liabilities=[], extra_monthly_prepayment=5000.0)
    assert res.total_initial_debt == 0.0
    assert res.baseline.months_to_debt_free == 0
    assert res.avalanche.months_to_debt_free == 0
    assert res.snowball.months_to_debt_free == 0
    assert res.recommended_strategy == "None"


def test_single_loan_prepayment_acceleration():
    """
    Verifies that for a single loan, adding prepayment cuts months and saves interest.
    """
    loans = [
        {
            "id": 1,
            "name": "Personal Loan",
            "principal": 500000.0,
            "annual_interest_rate": 0.12,
            "monthly_payment": 15000.0,
        }
    ]

    res = compare_debt_acceleration(liabilities=loans, extra_monthly_prepayment=10000.0)

    assert res.total_initial_debt == 500000.0
    assert res.baseline.months_to_debt_free > res.avalanche.months_to_debt_free
    assert res.avalanche.months_saved_vs_baseline > 0
    assert res.avalanche.interest_saved_vs_baseline > 0.0
    # For a single loan, avalanche and snowball behave identically
    assert res.avalanche.months_to_debt_free == res.snowball.months_to_debt_free
    assert res.avalanche.total_interest_paid == res.snowball.total_interest_paid


def test_avalanche_vs_snowball_differential():
    """
    Verifies mathematical superiority of Avalanche over Snowball in minimizing total interest:
    - High interest, large balance: ₹500,000 at 18% (Personal Loan)
    - Low interest, small balance: ₹100,000 at 8% (Bike Loan)
    """
    loans = [
        {
            "id": 1,
            "name": "High Rate Personal Loan",
            "principal": 500000.0,
            "annual_interest_rate": 0.18,
            "monthly_payment": 15000.0,
        },
        {
            "id": 2,
            "name": "Low Rate Bike Loan",
            "principal": 100000.0,
            "annual_interest_rate": 0.08,
            "monthly_payment": 5000.0,
        },
    ]

    res = compare_debt_acceleration(liabilities=loans, extra_monthly_prepayment=10000.0)

    # Avalanche pays less total interest than Snowball
    assert res.avalanche.total_interest_paid < res.snowball.total_interest_paid
    assert res.financial_savings_diff_avalanche_vs_snowball > 0.0
    assert res.recommended_strategy == "Debt Avalanche"

    # Snowball should pay off the bike loan first (month earlier than avalanche does)
    snowball_bike_payoff = next(m.payoff_month for m in res.snowball.loan_milestones if m.liability_name == "Low Rate Bike Loan")
    avalanche_bike_payoff = next(m.payoff_month for m in res.avalanche.loan_milestones if m.liability_name == "Low Rate Bike Loan")
    assert snowball_bike_payoff < avalanche_bike_payoff


def test_lump_sum_bonus_acceleration():
    """
    Verifies that a ₹100,000 lump sum bonus reduces baseline debt and saves interest.
    """
    loans = [
        {
            "id": 1,
            "name": "Credit Card Outstanding",
            "principal": 200000.0,
            "annual_interest_rate": 0.36,
            "monthly_payment": 10000.0,
        }
    ]

    res_without_lump = compare_debt_acceleration(liabilities=loans, extra_monthly_prepayment=0.0, lump_sum_prepayment=0.0)
    res_with_lump = compare_debt_acceleration(liabilities=loans, extra_monthly_prepayment=0.0, lump_sum_prepayment=100000.0)

    assert res_with_lump.avalanche.months_to_debt_free < res_without_lump.avalanche.months_to_debt_free
    assert res_with_lump.avalanche.total_interest_paid < res_without_lump.avalanche.total_interest_paid


@pytest.mark.asyncio
async def test_api_optimize_debt_endpoint(client, db_session):
    """
    Tests POST /api/debt/optimize endpoint with database liabilities and authenticated user.
    """
    user = User(
        name="Debt Test User",
        email="debt_user@chronofi.demo",
        hashed_password="pw",
        age=31,
        risk_tolerance="Moderate",
        base_salary=150000.0,
    )
    db_session.add(user)
    await db_session.commit()
    await db_session.refresh(user)

    l1 = Liability(
        user_id=user.id,
        name="Credit Card Debt",
        principal_remaining=150000.0,
        interest_rate=0.24,
        monthly_payment=8000.0,
    )
    l2 = Liability(
        user_id=user.id,
        name="Education Loan",
        principal_remaining=400000.0,
        interest_rate=0.09,
        monthly_payment=10000.0,
    )
    user.liabilities.append(l1)
    user.liabilities.append(l2)
    await db_session.commit()

    token = create_access_token(data={"sub": str(user.id)})
    headers = {"Authorization": f"Bearer {token}"}

    payload = {
        "extra_monthly_prepayment": 7000.0,
        "lump_sum_prepayment": 0.0,
    }

    res = await client.post("/api/debt/optimize", json=payload, headers=headers)
    assert res.status_code == 200
    data = res.json()

    assert data["total_initial_debt"] == 550000.0
    assert data["extra_monthly_prepayment"] == 7000.0
    assert "baseline" in data
    assert "avalanche" in data
    assert "snowball" in data
    assert data["avalanche"]["interest_saved_vs_baseline"] > 0
    assert len(data["actionable_takeaways"]) > 0
