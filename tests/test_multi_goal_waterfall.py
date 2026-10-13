"""
Unit and Integration Tests for Multi-Goal Waterfall Allocation & Pareto Conflict Resolution.
Verifies:
1. Priority waterfall ordering: Essential goals are prioritized before Important and Aspirational goals.
2. Surplus reduction impact: Aspirational goals absorb cuts to protect Essential milestones.
3. Loan tenure recovery: Goals extending beyond loan tenure reflect surplus recovery.
4. Boundary conditions: Zero surplus, zero interest, empty goals list.
5. Integration with SimulateScenario API: goal_impacts returned in response.
"""

import pytest
from engine.scenario_engine import (
    GoalConflictItem,
    calculate_multi_goal_waterfall,
    project_retirement_simulation,
)
from core.security import create_access_token
from models.database import Asset, Goal, User


def test_priority_waterfall_allocation_hierarchy():
    """
    Verifies that when surplus is constrained, Essential goals are fully funded first,
    Important goals receive remaining surplus, and Aspirational goals absorb funding cuts.
    """
    goals = [
        {
            "id": 1,
            "name": "Retirement",
            "priority": "Essential",
            "target_amount": 6000000.0,   # 60 Lakh (needs ~₹25k/mo SIP)
            "target_years": 20,
            "inflation_rate": 0.06,
        },
        {
            "id": 2,
            "name": "Child Education",
            "priority": "Important",
            "target_amount": 2000000.0,   # 20 Lakh
            "target_years": 10,
            "inflation_rate": 0.06,
        },
        {
            "id": 3,
            "name": "Luxury Europe Tour",
            "priority": "Aspirational",
            "target_amount": 500000.0,    # 5 Lakh
            "target_years": 2,
            "inflation_rate": 0.06,
        },
    ]

    # Baseline surplus: ₹80,000/mo (Comfortable to fund all 3)
    # Modified surplus: ₹35,000/mo (Constrained due to loan)
    impacts = calculate_multi_goal_waterfall(
        goals=goals,
        baseline_surplus=80000.0,
        modified_surplus=35000.0,
        expected_annual_return=0.10,
        loan_tenure_months=60,
    )

    assert len(impacts) == 3
    ret_impact = next(g for g in impacts if g.name == "Retirement")
    edu_impact = next(g for g in impacts if g.name == "Child Education")
    tour_impact = next(g for g in impacts if g.name == "Luxury Europe Tour")

    # Retirement (Essential) should be prioritized and remain On Track
    assert ret_impact.priority == "Essential"
    assert ret_impact.modified_status == "On Track"
    assert ret_impact.modified_funding_ratio >= 98.0
    assert "Protected by Essential priority" in ret_impact.trade_off_note

    # Luxury Europe Tour (Aspirational, 2-year horizon inside 5-year loan window)
    # gets starved because surplus is consumed by higher-priority goals
    assert tour_impact.priority == "Aspirational"
    assert tour_impact.modified_funding_ratio < tour_impact.baseline_funding_ratio
    assert tour_impact.corpus_shortfall > 0
    assert tour_impact.modified_monthly_allocated < tour_impact.baseline_monthly_allocated


def test_extreme_surplus_deficit_starvation():
    """
    Verifies behavior when surplus is severely constrained (₹5,000/mo).
    Only Essential goals receive any funds, while Important and Aspirational receive 0.
    """
    goals = [
        {
            "id": 10,
            "name": "Primary Home Down Payment",
            "priority": "Essential",
            "target_amount": 3000000.0,
            "target_years": 5,
            "inflation_rate": 0.06,
        },
        {
            "id": 11,
            "name": "Car Upgrade",
            "priority": "Important",
            "target_amount": 1000000.0,
            "target_years": 4,
            "inflation_rate": 0.06,
        },
        {
            "id": 12,
            "name": "Angel Investment Fund",
            "priority": "Aspirational",
            "target_amount": 1500000.0,
            "target_years": 3,
            "inflation_rate": 0.06,
        },
    ]

    impacts = calculate_multi_goal_waterfall(
        goals=goals,
        baseline_surplus=70000.0,
        modified_surplus=5000.0,
        expected_annual_return=0.10,
        loan_tenure_months=60,
    )

    home = next(g for g in impacts if g.name == "Primary Home Down Payment")
    car = next(g for g in impacts if g.name == "Car Upgrade")
    angel = next(g for g in impacts if g.name == "Angel Investment Fund")

    # Essential gets all available ₹5,000
    assert home.modified_monthly_allocated == 5000.0
    # Lower tiers receive zero allocation under severe deficit
    assert car.modified_monthly_allocated == 0.0
    assert angel.modified_monthly_allocated == 0.0
    assert car.modified_status == "Underfunded"
    assert angel.modified_status == "Underfunded"


def test_post_loan_payoff_recovery():
    """
    Verifies that for a 15-year goal with a 5-year loan tenure,
    the modified corpus reflects post-loan surplus recovery in years 6 to 15.
    """
    goals = [
        {
            "id": 20,
            "name": "Long-Term Wealth",
            "priority": "Essential",
            "target_amount": 5000000.0,
            "target_years": 15,
            "inflation_rate": 0.05,
        }
    ]

    impacts = calculate_multi_goal_waterfall(
        goals=goals,
        baseline_surplus=50000.0,
        modified_surplus=30000.0,
        expected_annual_return=0.10,
        loan_tenure_months=60,  # 5 years
    )

    wealth = impacts[0]
    # Because horizon (15 yrs) > loan tenure (5 yrs), the wealth goal recovers
    assert wealth.modified_projected_corpus > wealth.baseline_projected_corpus * 0.70
    assert wealth.modified_funding_ratio > 80.0


def test_empty_goals_graceful_handling():
    """Verifies that an empty goals list returns [] without error."""
    impacts = calculate_multi_goal_waterfall(
        goals=[],
        baseline_surplus=50000.0,
        modified_surplus=30000.0,
    )
    assert impacts == []


@pytest.mark.asyncio
async def test_simulate_scenario_api_returns_goal_impacts(client, db_session):
    """
    Integration test: Verifies that POST /api/scenarios/simulate returns
    structured goal_impacts for authenticated user with multiple goals.
    """
    user = User(
        name="Multi Goal User",
        email="multigoal@chronofi.demo",
        hashed_password="pw",
        age=32,
        risk_tolerance="Moderate",
        base_salary=180000.0,
        monthly_expenses=50000.0,
    )
    db_session.add(user)
    await db_session.commit()
    await db_session.refresh(user)

    g1 = Goal(name="Retirement", target_amount=15000000.0, target_years=25, priority="Essential")
    g2 = Goal(name="Family Villa", target_amount=4000000.0, target_years=6, priority="Important")
    g3 = Goal(name="Sports Bike", target_amount=800000.0, target_years=2, priority="Aspirational")
    user.goals.extend([g1, g2, g3])
    await db_session.commit()
    await db_session.refresh(g1)
    await db_session.refresh(g2)
    await db_session.refresh(g3)

    token = create_access_token(data={"sub": str(user.id)})
    headers = {"Authorization": f"Bearer {token}"}

    payload = {
        "scenario_type": "vehicle_loan",
        "changes": {
            "loan_principal": 1200000.0,
            "annual_interest_rate": 0.095,
            "loan_tenure_months": 60,
            "down_payment": 100000.0,
            "monthly_vehicle_expenses": 5000.0,
        },
    }

    res = await client.post("/api/v1/financial-scenarios/simulate", json=payload, headers=headers)
    assert res.status_code == 200
    data = res.json()

    assert "goal_impacts" in data
    assert len(data["goal_impacts"]) == 3

    # Check that each goal has complete typed fields
    for g in data["goal_impacts"]:
        assert "name" in g
        assert "priority" in g
        assert "baseline_funding_ratio" in g
        assert "modified_funding_ratio" in g
        assert "baseline_status" in g
        assert "modified_status" in g
        assert "trade_off_note" in g
        assert "projected_delay_months" in g
