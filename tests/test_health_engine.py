import pytest
from core.security import create_access_token

from engine.health_engine import (
    calculate_actual_vs_projected_variance,
    compute_dti_score,
    compute_emergency_runway_score,
    compute_financial_health_score,
    compute_goal_funding_score,
    compute_savings_rate_score,
)


def test_emergency_runway_scoring():
    # 6+ months runway -> 25 points, EXCELLENT
    res_exc = compute_emergency_runway_score(liquid_cash=600000.0, monthly_expenses=100000.0)
    assert res_exc.score == 25.0
    assert res_exc.status == "EXCELLENT"
    assert res_exc.recommendation is None

    # 3 months runway -> 12.5 points, GOOD
    res_good = compute_emergency_runway_score(liquid_cash=300000.0, monthly_expenses=100000.0)
    assert res_good.score == 12.5
    assert res_good.status == "GOOD"
    assert res_good.recommendation is not None

    # 0.5 month runway -> CRITICAL
    res_crit = compute_emergency_runway_score(liquid_cash=50000.0, monthly_expenses=100000.0)
    assert res_crit.score < 5.0
    assert res_crit.status == "CRITICAL"


def test_dti_scoring():
    # 15% DTI -> 25 points, EXCELLENT
    res_exc = compute_dti_score(gross_monthly_income=100000.0, monthly_debt_obligations=15000.0)
    assert res_exc.score == 25.0
    assert res_exc.status == "EXCELLENT"

    # 30% DTI -> GOOD
    res_good = compute_dti_score(gross_monthly_income=100000.0, monthly_debt_obligations=30000.0)
    assert 15.0 <= res_good.score < 25.0
    assert res_good.status == "GOOD"

    # 60% DTI -> CRITICAL
    res_crit = compute_dti_score(gross_monthly_income=100000.0, monthly_debt_obligations=60000.0)
    assert res_crit.score < 5.0
    assert res_crit.status == "CRITICAL"


def test_savings_rate_scoring():
    # 35% savings rate -> 25 points, EXCELLENT
    res_exc = compute_savings_rate_score(monthly_income=100000.0, monthly_expenses=65000.0)
    assert res_exc.score == 25.0
    assert res_exc.status == "EXCELLENT"

    # 20% savings rate -> GOOD
    res_good = compute_savings_rate_score(monthly_income=100000.0, monthly_expenses=80000.0)
    assert 15.0 <= res_good.score < 25.0
    assert res_good.status == "GOOD"

    # Deficit spending -> 0 points, CRITICAL
    res_crit = compute_savings_rate_score(monthly_income=100000.0, monthly_expenses=120000.0)
    assert res_crit.score == 0.0
    assert res_crit.status == "CRITICAL"


def test_goal_funding_scoring():
    # Empty goals
    res_empty = compute_goal_funding_score([])
    assert res_empty.score == 20.0

    # 80% funded goals -> EXCELLENT
    res_funded = compute_goal_funding_score([
        {"target_amount": 1000000.0, "current_amount": 800000.0},
    ])
    assert res_funded.score == 20.0
    assert res_funded.status == "EXCELLENT"


def test_actual_vs_projected_variance():
    # Ahead: Actual 12L vs Projected 10L
    v_ahead = calculate_actual_vs_projected_variance(1200000.0, 1000000.0)
    assert v_ahead["status"] == "AHEAD"
    assert v_ahead["variance_amount"] == 200000.0
    assert v_ahead["variance_pct"] == 20.0

    # On Track: Actual 10.2L vs Projected 10L (within 5%)
    v_on_track = calculate_actual_vs_projected_variance(1020000.0, 1000000.0)
    assert v_on_track["status"] == "ON_TRACK"

    # Behind: Actual 8L vs Projected 10L
    v_behind = calculate_actual_vs_projected_variance(800000.0, 1000000.0)
    assert v_behind["status"] == "BEHIND"
    assert v_behind["variance_amount"] == -200000.0


def test_composite_health_score_calculation():
    result = compute_financial_health_score(
        liquid_cash=600000.0,
        monthly_expenses=80000.0,
        gross_monthly_income=150000.0,
        monthly_debt_obligations=20000.0,
        goals=[{"target_amount": 2000000.0, "current_amount": 1600000.0}],
        actual_net_worth=5000000.0,
        projected_net_worth=4500000.0,
    )
    assert result.composite_score >= 80.0
    assert result.tier in ["EXCELLENT", "GOOD"]
    assert len(result.breakdown) == 4
    assert result.net_worth_variance is not None
    assert result.net_worth_variance["status"] == "AHEAD"


@pytest.mark.asyncio
async def test_health_evaluate_api(client, test_user):
    token = create_access_token(data={"sub": str(test_user.id)})
    headers = {"Authorization": f"Bearer {token}"}
    payload = {
        "liquid_cash": 400000.0,
        "monthly_expenses": 60000.0,
        "gross_monthly_income": 120000.0,
        "monthly_debt_obligations": 15000.0,
        "goals": [{"target_amount": 1000000.0, "current_amount": 500000.0}],
        "actual_net_worth": 3000000.0,
        "projected_net_worth": 3000000.0,
    }
    response = await client.post("/api/health/evaluate", json=payload, headers=headers)
    assert response.status_code == 200
    data = response.json()
    assert "composite_score" in data
    assert "tier" in data
    assert len(data["breakdown"]) == 4
    assert data["net_worth_variance"]["status"] == "ON_TRACK"


@pytest.mark.asyncio
async def test_health_score_live_user_api(client, test_user):
    token = create_access_token(data={"sub": str(test_user.id)})
    headers = {"Authorization": f"Bearer {token}"}
    response = await client.get("/api/health/score?projected_net_worth=4000000", headers=headers)
    assert response.status_code == 200
    data = response.json()
    assert "composite_score" in data
    assert data["composite_score"] >= 0.0
    assert "tier" in data
    assert "breakdown" in data

