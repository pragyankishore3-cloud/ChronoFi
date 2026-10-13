"""
Tests for India Capital Gains (LTCG / STCG) Goal Drawdown Engine (Budget 2024 / Finance Act 2024).
"""

import pytest
from core.security import create_access_token
from engine.tax_drawdown import (
    BUDGET_2024_EQUITY_LTCG_EXEMPTION,
    BUDGET_2024_EQUITY_LTCG_RATE,
    BUDGET_2024_EQUITY_STCG_RATE,
    HEALTH_AND_EDUCATION_CESS,
    calculate_goal_tax_drawdown,
    calculate_portfolio_goals_tax,
)
from models.database import Goal, User


def test_budget_2024_equity_ltcg_exemption():
    """
    Verifies that for a 100% equity goal with 5-year horizon:
    - ₹1,25,000 exemption is claimed under Budget 2024.
    - Gains above ₹1.25L are taxed at 12.5% + 4% cess (13.0%).
    """
    gross_corpus = 1000000.0  # ₹10 Lakh
    principal = 600000.0  # ₹6 Lakh
    # Total gains = ₹400,000
    # Exemption = ₹125,000
    # Taxable gains = ₹275,000
    # Tax = 275,000 * 0.125 * 1.04 = ₹35,750.0

    report = calculate_goal_tax_drawdown(
        goal_name="Long Term Equity Wealth",
        target_years=5,
        target_amount=700000.0,
        gross_projected_corpus=gross_corpus,
        total_principal_invested=principal,
        equity_pct=1.0,
        debt_pct=0.0,
        exempt_pct=0.0,
    )

    assert report.total_capital_gains == 400000.0
    assert report.equity_tax == 35750.0
    assert report.debt_tax == 0.0
    assert report.total_tax_liability == 35750.0
    assert report.net_post_tax_corpus == 1000000.0 - 35750.0
    assert report.is_fully_funded_gross is True
    assert report.is_fully_funded_post_tax is True


def test_equity_stcg_one_year_horizon():
    """
    Verifies that for horizon <= 1 year (STCG):
    - Zero LTCG exemption is allowed.
    - Flat 20% + 4% cess = 20.8% tax applies under Section 111A.
    """
    gross_corpus = 500000.0
    principal = 400000.0
    # Gains = ₹100,000
    # STCG Tax = 100,000 * 0.20 * 1.04 = ₹20,800.0

    report = calculate_goal_tax_drawdown(
        goal_name="Short Term Goal",
        target_years=1,
        target_amount=450000.0,
        gross_projected_corpus=gross_corpus,
        total_principal_invested=principal,
        equity_pct=1.0,
        debt_pct=0.0,
        exempt_pct=0.0,
    )

    assert report.total_capital_gains == 100000.0
    assert report.equity_tax == 20800.0
    assert report.total_tax_liability == 20800.0
    assert report.net_post_tax_corpus == 479200.0


def test_debt_marginal_tax_and_exempt_assets():
    """
    Verifies that:
    - Debt is taxed at marginal slab rate (30% + 4% cess = 31.2%).
    - Tax-exempt assets (PPF, SSY) incur ₹0 tax.
    """
    gross_corpus = 1000000.0
    principal = 600000.0  # Gains = 400,000
    # 50% Debt, 50% Tax-Exempt
    # Debt gains = 200,000 -> Tax = 200,000 * 0.30 * 1.04 = ₹62,400.0
    # Exempt gains = 200,000 -> Tax = ₹0

    report = calculate_goal_tax_drawdown(
        goal_name="Debt and PPF Portfolio",
        target_years=5,
        target_amount=800000.0,
        gross_projected_corpus=gross_corpus,
        total_principal_invested=principal,
        equity_pct=0.0,
        debt_pct=0.50,
        exempt_pct=0.50,
        marginal_tax_rate=0.30,
    )

    assert report.debt_tax == 62400.0
    assert report.equity_tax == 0.0
    assert report.total_tax_liability == 62400.0
    assert report.net_post_tax_corpus == 1000000.0 - 62400.0


def test_post_tax_shortfall_and_gross_up_advice():
    """
    Verifies that when gross corpus matches target but tax reduces it below target,
    a post-tax shortfall is detected and actionable advice is generated.
    """
    # Target in 5 years is 10 Lakh (no inflation for test simplicity)
    target = 1000000.0
    gross_corpus = 1000000.0
    principal = 400000.0  # 6 Lakh gains -> significant tax

    report = calculate_goal_tax_drawdown(
        goal_name="Dream House Down Payment",
        target_years=5,
        target_amount=target,
        gross_projected_corpus=gross_corpus,
        total_principal_invested=principal,
        equity_pct=0.60,
        debt_pct=0.30,
        exempt_pct=0.10,
        inflation_rate=0.0,
    )

    assert report.is_fully_funded_gross is True
    assert report.is_fully_funded_post_tax is False
    assert report.post_tax_shortfall > 0.0
    assert report.required_gross_target_to_meet_net > gross_corpus
    assert any("Tax Gap Alert" in tip for tip in report.actionable_tax_tips)


def test_portfolio_drawdown_aggregation():
    """
    Tests portfolio-level tax evaluation across multiple goals.
    """
    goals = [
        {"name": "Retirement", "target_years": 20, "target_amount": 10000000.0, "gross_projected_corpus": 12000000.0, "total_principal_invested": 5000000.0},
        {"name": "Kid College", "target_years": 10, "target_amount": 3000000.0, "gross_projected_corpus": 3500000.0, "total_principal_invested": 1800000.0},
    ]

    res = calculate_portfolio_goals_tax(goals=goals, risk_tolerance="Moderate")
    assert res["total_gross_corpus"] == 15500000.0
    assert res["total_tax_liability"] > 0.0
    assert res["total_net_post_tax_corpus"] < res["total_gross_corpus"]
    assert len(res["goals_evaluated"]) == 2


@pytest.mark.asyncio
async def test_api_goal_drawdown_endpoint(client, db_session):
    """
    Tests POST /api/tax/goal-drawdown with authenticated user.
    """
    user = User(
        name="Tax User",
        email="tax_user@chronofi.demo",
        hashed_password="pw",
        age=30,
        risk_tolerance="Aggressive",
        base_salary=200000.0,
    )
    db_session.add(user)
    await db_session.commit()
    await db_session.refresh(user)

    token = create_access_token(data={"sub": str(user.id)})
    headers = {"Authorization": f"Bearer {token}"}

    payload = {
        "goal_name": "Retirement Corpus",
        "target_years": 10,
        "target_amount": 5000000.0,
        "gross_projected_corpus": 6000000.0,
        "total_principal_invested": 3000000.0,
        "risk_tolerance": "Aggressive",
    }

    res = await client.post("/api/tax/goal-drawdown", json=payload, headers=headers)
    assert res.status_code == 200
    data = res.json()

    assert data["goal_name"] == "Retirement Corpus"
    assert "total_tax_liability" in data
    assert "net_post_tax_corpus" in data
    assert data["total_tax_liability"] > 0
    assert len(data["asset_breakdowns"]) == 3
