"""ChronoFi Financial Health & Variance Engine (CF-006).

Provides deterministic 0-100 composite Financial Health Scoring,
component breakdown (Emergency Runway, DTI, Savings Rate, Goal Funding),
and Actual vs Projected milestone variance tracking.
"""

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class HealthBreakdown(BaseModel):
    category: str
    score: float = Field(..., ge=0.0, le=25.0)
    max_score: float = 25.0
    status: str  # "EXCELLENT", "GOOD", "FAIR", "CRITICAL"
    metrics: Dict[str, Any]
    recommendation: Optional[str] = None


class FinancialHealthResult(BaseModel):
    composite_score: float = Field(..., ge=0.0, le=100.0)
    tier: str  # "EXCELLENT", "GOOD", "FAIR", "CRITICAL"
    breakdown: List[HealthBreakdown]
    actionable_priorities: List[str]
    net_worth_variance: Optional[Dict[str, Any]] = None


def compute_emergency_runway_score(
    liquid_cash: float,
    monthly_expenses: float,
) -> HealthBreakdown:
    """Computes emergency fund resilience out of 25 points.
    Target: 6 months of living expenses.
    """
    if monthly_expenses <= 0:
        months_runway = 12.0 if liquid_cash > 0 else 0.0
        score = 25.0 if liquid_cash > 0 else 0.0
    else:
        months_runway = round(liquid_cash / monthly_expenses, 2)
        score = min(25.0, round((months_runway / 6.0) * 25.0, 2))

    if months_runway >= 6.0:
        status = "EXCELLENT"
        rec = None
    elif months_runway >= 3.0:
        status = "GOOD"
        rec = f"Current runway is {months_runway} months. Consider adding ₹{round((6 - months_runway) * monthly_expenses):,} to reach the ideal 6-month buffer."
    elif months_runway >= 1.0:
        status = "FAIR"
        rec = f"Runway is only {months_runway} months. Prioritize building liquid savings before taking on new long-term commitments."
    else:
        status = "CRITICAL"
        rec = "Critical vulnerability: liquid emergency reserve is less than 1 month of living expenses."

    return HealthBreakdown(
        category="Emergency Runway",
        score=score,
        max_score=25.0,
        status=status,
        metrics={"liquid_cash": liquid_cash, "monthly_expenses": monthly_expenses, "months_runway": months_runway},
        recommendation=rec,
    )


def compute_dti_score(
    gross_monthly_income: float,
    monthly_debt_obligations: float,
) -> HealthBreakdown:
    """Computes Debt-to-Income (DTI) resilience out of 25 points.
    Ideal: <= 20%, Healthy: <= 35%, Critical: > 50%.
    """
    if gross_monthly_income <= 0:
        dti_pct = 100.0 if monthly_debt_obligations > 0 else 0.0
        score = 0.0 if monthly_debt_obligations > 0 else 25.0
    else:
        dti_pct = round((monthly_debt_obligations / gross_monthly_income) * 100.0, 2)
        if dti_pct <= 20.0:
            score = 25.0
        elif dti_pct <= 35.0:
            # 20% to 35% scales from 25.0 down to 15.0
            score = round(25.0 - ((dti_pct - 20.0) / 15.0) * 10.0, 2)
        elif dti_pct <= 50.0:
            # 35% to 50% scales from 15.0 down to 5.0
            score = round(15.0 - ((dti_pct - 35.0) / 15.0) * 10.0, 2)
        else:
            # >50% scales from 5.0 down to 0.0
            excess = min(30.0, dti_pct - 50.0)
            score = max(0.0, round(5.0 - (excess / 30.0) * 5.0, 2))

    if dti_pct <= 20.0:
        status = "EXCELLENT"
        rec = None
    elif dti_pct <= 35.0:
        status = "GOOD"
        rec = f"DTI is {dti_pct}%, which is manageable but leave little room for aggressive new leverage."
    elif dti_pct <= 50.0:
        status = "FAIR"
        rec = f"DTI is {dti_pct}%. Over a third of your income serves debt. Accelerate high-interest debt payoffs."
    else:
        status = "CRITICAL"
        rec = f"Critical DTI at {dti_pct}%. Immediate debt consolidation or aggressive avalanche paydown is strongly recommended."

    return HealthBreakdown(
        category="Debt-to-Income Ratio",
        score=score,
        max_score=25.0,
        status=status,
        metrics={"gross_income": gross_monthly_income, "monthly_debt": monthly_debt_obligations, "dti_pct": dti_pct},
        recommendation=rec,
    )


def compute_savings_rate_score(
    monthly_income: float,
    monthly_expenses: float,
    monthly_debt_obligations: float = 0.0,
) -> HealthBreakdown:
    """Computes Monthly Savings & Investment Rate out of 25 points.
    Target: >= 30% savings rate (25 pts), >= 15% (15-25 pts), Deficit (0 pts).
    """
    total_outflows = monthly_expenses + monthly_debt_obligations
    net_surplus = monthly_income - total_outflows

    if monthly_income <= 0:
        savings_rate_pct = 0.0
        score = 0.0
    else:
        savings_rate_pct = round((net_surplus / monthly_income) * 100.0, 2)
        if savings_rate_pct >= 30.0:
            score = 25.0
        elif savings_rate_pct >= 15.0:
            score = round(15.0 + ((savings_rate_pct - 15.0) / 15.0) * 10.0, 2)
        elif savings_rate_pct > 0.0:
            score = round((savings_rate_pct / 15.0) * 15.0, 2)
        else:
            score = 0.0

    if savings_rate_pct >= 30.0:
        status = "EXCELLENT"
        rec = None
    elif savings_rate_pct >= 15.0:
        status = "GOOD"
        rec = f"Healthy savings rate of {savings_rate_pct}%. Increasing monthly SIPs by 5-10% can compound significantly."
    elif savings_rate_pct > 0.0:
        status = "FAIR"
        rec = f"Low savings rate of {savings_rate_pct}%. Optimize discretionary spending to achieve a 20%+ investment rate."
    else:
        status = "CRITICAL"
        rec = f"Deficit spending: monthly outflows exceed income by ₹{abs(round(net_surplus)):,}."

    return HealthBreakdown(
        category="Savings & Surplus Rate",
        score=score,
        max_score=25.0,
        status=status,
        metrics={"monthly_income": monthly_income, "net_surplus": net_surplus, "savings_rate_pct": savings_rate_pct},
        recommendation=rec,
    )


def compute_goal_funding_score(
    goals: List[Dict[str, Any]],
) -> HealthBreakdown:
    """Computes Goal Funding Health out of 25 points.
    Evaluates current saved corpus against target corpus for active goals.
    """
    if not goals:
        return HealthBreakdown(
            category="Goal Funding Progress",
            score=20.0,
            max_score=25.0,
            status="GOOD",
            metrics={"total_goals": 0, "avg_funding_ratio_pct": 0.0},
            recommendation="No active financial goals configured. Define retirement, housing, or education targets to unlock full tracking.",
        )

    ratios = []
    for g in goals:
        target = float(g.get("target_amount") or 0.0)
        current = float(g.get("current_amount") or 0.0)
        if target > 0:
            ratios.append(min(1.0, current / target))

    avg_ratio = sum(ratios) / len(ratios) if ratios else 0.0
    avg_pct = round(avg_ratio * 100.0, 2)
    score = round(avg_ratio * 25.0, 2)

    if avg_pct >= 75.0:
        status = "EXCELLENT"
        rec = None
    elif avg_pct >= 40.0:
        status = "GOOD"
        rec = f"Goals are on average {avg_pct}% funded. Maintain disciplined SIP contributions."
    elif avg_pct >= 15.0:
        status = "FAIR"
        rec = f"Goals are only {avg_pct}% funded. Consider prioritizing essential tier goals before aspirational goals."
    else:
        status = "CRITICAL"
        rec = "Early stage goal funding: current corpus covers less than 15% of total goal capital requirements."

    return HealthBreakdown(
        category="Goal Funding Progress",
        score=score,
        max_score=25.0,
        status=status,
        metrics={"total_goals": len(goals), "avg_funding_ratio_pct": avg_pct},
        recommendation=rec,
    )


def calculate_actual_vs_projected_variance(
    actual_net_worth: float,
    projected_net_worth: float,
) -> Dict[str, Any]:
    """Calculates net worth variance against financial projections."""
    variance_amount = round(actual_net_worth - projected_net_worth, 2)
    if projected_net_worth > 0:
        variance_pct = round((variance_amount / projected_net_worth) * 100.0, 2)
    else:
        variance_pct = 0.0

    if variance_amount >= 0.05 * projected_net_worth:
        status = "AHEAD"
        narrative = f"You are ₹{abs(variance_amount):,} ({abs(variance_pct)}%) ahead of your projected milestone roadmap."
    elif variance_amount >= -0.05 * projected_net_worth:
        status = "ON_TRACK"
        narrative = f"You are tracking closely to your baseline plan within normal market fluctuations ({variance_pct}% variance)."
    else:
        status = "BEHIND"
        narrative = f"You are ₹{abs(variance_amount):,} ({abs(variance_pct)}%) behind your projected roadmap. Review asset allocations and increase savings rate."

    return {
        "actual_net_worth": actual_net_worth,
        "projected_net_worth": projected_net_worth,
        "variance_amount": variance_amount,
        "variance_pct": variance_pct,
        "status": status,
        "narrative": narrative,
    }


def compute_financial_health_score(
    liquid_cash: float,
    monthly_expenses: float,
    gross_monthly_income: float,
    monthly_debt_obligations: float,
    goals: Optional[List[Dict[str, Any]]] = None,
    actual_net_worth: Optional[float] = None,
    projected_net_worth: Optional[float] = None,
) -> FinancialHealthResult:
    """Calculates the complete 100-point composite Financial Health Score."""
    goals = goals or []

    runway_b = compute_emergency_runway_score(liquid_cash, monthly_expenses)
    dti_b = compute_dti_score(gross_monthly_income, monthly_debt_obligations)
    savings_b = compute_savings_rate_score(gross_monthly_income, monthly_expenses, monthly_debt_obligations)
    goal_b = compute_goal_funding_score(goals)

    composite = round(runway_b.score + dti_b.score + savings_b.score + goal_b.score, 1)
    composite = max(0.0, min(100.0, composite))

    if composite >= 85.0:
        tier = "EXCELLENT"
    elif composite >= 70.0:
        tier = "GOOD"
    elif composite >= 50.0:
        tier = "FAIR"
    else:
        tier = "CRITICAL"

    breakdowns = [runway_b, dti_b, savings_b, goal_b]
    priorities = [b.recommendation for b in breakdowns if b.recommendation is not None]

    variance_info = None
    if actual_net_worth is not None and projected_net_worth is not None:
        variance_info = calculate_actual_vs_projected_variance(actual_net_worth, projected_net_worth)

    return FinancialHealthResult(
        composite_score=composite,
        tier=tier,
        breakdown=breakdowns,
        actionable_priorities=priorities,
        net_worth_variance=variance_info,
    )
