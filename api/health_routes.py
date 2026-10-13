"""ChronoFi Financial Health Routes (CF-006).

Provides live composite Financial Health scoring and actual vs projected variance tracking.
"""

from typing import Any, Dict, List, Optional
from fastapi import APIRouter, Depends
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select

from api import deps
from engine.health_engine import (
    FinancialHealthResult,
    compute_financial_health_score,
)
from models import database

router = APIRouter()


class CustomHealthRequest(BaseModel):
    liquid_cash: float
    monthly_expenses: float
    gross_monthly_income: float
    monthly_debt_obligations: float
    goals: Optional[List[Dict[str, Any]]] = None
    actual_net_worth: Optional[float] = None
    projected_net_worth: Optional[float] = None


@router.get("/score", response_model=FinancialHealthResult)
async def get_user_financial_health(
    projected_net_worth: Optional[float] = None,
    current_user: database.User = Depends(deps.get_current_user),
    db: AsyncSession = Depends(deps.get_db),
):
    """Calculates the authenticated user's real-time 0-100 composite Financial Health Score."""
    # Retrieve user's live financial data
    assets_res = await db.execute(
        select(database.Asset).where(database.Asset.user_id == current_user.id)
    )
    assets = assets_res.scalars().all()

    liabilities_res = await db.execute(
        select(database.Liability).where(database.Liability.user_id == current_user.id)
    )
    liabilities = liabilities_res.scalars().all()

    goals_res = await db.execute(
        select(database.Goal).where(database.Goal.user_id == current_user.id)
    )
    goals = goals_res.scalars().all()

    # Aggregate figures
    # Liquid cash is cash/savings/bank accounts
    liquid_cash = sum(
        a.current_value
        for a in assets
        if "cash" in a.name.lower() or "bank" in a.name.lower() or "emergency" in a.name.lower() or a.asset_type.lower() == "cash"
    )
    # If no specific liquid tag, default to minimum 15% of total liquid-like assets or 0
    total_assets = sum(a.current_value for a in assets)
    if liquid_cash == 0 and total_assets > 0:
        liquid_cash = min(total_assets, 100000.0)

    total_liabilities = sum(l.principal_remaining for l in liabilities)
    monthly_debt = sum(l.monthly_payment for l in liabilities)
    actual_net_worth = total_assets - total_liabilities

    gross_monthly_income = (current_user.base_salary or 0.0) / 12.0
    monthly_expenses = current_user.monthly_expenses or 0.0

    goal_dicts = [
        {"target_amount": g.target_amount, "current_amount": g.current_amount, "title": g.title}
        for g in goals
    ]

    return compute_financial_health_score(
        liquid_cash=liquid_cash,
        monthly_expenses=monthly_expenses,
        gross_monthly_income=gross_monthly_income,
        monthly_debt_obligations=monthly_debt,
        goals=goal_dicts,
        actual_net_worth=actual_net_worth,
        projected_net_worth=projected_net_worth,
    )


@router.post("/evaluate", response_model=FinancialHealthResult)
async def evaluate_custom_health(
    req: CustomHealthRequest,
    current_user: database.User = Depends(deps.get_current_user),
):
    """Evaluates financial health score for custom or what-if hypothetical figures."""
    return compute_financial_health_score(
        liquid_cash=req.liquid_cash,
        monthly_expenses=req.monthly_expenses,
        gross_monthly_income=req.gross_monthly_income,
        monthly_debt_obligations=req.monthly_debt_obligations,
        goals=req.goals or [],
        actual_net_worth=req.actual_net_worth,
        projected_net_worth=req.projected_net_worth,
    )
