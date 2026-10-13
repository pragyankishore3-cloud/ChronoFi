"""
ChronoFi Debt Optimization API (Avalanche vs Snowball Prepayment Engine).
"""

from typing import Any, Dict, List, Optional
from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from api import deps
from engine.debt_engine import DebtOptimizerResult, compare_debt_acceleration
from models import database

router = APIRouter()


class DebtOptimizeRequest(BaseModel):
    extra_monthly_prepayment: float = Field(default=5000.0, ge=0.0, description="Additional monthly prepayment amount")
    lump_sum_prepayment: float = Field(default=0.0, ge=0.0, description="Immediate upfront lump-sum payment (e.g. bonus)")
    custom_liabilities: Optional[List[Dict[str, Any]]] = Field(
        default=None,
        description="Optional custom liabilities. If omitted, uses authenticated user's active liabilities from database.",
    )


class LoanMilestoneResponse(BaseModel):
    liability_id: Optional[int]
    liability_name: str
    initial_principal: float
    annual_interest_rate: float
    payoff_month: int
    total_interest_paid: float


class DebtTrajectoryPointResponse(BaseModel):
    month: int
    total_remaining_balance: float
    total_interest_paid_cumulative: float
    total_principal_paid_cumulative: float
    active_loans_count: int


class StrategySummaryResponse(BaseModel):
    strategy_name: str
    description: str
    months_to_debt_free: int
    total_interest_paid: float
    total_principal_paid: float
    total_amount_paid: float
    interest_saved_vs_baseline: float
    months_saved_vs_baseline: int
    loan_milestones: List[LoanMilestoneResponse]
    trajectory: List[DebtTrajectoryPointResponse]


class DebtOptimizeResponse(BaseModel):
    total_initial_debt: float
    total_monthly_contractual_emi: float
    extra_monthly_prepayment: float
    lump_sum_prepayment: float
    baseline: StrategySummaryResponse
    avalanche: StrategySummaryResponse
    snowball: StrategySummaryResponse
    recommended_strategy: str
    financial_savings_diff_avalanche_vs_snowball: float
    executive_summary: str
    actionable_takeaways: List[str]


@router.post("/optimize", response_model=DebtOptimizeResponse)
async def optimize_debt_payoff(
    request: DebtOptimizeRequest,
    current_user: database.User = Depends(deps.get_current_user),
    db: AsyncSession = Depends(deps.get_db),
):
    """
    Computes mathematical debt acceleration comparing Baseline vs Avalanche vs Snowball strategies.
    Uses authenticated user liabilities from database if custom_liabilities is not provided.
    """
    liabilities_data = request.custom_liabilities

    if not liabilities_data:
        res = await db.execute(
            select(database.User)
            .where(database.User.id == current_user.id)
            .options(selectinload(database.User.liabilities))
        )
        user = res.scalars().first()
        if not user or not user.liabilities:
            liabilities_data = []
        else:
            liabilities_data = [
                {
                    "id": l.id,
                    "name": l.name,
                    "principal": l.principal_remaining,
                    "annual_interest_rate": l.interest_rate,
                    "monthly_payment": l.monthly_payment,
                }
                for l in user.liabilities
            ]

    result = compare_debt_acceleration(
        liabilities=liabilities_data,
        extra_monthly_prepayment=request.extra_monthly_prepayment,
        lump_sum_prepayment=request.lump_sum_prepayment,
    )

    return result.to_dict()
