"""
ChronoFi Liquidity Shock & Emergency Stress Testing API.
"""

from typing import Any, Dict, List, Optional
from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from api import deps
from engine.stress_engine import StressTestResult, run_liquidity_stress_test
from models import database

router = APIRouter()


class StressTestRequest(BaseModel):
    shock_type: str = Field(default="job_loss", description="Type of shock: job_loss, medical_emergency, income_reduction, combined")
    shock_duration_months: int = Field(default=6, ge=1, le=60, description="Duration of shock in months")
    income_reduction_pct: float = Field(default=100.0, ge=0.0, le=100.0, description="Percentage drop in salary (100% for full job loss)")
    lump_sum_cost: float = Field(default=0.0, ge=0.0, description="Immediate lump sum cost (e.g. medical bill)")


class AssetTierResponse(BaseModel):
    tier_name: str
    asset_names: List[str]
    initial_value: float
    liquidated_value: float
    remaining_value: float
    friction_cost: float


class MonthlyStressPointResponse(BaseModel):
    month: int
    income: float
    expenses: float
    debt_service: float
    net_deficit: float
    remaining_total_wealth: float
    tier1_liquid_remaining: float
    tier2_fd_remaining: float
    tier3_debt_remaining: float
    tier4_equity_remaining: float
    is_solvent: bool


class StressTestResponse(BaseModel):
    scenario_name: str
    shock_type: str
    shock_duration_months: int
    monthly_burn_rate: float
    net_monthly_deficit: float
    total_shock_cost: float
    survival_runway_months: float
    is_solvent_through_shock: bool
    insolvency_month: Optional[int] = None
    total_wealth_before: float
    total_wealth_after: float
    total_friction_losses: float
    recommended_emergency_fund: float
    emergency_fund_gap: float
    resilience_rating: str
    asset_tiers: List[AssetTierResponse]
    monthly_trajectory: List[MonthlyStressPointResponse]
    actionable_advice: List[str]


@router.post("/simulate", response_model=StressTestResponse)
async def simulate_liquidity_stress(
    request: StressTestRequest,
    current_user: database.User = Depends(deps.get_current_user),
    db: AsyncSession = Depends(deps.get_db),
):
    """
    Simulates a liquidity shock against the authenticated user's verified balance sheet.
    Computes exact survival runway, asset liquidation sequence, and solvency risks.
    """
    res = await db.execute(
        select(database.User)
        .where(database.User.id == current_user.id)
        .options(
            selectinload(database.User.assets),
            selectinload(database.User.liabilities),
        )
    )
    user = res.scalars().first()
    if not user:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="User not found")

    monthly_expenses = user.monthly_expenses if (user.monthly_expenses and user.monthly_expenses > 0) else (user.base_salary * 0.40)
    existing_emis = sum(l.monthly_payment for l in user.liabilities) if user.liabilities else 0.0

    assets_data = [
        {
            "name": a.name,
            "asset_type": a.asset_type,
            "current_value": a.current_value,
            "expected_return_rate": a.expected_return_rate,
        }
        for a in user.assets
    ] if user.assets else []

    stress_result = run_liquidity_stress_test(
        monthly_expenses=monthly_expenses,
        base_salary=user.base_salary,
        existing_liabilities_monthly=existing_emis,
        assets=assets_data,
        shock_type=request.shock_type,
        shock_duration_months=request.shock_duration_months,
        income_reduction_pct=request.income_reduction_pct,
        lump_sum_cost=request.lump_sum_cost,
    )

    return stress_result.to_dict()
