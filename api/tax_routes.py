from typing import Any, Dict, List, Optional
from fastapi import APIRouter, Depends
from pydantic import BaseModel, Field

from api.deps import get_current_user
from engine.tax_calc import calculate_new_regime_tax, calculate_old_regime_tax
from models.database import User

router = APIRouter()

class TaxRequest(BaseModel):
    salary: float
    deductions: float = 0.0
    agri_income: float = 0.0

class TaxResponse(BaseModel):
    new_regime_tax: float
    new_regime_agri_rebate: float
    old_regime_tax: float
    old_regime_agri_rebate: float
    recommendation: str

@router.post("/calculate", response_model=TaxResponse)
async def calculate_taxes(req: TaxRequest, current_user: User = Depends(get_current_user)):
    """
    Calculates tax liability under both Old and New regimes (Tax Year 2026-27 rules)
    and recommends the most tax-efficient regime.
    """
    new_regime = calculate_new_regime_tax(req.salary, req.agri_income)
    old_regime = calculate_old_regime_tax(req.salary, req.deductions, req.agri_income)
    
    return TaxResponse(
        new_regime_tax=round(new_regime["tax"], 2),
        new_regime_agri_rebate=round(new_regime["agri_rebate"], 2),
        old_regime_tax=round(old_regime["tax"], 2),
        old_regime_agri_rebate=round(old_regime["agri_rebate"], 2),
        recommendation="New Regime" if new_regime["tax"] <= old_regime["tax"] else "Old Regime"
    )


class GoalDrawdownRequest(BaseModel):
    goal_name: str = "Retirement Goal"
    target_years: int = 15
    target_amount: float = 10000000.0
    gross_projected_corpus: float = 12500000.0
    total_principal_invested: float = 6000000.0
    equity_pct: Optional[float] = None
    debt_pct: Optional[float] = None
    exempt_pct: Optional[float] = None
    risk_tolerance: Optional[str] = "Moderate"
    marginal_tax_rate: Optional[float] = 0.30
    inflation_rate: Optional[float] = 0.06


class PortfolioDrawdownRequest(BaseModel):
    risk_tolerance: Optional[str] = "Moderate"
    marginal_tax_rate: Optional[float] = 0.30
    custom_goals: Optional[List[Dict[str, Any]]] = None


@router.post("/goal-drawdown")
async def calculate_goal_drawdown_taxes(
    req: GoalDrawdownRequest,
    current_user: User = Depends(get_current_user),
):
    """
    Computes Indian capital gains tax (Budget 2024 LTCG/STCG rules) and post-tax
    corpus for a goal maturity.
    """
    from engine.tax_drawdown import calculate_goal_tax_drawdown

    report = calculate_goal_tax_drawdown(
        goal_name=req.goal_name,
        target_years=req.target_years,
        target_amount=req.target_amount,
        gross_projected_corpus=req.gross_projected_corpus,
        total_principal_invested=req.total_principal_invested,
        equity_pct=req.equity_pct,
        debt_pct=req.debt_pct,
        exempt_pct=req.exempt_pct,
        risk_tolerance=req.risk_tolerance or current_user.risk_tolerance or "Moderate",
        marginal_tax_rate=req.marginal_tax_rate or 0.30,
        inflation_rate=req.inflation_rate or 0.06,
    )
    return report.to_dict()


@router.post("/portfolio-drawdown")
async def calculate_portfolio_drawdown_taxes(
    req: PortfolioDrawdownRequest,
    current_user: User = Depends(get_current_user),
):
    """
    Computes capital gains tax drag across multiple goals for the user.
    """
    from engine.tax_drawdown import calculate_portfolio_goals_tax

    goals_to_evaluate = req.custom_goals or []
    if not goals_to_evaluate:
        # Fallback to user's saved goals
        goals_to_evaluate = [
            {
                "name": g.name,
                "target_years": g.target_years,
                "target_amount": g.target_amount,
                "gross_projected_corpus": g.target_amount * 1.15,
                "total_principal_invested": g.target_amount * 0.65,
                "inflation_rate": 0.06,
            }
            for g in (current_user.goals or [])
        ]

    report = calculate_portfolio_goals_tax(
        goals=goals_to_evaluate,
        risk_tolerance=req.risk_tolerance or current_user.risk_tolerance or "Moderate",
        marginal_tax_rate=req.marginal_tax_rate or 0.30,
    )
    return report

