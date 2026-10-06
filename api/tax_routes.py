from fastapi import APIRouter, Depends
from pydantic import BaseModel
from engine.tax_calc import calculate_new_regime_tax, calculate_old_regime_tax
from api.deps import get_current_user
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
