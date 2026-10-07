from langchain_core.tools import tool
from engine.tax_calc import calculate_new_regime_tax, calculate_old_regime_tax

@tool
def calculate_tax_liability(salary: float, agricultural_income: float = 0.0, deductions: float = 0.0) -> dict:
    """
    Calculates Indian income tax liability under both Old and New regimes. 
    ALWAYS use this tool when a user asks how much tax they will pay based on their salary.
    """
    new_regime = calculate_new_regime_tax(salary, agricultural_income)
    old_regime = calculate_old_regime_tax(salary, deductions, agricultural_income)
    return {
        "tax_year": "Tax Year 2026-27",
        "new_regime_tax": new_regime["tax"],
        "old_regime_tax": old_regime["tax"],
        "recommendation": "New Regime" if new_regime["tax"] <= old_regime["tax"] else "Old Regime"
    }
