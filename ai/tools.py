from langchain_core.tools import tool
from engine.tax_calc import calculate_new_regime_tax, calculate_old_regime_tax

@tool
def calculate_tax_liability(salary: float, agricultural_income: float = 0.0) -> dict:
    """
    Calculates Indian income tax liability. 
    ALWAYS use this tool when a user asks how much tax they will pay based on their salary.
    """
    # Using the deep engine just built
    new_regime = calculate_new_regime_tax(salary, agricultural_income)
    return {
        "regime": "New Regime (Budget 2024)",
        "tax_owed": new_regime["tax"],
        "agricultural_rebate_saved": new_regime["agri_rebate"]
    }
