# Updated Budget Constants
STANDARD_DEDUCTION_NEW = 75000.0 # Budget 2024 update
STANDARD_DEDUCTION_OLD = 50000.0
HEALTH_CESS_RATE = 0.04

def _calc_slabs_new(income: float) -> float:
    """Core slab logic for the Latest New Regime (Budget 2024+)."""
    tax = 0.0
    if income > 1500000: tax += (income - 1500000) * 0.30; income = 1500000
    if income > 1200000: tax += (income - 1200000) * 0.20; income = 1200000
    if income > 1000000: tax += (income - 1000000) * 0.15; income = 1000000
    if income > 700000:  tax += (income - 700000) * 0.10; income = 700000
    if income > 300000:  tax += (income - 300000) * 0.05
    return tax

def calculate_new_regime_tax(salary: float, agri_income: float = 0.0) -> dict:
    """Calculates New Regime tax and isolates the agricultural rebate."""
    taxable_income = max(0.0, salary - STANDARD_DEDUCTION_NEW)
    agri_rebate = 0.0
    
    # 87A Rebate: No tax if taxable income is up to 7 Lakhs
    if taxable_income <= 700000:
        return {"tax": 0.0, "agri_rebate": 0.0}
        
    # PARTIAL INTEGRATION RULE for Agricultural Income
    if agri_income > 5000:
        tax_total = _calc_slabs_new(taxable_income + agri_income)
        # Basic Exemption Limit in New Regime is 3L
        agri_rebate = _calc_slabs_new(300000 + agri_income)
        tax = tax_total - agri_rebate
    else:
        tax = _calc_slabs_new(taxable_income)
        
    return {
        "tax": tax * (1 + HEALTH_CESS_RATE), 
        "agri_rebate": agri_rebate
    }

def _calc_slabs_old(income: float) -> float:
    """Core slab logic for the Old Regime."""
    tax = 0.0
    if income > 1000000: tax += (income - 1000000) * 0.30; income = 1000000
    if income > 500000: tax += (income - 500000) * 0.20; income = 500000
    if income > 250000: tax += (income - 250000) * 0.05
    return tax

def calculate_old_regime_tax(salary: float, deductions: float, agri_income: float = 0.0) -> dict:
    """Calculates Old Regime tax and isolates the agricultural rebate."""
    taxable_income = max(0.0, salary - STANDARD_DEDUCTION_OLD - deductions)
    agri_rebate = 0.0
    
    # 87A Rebate: No tax if taxable income is up to 5 Lakhs
    if taxable_income <= 500000:
        return {"tax": 0.0, "agri_rebate": 0.0}
        
    if agri_income > 5000:
        tax_total = _calc_slabs_old(taxable_income + agri_income)
        # Basic Exemption Limit in Old Regime is 2.5L
        agri_rebate = _calc_slabs_old(250000 + agri_income)
        tax = tax_total - agri_rebate
    else:
        tax = _calc_slabs_old(taxable_income)
        
    return {
        "tax": tax * (1 + HEALTH_CESS_RATE), 
        "agri_rebate": agri_rebate
    }
