# Updated Budget Constants
STANDARD_DEDUCTION_NEW = 75000.0
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
    """Calculates New Regime tax with proper 87A cap and Basic Exemption checks."""
    taxable_income = max(0.0, salary - STANDARD_DEDUCTION_NEW)
    BASIC_EXEMPTION = 300000.0
    agri_rebate = 0.0
    
    # 1. Calculate Base Tax (with or without Agricultural Integration)
    # Integration only triggers if Agri Income > 5000 AND Non-Agri Income > Basic Exemption Limit
    if agri_income > 5000 and taxable_income > BASIC_EXEMPTION:
        tax_total = _calc_slabs_new(taxable_income + agri_income)
        agri_rebate = _calc_slabs_new(BASIC_EXEMPTION + agri_income)
        tax_before_87a = tax_total - agri_rebate
    else:
        tax_before_87a = _calc_slabs_new(taxable_income)
        
    # 2. Apply Section 87A Rebate AFTER integration, respecting the maximum cap
    # 87A eligibility is strictly based on Non-Agri Taxable Income being <= 7 Lakhs
    if taxable_income <= 700000:
        # The rebate is capped at ₹25,000 for the New Regime
        rebate_87a = min(tax_before_87a, 25000.0)
        tax_before_cess = max(0.0, tax_before_87a - rebate_87a)
    else:
        tax_before_cess = tax_before_87a
        
    return {
        "tax": tax_before_cess * (1 + HEALTH_CESS_RATE), 
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
    """Calculates Old Regime tax with proper 87A cap and Basic Exemption checks."""
    taxable_income = max(0.0, salary - STANDARD_DEDUCTION_OLD - deductions)
    BASIC_EXEMPTION = 250000.0
    agri_rebate = 0.0
    
    # Integration only triggers if Agri Income > 5000 AND Non-Agri Income > Basic Exemption Limit
    if agri_income > 5000 and taxable_income > BASIC_EXEMPTION:
        tax_total = _calc_slabs_old(taxable_income + agri_income)
        agri_rebate = _calc_slabs_old(BASIC_EXEMPTION + agri_income)
        tax_before_87a = tax_total - agri_rebate
    else:
        tax_before_87a = _calc_slabs_old(taxable_income)
        
    # Section 87A eligibility for Old Regime is <= 5 Lakhs
    if taxable_income <= 500000:
        # The rebate is capped at ₹12,500 for the Old Regime
        rebate_87a = min(tax_before_87a, 12500.0)
        tax_before_cess = max(0.0, tax_before_87a - rebate_87a)
    else:
        tax_before_cess = tax_before_87a
        
    return {
        "tax": tax_before_cess * (1 + HEALTH_CESS_RATE), 
        "agri_rebate": agri_rebate
    }
