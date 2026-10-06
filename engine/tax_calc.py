# Updated Budget Constants
STANDARD_DEDUCTION_NEW = 75000.0
STANDARD_DEDUCTION_OLD = 50000.0
HEALTH_CESS_RATE = 0.04

def _calc_slabs_new(income: float) -> float:
    """Core slab logic for the Latest New Regime (Tax Year 2026-27)."""
    tax = 0.0
    if income > 2400000: tax += (income - 2400000) * 0.30; income = 2400000
    if income > 2000000: tax += (income - 2000000) * 0.25; income = 2000000
    if income > 1600000: tax += (income - 1600000) * 0.20; income = 1600000
    if income > 1200000: tax += (income - 1200000) * 0.15; income = 1200000
    if income > 800000:  tax += (income - 800000) * 0.10; income = 800000
    if income > 400000:  tax += (income - 400000) * 0.05
    return tax

def calculate_new_regime_tax(salary: float, agri_income: float = 0.0) -> dict:
    """Calculates New Regime tax for Tax Year 2026-27 with Sec 156 (formerly 87A) rebate and Marginal Relief."""
    taxable_income = max(0.0, salary - STANDARD_DEDUCTION_NEW)
    BASIC_EXEMPTION = 400000.0
    agri_rebate = 0.0
    
    # 1. Calculate Base Tax (with or without Agricultural Integration)
    # Integration only triggers if Agri Income > 5000 AND Non-Agri Income > Basic Exemption Limit
    if agri_income > 5000 and taxable_income > BASIC_EXEMPTION:
        tax_total = _calc_slabs_new(taxable_income + agri_income)
        agri_rebate = _calc_slabs_new(BASIC_EXEMPTION + agri_income)
        tax_before_rebate = tax_total - agri_rebate
    else:
        tax_before_rebate = _calc_slabs_new(taxable_income)
        
    # 2. Apply Section 156 Rebate AFTER integration, respecting the maximum cap
    # Eligibility is strictly based on Non-Agri Taxable Income being <= 12 Lakhs
    if taxable_income <= 1200000:
        # The rebate is capped at ₹60,000 for the New Regime
        rebate = min(tax_before_rebate, 60000.0)
        tax_before_cess = max(0.0, tax_before_rebate - rebate)
    else:
        # 3. Apply Marginal Relief if taxable_income slightly exceeds 12 Lakhs
        # Tax payable cannot exceed the income earned above the 12 Lakh threshold
        marginal_relief_cap = max(0.0, taxable_income - 1200000)
        tax_before_cess = min(tax_before_rebate, marginal_relief_cap)
        
    return {
        "tax": tax_before_cess * (1 + HEALTH_CESS_RATE), 
        "agri_rebate": agri_rebate
    }

def _calc_slabs_old(income: float) -> float:
    """Core slab logic for the Old Regime (Tax Year 2026-27)."""
    tax = 0.0
    if income > 1000000: tax += (income - 1000000) * 0.30; income = 1000000
    if income > 500000: tax += (income - 500000) * 0.20; income = 500000
    if income > 250000: tax += (income - 250000) * 0.05
    return tax

def calculate_old_regime_tax(salary: float, deductions: float, agri_income: float = 0.0) -> dict:
    """Calculates Old Regime tax for Tax Year 2026-27 with Sec 156 (formerly 87A) cap and Basic Exemption checks."""
    taxable_income = max(0.0, salary - STANDARD_DEDUCTION_OLD - deductions)
    BASIC_EXEMPTION = 250000.0
    agri_rebate = 0.0
    
    # Integration only triggers if Agri Income > 5000 AND Non-Agri Income > Basic Exemption Limit
    if agri_income > 5000 and taxable_income > BASIC_EXEMPTION:
        tax_total = _calc_slabs_old(taxable_income + agri_income)
        agri_rebate = _calc_slabs_old(BASIC_EXEMPTION + agri_income)
        tax_before_rebate = tax_total - agri_rebate
    else:
        tax_before_rebate = _calc_slabs_old(taxable_income)
        
    # Section 156 (formerly 87A) eligibility for Old Regime is <= 5 Lakhs
    if taxable_income <= 500000:
        # The rebate is capped at ₹12,500 for the Old Regime
        rebate = min(tax_before_rebate, 12500.0)
        tax_before_cess = max(0.0, tax_before_rebate - rebate)
    else:
        tax_before_cess = tax_before_rebate
        
    return {
        "tax": tax_before_cess * (1 + HEALTH_CESS_RATE), 
        "agri_rebate": agri_rebate
    }
