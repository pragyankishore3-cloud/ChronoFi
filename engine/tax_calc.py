STANDARD_DEDUCTION = 50000.0
HEALTH_CESS_RATE = 0.04

def calculate_new_regime_tax(salary: float) -> float:
    """Calculates tax under the New Regime (default)."""
    taxable_income = max(0.0, salary - STANDARD_DEDUCTION)
    
    # Rebate under 87A: No tax if taxable income is up to 7 Lakhs
    if taxable_income <= 700000:
        return 0.0

    tax = 0.0
    # Current New Regime Slabs
    if taxable_income > 1500000:
        tax += (taxable_income - 1500000) * 0.30
        taxable_income = 1500000
    if taxable_income > 1200000:
        tax += (taxable_income - 1200000) * 0.20
        taxable_income = 1200000
    if taxable_income > 900000:
        tax += (taxable_income - 900000) * 0.15
        taxable_income = 900000
    if taxable_income > 600000:
        tax += (taxable_income - 600000) * 0.10
        taxable_income = 600000
    if taxable_income > 300000:
        tax += (taxable_income - 300000) * 0.05

    return tax * (1 + HEALTH_CESS_RATE)

def calculate_old_regime_tax(salary: float, total_deductions: float) -> float:
    """Calculates tax under the Old Regime with applied deductions."""
    taxable_income = max(0.0, salary - STANDARD_DEDUCTION - total_deductions)
    
    # Rebate under 87A: No tax if taxable income is up to 5 Lakhs
    if taxable_income <= 500000:
        return 0.0

    tax = 0.0
    # Standard Old Regime Slabs
    if taxable_income > 1000000:
        tax += (taxable_income - 1000000) * 0.30
        taxable_income = 1000000
    if taxable_income > 500000:
        tax += (taxable_income - 500000) * 0.20
        taxable_income = 500000
    if taxable_income > 250000:
        tax += (taxable_income - 250000) * 0.05

    return tax * (1 + HEALTH_CESS_RATE)

def generate_tax_benefits() -> list[dict]:
    """Returns standard tax saving suggestions for the Old Regime."""
    return [
        {
            "section": "80C",
            "suggestion": "Invest in ELSS Mutual Funds, PPF, or EPF to maximize this deduction.",
            "max_deduction": 150000.0
        },
        {
            "section": "80D",
            "suggestion": "Purchase health insurance for yourself and parents.",
            "max_deduction": 75000.0 # 25k self + 50k senior citizen parents
        },
        {
            "section": "24(b)",
            "suggestion": "Claim interest paid on a home loan.",
            "max_deduction": 200000.0
        }
    ]
