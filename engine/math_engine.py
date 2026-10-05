import numpy_financial as npf

def calculate_future_value(current_value: float, annual_return_rate: float, years: int) -> float:
    """
    Project current assets forward based on expected returns.
    """
    # FV = PV * (1 + r)^n
    # Using numpy_financial.fv(rate, nper, pmt, pv)
    fv = npf.fv(annual_return_rate, years, 0, -current_value)
    return round(float(fv), 2)

def calculate_required_sip(target_amount: float, current_savings: float, 
                           expected_annual_return: float, years: int, 
                           inflation_rate: float = 0.06) -> dict:
    """
    Determine exact monthly savings needed to hit a specific Goal by its target date, 
    adjusting for inflation.
    """
    # 1. Adjust target amount for inflation
    inflation_adjusted_target = calculate_future_value(target_amount, inflation_rate, years)
    
    # 2. Monthly rate and periods
    monthly_rate = expected_annual_return / 12
    months = years * 12
    
    # 3. Calculate SIP using PMT function
    # PMT(rate, nper, pv, fv)
    # We want FV = inflation_adjusted_target, PV = current_savings
    pmt = npf.pmt(monthly_rate, months, -current_savings, inflation_adjusted_target)
    
    # PMT returns a negative number for cash outflow, make it positive
    required_monthly = -pmt
    
    return {
        "original_target": round(target_amount, 2),
        "inflation_adjusted_target": round(inflation_adjusted_target, 2),
        "required_monthly_sip": round(float(required_monthly), 2),
        "assumed_inflation": inflation_rate,
        "assumed_return": expected_annual_return
    }

def run_scenario(base_salary: float, monthly_expenses: float, 
                 months_unemployed: int = 0, salary_growth_rate: float = 0.10) -> dict:
    """
    Recalculate basic cash flow timeline if a variable changes (e.g., job loss).
    Simplified deterministic cash flow projection for 1 year.
    """
    annual_income = (base_salary * (12 - months_unemployed))
    annual_expenses = (monthly_expenses * 12)
    surplus = annual_income - annual_expenses
    
    return {
        "projected_annual_income": round(annual_income, 2),
        "projected_annual_expenses": round(annual_expenses, 2),
        "annual_surplus": round(surplus, 2),
        "months_unemployed": months_unemployed,
        "is_deficit": surplus < 0
    }
