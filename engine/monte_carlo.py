import numpy as np

def run_monte_carlo_sip(
    target_amount: float, 
    current_savings: float, 
    monthly_sip: float, 
    years: int, 
    mean_annual_return: float = 0.12, # 12% default equity return
    annual_volatility: float = 0.15,  # 15% standard deviation
    num_simulations: int = 10000
) -> dict:
    """
    Runs a vectorized Monte Carlo simulation to project portfolio value with monthly contributions.
    Returns probabilities and percentiles.
    """
    months = years * 12
    # Convert annual metrics to monthly
    mean_monthly_return = mean_annual_return / 12
    monthly_volatility = annual_volatility / np.sqrt(12)
    
    # Generate random monthly returns for all simulations simultaneously using vectorized Normal distribution
    # Shape: (months, num_simulations)
    random_returns = np.random.normal(mean_monthly_return, monthly_volatility, (months, num_simulations))
    
    # Add 1 to returns to get growth multipliers (e.g., 1.01 for 1% return)
    growth_multipliers = 1 + random_returns
    
    # Initialize a matrix to hold the portfolio values
    portfolio_paths = np.zeros((months + 1, num_simulations))
    portfolio_paths[0] = current_savings
    
    # Calculate the path month by month
    for month in range(1, months + 1):
        # Current month value = Previous month value * market growth + monthly SIP
        portfolio_paths[month] = (portfolio_paths[month - 1] * growth_multipliers[month - 1]) + monthly_sip
        
    final_values = portfolio_paths[-1]
    
    # Calculate statistics
    success_count = np.sum(final_values >= target_amount)
    probability_of_success = (success_count / num_simulations) * 100
    
    worst_case = np.percentile(final_values, 10)
    expected_case = np.percentile(final_values, 50)
    best_case = np.percentile(final_values, 90)
    
    return {
        "probability_of_success_percent": round(probability_of_success, 1),
        "worst_case_10th_percentile": round(worst_case, 2),
        "expected_50th_percentile": round(expected_case, 2),
        "best_case_90th_percentile": round(best_case, 2),
        "target_amount": target_amount
    }
