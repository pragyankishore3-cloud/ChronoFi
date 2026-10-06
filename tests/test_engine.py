from engine.advanced_monte_carlo import run_advanced_monte_carlo

def test_advanced_monte_carlo_math():
    """
    Tests the quantitative math engine to ensure it returns correctly structured
    probabilistic data and that basic logic holds (e.g. Best Case > Worst Case).
    """
    result = run_advanced_monte_carlo(
        target_amount=1000000.0,
        current_savings=100000.0,
        monthly_sip=5000.0,
        years=5,
        initial_equity_allocation=0.80,
        num_simulations=1000  # Smaller simulation size for faster unit testing
    )
    
    # 1. Check data structure
    assert "probability_of_success_percent" in result
    assert "worst_case_10th_percentile" in result
    assert "expected_50th_percentile" in result
    assert "best_case_90th_percentile" in result
    assert "conditional_value_at_risk_cvar" in result
    
    # 2. Check mathematical constraints
    assert result["best_case_90th_percentile"] > result["expected_50th_percentile"]
    assert result["expected_50th_percentile"] > result["worst_case_10th_percentile"]
    
    # CVaR (average of worst 10%) must be less than or equal to the 10th percentile boundary
    assert result["conditional_value_at_risk_cvar"] <= result["worst_case_10th_percentile"]
    
    # Probability must be bounded
    assert 0.0 <= result["probability_of_success_percent"] <= 100.0
