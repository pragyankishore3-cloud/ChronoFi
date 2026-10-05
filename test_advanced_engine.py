from engine.advanced_monte_carlo import run_advanced_monte_carlo
import time

def test_advanced_engine():
    print("Testing Advanced Quant Engine (Merton Jump-Diffusion, Vasicek Inflation, Glide Path)...")
    start = time.time()
    
    # Target: 5,000,000 in 10 years. Current: 500,000. SIP: 20,000/mo.
    result = run_advanced_monte_carlo(
        target_amount=5000000.0,
        current_savings=500000.0,
        monthly_sip=20000.0,
        years=10,
        initial_equity_allocation=0.80,
        num_simulations=10000
    )
    
    duration = (time.time() - start) * 1000
    print(f"Simulation completed in {duration:.2f} ms")
    print(f"Probability of Success: {result['probability_of_success_percent']}%")
    print(f"Original Target: 5000000")
    print(f"Inflated Target (Vasicek): {result['median_inflated_target']}")
    print(f"Expected Portfolio: {result['expected_50th_percentile']}")
    print(f"Worst Case (10th Percentile): {result['worst_case_10th_percentile']}")
    print(f"CVaR (Black Swan Risk): {result['conditional_value_at_risk_cvar']}")

if __name__ == "__main__":
    test_advanced_engine()
