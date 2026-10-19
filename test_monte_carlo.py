import sys
import time

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from engine.monte_carlo import run_monte_carlo_sip


def test_monte_carlo():
    print("Testing Monte Carlo Simulation (10,000 paths)...")
    start = time.time()
    
    # Target: 5,000,000 in 10 years. Current: 500,000. SIP: 20,000/mo.
    result = run_monte_carlo_sip(
        target_amount=5000000.0,
        current_savings=500000.0,
        monthly_sip=20000.0,
        years=10,
        mean_annual_return=0.12,
        annual_volatility=0.15,
        num_simulations=10000
    )
    
    duration = (time.time() - start) * 1000
    print(f"Simulation completed in {duration:.2f} ms")
    print(f"Probability of Success: {result['probability_of_success_percent']}%")
    print(f"10th Percentile (Worst): ₹{result['worst_case_10th_percentile']}")
    print(f"50th Percentile (Expected): ₹{result['expected_50th_percentile']}")
    print(f"90th Percentile (Best): ₹{result['best_case_90th_percentile']}")

if __name__ == "__main__":
    test_monte_carlo()
