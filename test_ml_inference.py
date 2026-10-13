import time

from engine.ml_forecaster import run_ml_forecaster


def test_ml_engine():
    print("Testing ML Forecaster (PyTorch LSTM + MC Dropout)...")
    start = time.time()
    
    result = run_ml_forecaster(
        target_amount=5000000.0,
        current_savings=500000.0,
        monthly_sip=20000.0,
        years=10,
        initial_equity_allocation=0.80,
        num_simulations=1000
    )
    
    duration = (time.time() - start) * 1000
    print(f"ML Inference completed in {duration:.2f} ms")
    print(f"Probability of Success: {result['probability_of_success_percent']}%")
    print("Original Target: 5000000")
    print(f"Inflated Target (Naive 6%): {result['median_inflated_target']}")
    print(f"Expected Portfolio: {result['expected_50th_percentile']}")
    print(f"Worst Case (10th Percentile): {result['worst_case_10th_percentile']}")
    print(f"CVaR (Black Swan Risk): {result['conditional_value_at_risk_cvar']}")

if __name__ == "__main__":
    test_ml_engine()
