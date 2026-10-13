import pytest
import math
from engine.scenario_engine import (
    calculate_loan_amortization,
    project_retirement_simulation,
    _run_deterministic_corpus,
)


def test_standard_amortizing_loan():
    """
    Test standard amortizing loan: ₹10 Lakh at 9% annual interest for 60 months.
    EMI = 1,000,000 * (0.09/12) * (1 + 0.09/12)^60 / ((1 + 0.09/12)^60 - 1)
    Expected EMI is approx 20,758.36
    """
    metrics = calculate_loan_amortization(
        principal=1000000.0,
        annual_interest_rate=0.09,
        tenure_months=60,
    )
    assert not metrics.is_zero_interest
    assert metrics.monthly_emi == 20758.36
    assert metrics.total_payment == 1245501.60
    assert metrics.total_interest == 245501.60
    assert metrics.tenure_months == 60
    assert metrics.principal == 1000000.0


def test_zero_interest_loan():
    """
    Test zero-interest loan: ₹6 Lakh at 0% for 36 months.
    EMI = 600,000 / 36 = 16,666.67
    Total interest = 0.0
    """
    metrics = calculate_loan_amortization(
        principal=600000.0,
        annual_interest_rate=0.0,
        tenure_months=36,
    )
    assert metrics.is_zero_interest
    assert metrics.monthly_emi == 16666.67
    assert metrics.total_interest == 0.0
    assert metrics.total_payment == 600000.0


def test_loan_invalid_inputs():
    """Test boundary and invalid conditions."""
    with pytest.raises(ValueError, match="Principal cannot be negative"):
        calculate_loan_amortization(principal=-1000, annual_interest_rate=0.09, tenure_months=60)

    with pytest.raises(ValueError, match="Interest rate cannot be negative"):
        calculate_loan_amortization(principal=100000, annual_interest_rate=-0.05, tenure_months=60)

    with pytest.raises(ValueError, match="Tenure months must be greater than zero"):
        calculate_loan_amortization(principal=100000, annual_interest_rate=0.09, tenure_months=0)


def test_zero_principal_loan():
    """Test 0 principal edge case."""
    metrics = calculate_loan_amortization(principal=0.0, annual_interest_rate=0.09, tenure_months=60)
    assert metrics.monthly_emi == 0.0
    assert metrics.total_interest == 0.0
    assert metrics.total_payment == 0.0


def test_retirement_simulation_baseline_vs_scenario():
    """
    Test that taking a ₹10 Lakh car loan reduces monthly surplus, reduces projected corpus,
    and shifts retirement metrics compared to baseline.
    """
    res = project_retirement_simulation(
        current_age=35,
        base_salary=250000.0,
        existing_liabilities_monthly=0.0,
        current_savings=3500000.0,
        target_amount=50000000.0,
        target_years=25,
        loan_principal=1000000.0,
        annual_interest_rate=0.09,
        loan_tenure_months=60,
        down_payment=0.0,
        monthly_vehicle_expenses=5000.0,
        expected_annual_return=0.105,
        inflation_rate=0.06,
        num_simulations=1000,
        seed=42,
    )

    assert res.status == "completed"
    assert res.loan_metrics is not None
    assert res.loan_metrics.monthly_emi == 20758.36
    assert res.loan_metrics.total_interest == 245501.60

    # Monthly cash flow impact
    assert res.impact.monthly_cash_flow_difference == -(20758.36 + 5000.0)
    assert res.modified.monthly_investable_surplus < res.baseline.monthly_investable_surplus

    # Corpus impact: Modified corpus should be lower than baseline due to reduced investment
    assert res.modified.projected_corpus < res.baseline.projected_corpus
    assert res.impact.corpus_difference < 0

    # Retirement age difference should be >= 0 (delay in retirement or equal)
    assert res.impact.retirement_age_difference >= 0

    # Goal success probability
    assert res.modified.goal_success_probability <= res.baseline.goal_success_probability

    # Alternatives generated
    assert len(res.alternatives) == 4
    for alt in res.alternatives:
        assert alt.monthly_emi > 0
        assert alt.trade_off_summary != ""

    # Trajectory checkpoints
    assert len(res.trajectory) == 26  # Year 0 to Year 25 inclusive
    assert res.trajectory[0].year == 0
    assert res.trajectory[-1].year == 25


def test_paired_monte_carlo_crn_reproducibility():
    """
    Test Common Random Numbers (CRN) reproducibility:
    Two runs with identical seeds produce identical paired differences.
    """
    res1 = project_retirement_simulation(
        current_age=35,
        base_salary=250000.0,
        existing_liabilities_monthly=0.0,
        current_savings=3500000.0,
        target_amount=50000000.0,
        target_years=25,
        loan_principal=1000000.0,
        annual_interest_rate=0.09,
        loan_tenure_months=60,
        num_simulations=500,
        seed=1234,
    )

    res2 = project_retirement_simulation(
        current_age=35,
        base_salary=250000.0,
        existing_liabilities_monthly=0.0,
        current_savings=3500000.0,
        target_amount=50000000.0,
        target_years=25,
        loan_principal=1000000.0,
        annual_interest_rate=0.09,
        loan_tenure_months=60,
        num_simulations=500,
        seed=1234,
    )

    assert res1.baseline.projected_corpus == res2.baseline.projected_corpus
    assert res1.modified.projected_corpus == res2.modified.projected_corpus
    assert res1.impact.corpus_difference == res2.impact.corpus_difference
    assert res1.baseline.goal_success_probability == res2.baseline.goal_success_probability
    assert res1.modified.goal_success_probability == res2.modified.goal_success_probability
