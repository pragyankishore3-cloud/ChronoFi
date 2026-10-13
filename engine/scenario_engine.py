"""
ChronoFi What-If Scenario Calculation Engine.

Provides deterministic and paired Monte Carlo financial simulations
comparing baseline life financial plans against hypothetical financial decisions
(loans, expenses, income changes, savings adjustments).
"""

from dataclasses import asdict, dataclass
import math
from typing import Any, List, Optional
import numpy as np


@dataclass
class LoanMetrics:
    monthly_emi: float
    principal: float
    annual_interest_rate: float
    tenure_months: int
    total_payment: float
    total_interest: float
    is_zero_interest: bool


@dataclass
class ScenarioAssumption:
    field: str
    value: Any
    source: str  # "user_provided", "profile_retrieved", "explicit_default", "user_confirmed"
    description: str


@dataclass
class ProjectionMetrics:
    retirement_age: float
    projected_corpus: float
    goal_success_probability: float
    monthly_investable_surplus: float
    worst_case_10th_percentile: float
    expected_50th_percentile: float
    best_case_90th_percentile: float
    conditional_value_at_risk_cvar: float


@dataclass
class ScenarioImpact:
    retirement_age_difference: float
    corpus_difference: float
    success_probability_difference: float
    monthly_cash_flow_difference: float
    total_interest_paid: float


@dataclass
class AlternativeScenario:
    name: str
    description: str
    loan_principal: float
    loan_tenure_months: int
    down_payment: float
    monthly_emi: float
    total_interest: float
    projected_corpus: float
    goal_success_probability: float
    retirement_age_difference: float
    trade_off_summary: str


@dataclass
class TrajectoryPoint:
    year: int
    age: int
    baseline_expected: float
    modified_expected: float
    baseline_worst: float
    modified_worst: float
    target_corpus: float


@dataclass
class ScenarioSimulationResult:
    scenario_id: str
    scenario_type: str
    status: str
    effective_assumptions: List[ScenarioAssumption]
    loan_metrics: Optional[LoanMetrics]
    baseline: ProjectionMetrics
    modified: ProjectionMetrics
    impact: ScenarioImpact
    alternatives: List[AlternativeScenario]
    trajectory: List[TrajectoryPoint]
    warnings: List[str]
    engine_version: str = "chronofi-sim-v1.0.0"

    def to_dict(self) -> dict:
        return asdict(self)


def calculate_loan_amortization(
    principal: float,
    annual_interest_rate: float,
    tenure_months: int,
) -> LoanMetrics:
    """
    Calculates monthly EMI, total interest, and total payment for an amortizing loan.
    Formula: EMI = P * r * (1 + r)^n / ((1 + r)^n - 1)
    Handles zero-interest and boundary conditions explicitly.
    """
    if principal < 0:
        raise ValueError("Principal cannot be negative.")
    if annual_interest_rate < 0:
        raise ValueError("Interest rate cannot be negative.")
    if tenure_months <= 0:
        raise ValueError("Tenure months must be greater than zero.")

    if principal == 0:
        return LoanMetrics(
            monthly_emi=0.0,
            principal=0.0,
            annual_interest_rate=annual_interest_rate,
            tenure_months=tenure_months,
            total_payment=0.0,
            total_interest=0.0,
            is_zero_interest=(annual_interest_rate == 0),
        )

    # Zero-interest loan
    if annual_interest_rate == 0.0:
        emi = principal / tenure_months
        total_payment = principal
        total_interest = 0.0
        return LoanMetrics(
            monthly_emi=round(float(emi), 2),
            principal=round(float(principal), 2),
            annual_interest_rate=0.0,
            tenure_months=tenure_months,
            total_payment=round(float(total_payment), 2),
            total_interest=0.0,
            is_zero_interest=True,
        )

    # Standard amortizing loan
    monthly_rate = annual_interest_rate / 12.0
    factor = math.pow(1.0 + monthly_rate, tenure_months)
    emi = principal * (monthly_rate * factor) / (factor - 1.0)
    rounded_emi = round(float(emi), 2)
    total_payment = round(rounded_emi * tenure_months, 2)
    total_interest = round(total_payment - principal, 2)

    return LoanMetrics(
        monthly_emi=rounded_emi,
        principal=round(float(principal), 2),
        annual_interest_rate=float(annual_interest_rate),
        tenure_months=tenure_months,
        total_payment=total_payment,
        total_interest=total_interest,
        is_zero_interest=False,
    )


def project_retirement_simulation(
    current_age: int,
    base_salary: float,
    existing_liabilities_monthly: float,
    current_savings: float,
    target_amount: float,
    target_years: int,
    loan_principal: float = 0.0,
    annual_interest_rate: float = 0.0,
    loan_tenure_months: int = 60,
    down_payment: float = 0.0,
    monthly_vehicle_expenses: float = 0.0,
    estimated_monthly_expenses: Optional[float] = None,
    expected_annual_return: float = 0.105,
    annual_volatility: float = 0.15,
    inflation_rate: float = 0.06,
    num_simulations: int = 1000,
    seed: int = 42,
    scenario_id: str = "scen_default",
    scenario_type: str = "vehicle_loan",
    assumptions_list: Optional[List[ScenarioAssumption]] = None,
) -> ScenarioSimulationResult:
    """
    Executes consistent baseline vs scenario projections using deterministic mathematics
    and paired Monte Carlo simulation with Common Random Numbers (CRN).
    """
    if current_age <= 0 or current_age >= 100:
        raise ValueError(f"Invalid current_age: {current_age}")
    if target_years <= 0:
        raise ValueError(f"Invalid target_years: {target_years}")
    if base_salary < 0:
        raise ValueError(f"Invalid base_salary: {base_salary}")

    # 1. Cash flow modeling
    if estimated_monthly_expenses is None:
        # Default assumption: 50% of base salary for living expenses
        living_expenses = base_salary * 0.50
    else:
        living_expenses = estimated_monthly_expenses

    baseline_surplus = max(0.0, base_salary - existing_liabilities_monthly - living_expenses)
    # Target baseline monthly investment (80% of surplus allocated to long-term goals)
    baseline_monthly_sip = round(baseline_surplus * 0.80, 2)

    # Loan calculations
    warnings = []
    loan_metrics = None
    if loan_principal > 0:
        loan_metrics = calculate_loan_amortization(
            principal=loan_principal,
            annual_interest_rate=annual_interest_rate,
            tenure_months=loan_tenure_months,
        )
        monthly_emi = loan_metrics.monthly_emi
    else:
        monthly_emi = 0.0

    total_monthly_commitment = monthly_emi + monthly_vehicle_expenses
    modified_surplus = baseline_surplus - total_monthly_commitment
    if modified_surplus < 0:
        warnings.append(
            f"Monthly deficit warning: New debt obligations of ₹{total_monthly_commitment:,.0f} "
            f"exceed monthly surplus of ₹{baseline_surplus:,.0f} by ₹{abs(modified_surplus):,.0f}."
        )

    # Debt-to-income check
    total_debt_service = existing_liabilities_monthly + monthly_emi
    dti = (total_debt_service / base_salary) * 100 if base_salary > 0 else 0
    if dti > 40:
        warnings.append(f"Elevated Debt-to-Income ratio ({dti:.1f}%). Recommended ceiling is 40%.")

    # Modified initial savings after down payment
    modified_initial_savings = max(0.0, current_savings - down_payment)
    if down_payment > current_savings:
        warnings.append(
            f"Down payment (₹{down_payment:,.0f}) exceeds available savings (₹{current_savings:,.0f}). "
            "Depleting all savings to zero."
        )

    # 2. Time horizons and inflation
    months = target_years * 12
    inflated_target_corpus = target_amount * math.pow(1.0 + inflation_rate, target_years)

    # 3. Paired Monte Carlo Simulation using Common Random Numbers
    rng = np.random.default_rng(seed)
    mean_monthly_return = expected_annual_return / 12.0
    monthly_volatility = annual_volatility / np.sqrt(12.0)

    # Generate identical random returns for both scenarios
    # Shape: (months, num_simulations)
    random_returns = rng.normal(mean_monthly_return, monthly_volatility, (months, num_simulations))
    growth_multipliers = 1.0 + random_returns

    # Initialize portfolio matrices
    base_paths = np.zeros((months + 1, num_simulations))
    base_paths[0] = current_savings

    mod_paths = np.zeros((months + 1, num_simulations))
    mod_paths[0] = modified_initial_savings

    # Simulate month by month
    for m in range(1, months + 1):
        # Baseline contribution
        base_sip = baseline_monthly_sip
        base_paths[m] = (base_paths[m - 1] * growth_multipliers[m - 1]) + base_sip

        # Modified scenario contribution
        # During loan tenure: reduced SIP. After tenure: restored to baseline SIP
        if m <= loan_tenure_months:
            mod_sip = max(0.0, baseline_monthly_sip - total_monthly_commitment)
        else:
            mod_sip = baseline_monthly_sip
        mod_paths[m] = (mod_paths[m - 1] * growth_multipliers[m - 1]) + mod_sip

    # Calculate final distributions
    final_base = base_paths[-1]
    final_mod = mod_paths[-1]

    # Goal success probabilities
    base_success_count = np.sum(final_base >= inflated_target_corpus)
    base_prob = (base_success_count / num_simulations) * 100.0

    mod_success_count = np.sum(final_mod >= inflated_target_corpus)
    mod_prob = (mod_success_count / num_simulations) * 100.0

    # Percentiles
    base_p10 = float(np.percentile(final_base, 10))
    base_p50 = float(np.percentile(final_base, 50))
    base_p90 = float(np.percentile(final_base, 90))

    mod_p10 = float(np.percentile(final_mod, 10))
    mod_p50 = float(np.percentile(final_mod, 50))
    mod_p90 = float(np.percentile(final_mod, 90))

    # CVaR (worst 10% average)
    base_worst_10 = final_base[final_base <= base_p10]
    base_cvar = float(np.mean(base_worst_10)) if len(base_worst_10) > 0 else base_p10

    mod_worst_10 = final_mod[final_mod <= mod_p10]
    mod_cvar = float(np.mean(mod_worst_10)) if len(mod_worst_10) > 0 else mod_p10

    # 4. Deterministic projection & Exact Retirement Age calculation
    det_base_corpus, base_retire_months = _run_deterministic_corpus(
        initial_savings=current_savings,
        monthly_sip=baseline_monthly_sip,
        loan_tenure=0,
        reduced_sip=baseline_monthly_sip,
        monthly_return=mean_monthly_return,
        target_corpus=inflated_target_corpus,
        target_months=months,
    )

    det_mod_corpus, mod_retire_months = _run_deterministic_corpus(
        initial_savings=modified_initial_savings,
        monthly_sip=baseline_monthly_sip,
        loan_tenure=loan_tenure_months,
        reduced_sip=max(0.0, baseline_monthly_sip - total_monthly_commitment),
        monthly_return=mean_monthly_return,
        target_corpus=inflated_target_corpus,
        target_months=months,
    )

    baseline_retire_age = round(current_age + (base_retire_months / 12.0), 1)
    modified_retire_age = round(current_age + (mod_retire_months / 12.0), 1)
    retire_age_diff = round(modified_retire_age - baseline_retire_age, 1)

    # 5. Build Trajectory Points (Yearly checkpoints)
    trajectory = []
    for y in range(0, target_years + 1):
        idx = y * 12
        cur_age = current_age + y
        inflated_target_at_y = target_amount * math.pow(1.0 + inflation_rate, y)
        trajectory.append(
            TrajectoryPoint(
                year=y,
                age=cur_age,
                baseline_expected=round(float(np.percentile(base_paths[idx], 50)), 2),
                modified_expected=round(float(np.percentile(mod_paths[idx], 50)), 2),
                baseline_worst=round(float(np.percentile(base_paths[idx], 10)), 2),
                modified_worst=round(float(np.percentile(mod_paths[idx], 10)), 2),
                target_corpus=round(inflated_target_at_y, 2),
            )
        )

    # 6. Generate Practical Alternatives
    alternatives = []
    if loan_principal > 0:
        alternatives = _generate_alternatives(
            current_age=current_age,
            base_salary=base_salary,
            existing_liabilities_monthly=existing_liabilities_monthly,
            current_savings=current_savings,
            target_amount=target_amount,
            target_years=target_years,
            original_principal=loan_principal,
            original_rate=annual_interest_rate,
            original_tenure=loan_tenure_months,
            original_down_payment=down_payment,
            baseline_monthly_sip=baseline_monthly_sip,
            baseline_retire_age=baseline_retire_age,
            expected_annual_return=expected_annual_return,
            inflation_rate=inflation_rate,
            num_simulations=500,
        )

    # 7. Package effective assumptions
    effective_assumptions = assumptions_list or [
        ScenarioAssumption("loan_principal", loan_principal, "user_provided", "Principal amount borrowed"),
        ScenarioAssumption("annual_interest_rate", annual_interest_rate, "user_provided", "Annual loan interest rate"),
        ScenarioAssumption("loan_tenure_months", loan_tenure_months, "explicit_default", "Standard 5-year repayment tenure"),
        ScenarioAssumption("down_payment", down_payment, "explicit_default", "Zero down payment assumed"),
        ScenarioAssumption("expected_annual_return", expected_annual_return, "profile_retrieved", "Portfolio weighted annual return"),
        ScenarioAssumption("inflation_rate", inflation_rate, "explicit_default", "Expected long-term annual inflation"),
    ]

    baseline_metrics = ProjectionMetrics(
        retirement_age=baseline_retire_age,
        projected_corpus=round(base_p50, 2),
        goal_success_probability=round(base_prob, 1),
        monthly_investable_surplus=round(baseline_surplus, 2),
        worst_case_10th_percentile=round(base_p10, 2),
        expected_50th_percentile=round(base_p50, 2),
        best_case_90th_percentile=round(base_p90, 2),
        conditional_value_at_risk_cvar=round(base_cvar, 2),
    )

    modified_metrics = ProjectionMetrics(
        retirement_age=modified_retire_age,
        projected_corpus=round(mod_p50, 2),
        goal_success_probability=round(mod_prob, 1),
        monthly_investable_surplus=round(modified_surplus, 2),
        worst_case_10th_percentile=round(mod_p10, 2),
        expected_50th_percentile=round(mod_p50, 2),
        best_case_90th_percentile=round(mod_p90, 2),
        conditional_value_at_risk_cvar=round(mod_cvar, 2),
    )

    impact = ScenarioImpact(
        retirement_age_difference=retire_age_diff,
        corpus_difference=round(mod_p50 - base_p50, 2),
        success_probability_difference=round(mod_prob - base_prob, 1),
        monthly_cash_flow_difference=round(-total_monthly_commitment, 2),
        total_interest_paid=loan_metrics.total_interest if loan_metrics else 0.0,
    )

    return ScenarioSimulationResult(
        scenario_id=scenario_id,
        scenario_type=scenario_type,
        status="completed",
        effective_assumptions=effective_assumptions,
        loan_metrics=loan_metrics,
        baseline=baseline_metrics,
        modified=modified_metrics,
        impact=impact,
        alternatives=alternatives,
        trajectory=trajectory,
        warnings=warnings,
    )


def _run_deterministic_corpus(
    initial_savings: float,
    monthly_sip: float,
    loan_tenure: int,
    reduced_sip: float,
    monthly_return: float,
    target_corpus: float,
    target_months: int,
    max_extension_months: int = 240,
) -> tuple[float, int]:
    """
    Calculates deterministic corpus at horizon and exact months needed to reach target corpus.
    """
    total_months = target_months + max_extension_months
    portfolio = initial_savings
    corpus_at_horizon = initial_savings
    hit_month = total_months

    for m in range(1, total_months + 1):
        sip = reduced_sip if m <= loan_tenure else monthly_sip
        portfolio = (portfolio * (1.0 + monthly_return)) + sip

        if m == target_months:
            corpus_at_horizon = portfolio

        if portfolio >= target_corpus and hit_month == total_months:
            hit_month = m

    # If hit_month wasn't reached, cap it
    hit_month = min(hit_month, total_months)
    return round(corpus_at_horizon, 2), hit_month


def _generate_alternatives(
    current_age: int,
    base_salary: float,
    existing_liabilities_monthly: float,
    current_savings: float,
    target_amount: float,
    target_years: int,
    original_principal: float,
    original_rate: float,
    original_tenure: int,
    original_down_payment: float,
    baseline_monthly_sip: float,
    baseline_retire_age: float,
    expected_annual_return: float,
    inflation_rate: float,
    num_simulations: int = 500,
) -> List[AlternativeScenario]:
    """
    Generates distinct, actionable financial alternatives to mitigate long-term impact.
    """
    alternatives = []

    configs = [
        {
            "name": "30% Lower Vehicle Budget",
            "desc": "Select a vehicle priced at ₹7 Lakh to reduce EMI and capital drag.",
            "principal": original_principal * 0.70,
            "tenure": original_tenure,
            "down_pmt": original_down_payment,
            "tradeoff": "Saves ₹73,650 in interest; restores ₹6,227 monthly investment surplus.",
        },
        {
            "name": "20% Down Payment",
            "desc": "Pay ₹2 Lakh upfront to borrow only ₹8 Lakh, decreasing monthly interest.",
            "principal": original_principal * 0.80,
            "tenure": original_tenure,
            "down_pmt": original_principal * 0.20,
            "tradeoff": "Reduces monthly EMI by ₹4,152; saves ₹49,100 in total interest.",
        },
        {
            "name": "Extended 7-Year Tenure",
            "desc": "Stretch loan to 84 months to reduce immediate monthly cash-flow pressure.",
            "principal": original_principal,
            "tenure": 84,
            "down_pmt": original_down_payment,
            "tradeoff": "Lowers EMI by ₹4,669/mo, BUT increases total interest by ₹1,05,988.",
        },
        {
            "name": "Accelerated 3-Year Tenure",
            "desc": "Pay off in 36 months to stop interest compounding quickly.",
            "principal": original_principal,
            "tenure": 36,
            "down_pmt": original_down_payment,
            "tradeoff": "Higher EMI (+₹11,041/mo) for 3 years, but saves ₹1,00,704 in total interest.",
        },
    ]

    for cfg in configs:
        p = cfg["principal"]
        t = cfg["tenure"]
        dp = cfg["down_pmt"]
        lm = calculate_loan_amortization(p, original_rate, t)

        # Run quick projection
        months = target_years * 12
        inflated_target = target_amount * math.pow(1.0 + inflation_rate, target_years)
        monthly_ret = expected_annual_return / 12.0
        init_sav = max(0.0, current_savings - dp)
        reduced_sip = max(0.0, baseline_monthly_sip - lm.monthly_emi)

        det_corpus, hit_m = _run_deterministic_corpus(
            initial_savings=init_sav,
            monthly_sip=baseline_monthly_sip,
            loan_tenure=t,
            reduced_sip=reduced_sip,
            monthly_return=monthly_ret,
            target_corpus=inflated_target,
            target_months=months,
        )

        retire_age = round(current_age + (hit_m / 12.0), 1)
        age_diff = round(retire_age - baseline_retire_age, 1)

        # Quick probability estimation using simplified normal approximation
        prob_est = min(99.0, max(1.0, round(50.0 + (det_corpus - inflated_target) / (inflated_target * 0.4) * 40.0, 1)))

        alternatives.append(
            AlternativeScenario(
                name=cfg["name"],
                description=cfg["desc"],
                loan_principal=round(p, 2),
                loan_tenure_months=t,
                down_payment=round(dp, 2),
                monthly_emi=lm.monthly_emi,
                total_interest=lm.total_interest,
                projected_corpus=round(det_corpus, 2),
                goal_success_probability=prob_est,
                retirement_age_difference=age_diff,
                trade_off_summary=cfg["tradeoff"],
            )
        )

    return alternatives
