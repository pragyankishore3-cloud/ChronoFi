"""
ChronoFi Debt Acceleration Engine (Avalanche vs Snowball Optimizer).

Provides deterministic amortization, prepayment acceleration, and repayment strategy
comparisons for personal balance sheets:
1. Baseline: Standard minimum contractual EMIs without prepayment.
2. Debt Avalanche: Extra prepayment allocated to the highest interest rate liability
   (mathematically optimal, minimizes total interest paid).
3. Debt Snowball: Extra prepayment allocated to the smallest balance liability
   (behaviorally optimal, maximizes rapid closure momentum).
"""

import copy
from dataclasses import asdict, dataclass, field
import math
from typing import Any, Dict, List, Optional


@dataclass
class LiabilityInput:
    name: str
    principal: float
    annual_interest_rate: float
    monthly_payment: float
    id: Optional[int] = None


@dataclass
class LoanPayoffMilestone:
    liability_id: Optional[int]
    liability_name: str
    initial_principal: float
    annual_interest_rate: float
    payoff_month: int
    total_interest_paid: float


@dataclass
class MonthlyDebtTrajectoryPoint:
    month: int
    total_remaining_balance: float
    total_interest_paid_cumulative: float
    total_principal_paid_cumulative: float
    active_loans_count: int


@dataclass
class DebtStrategyResult:
    strategy_name: str  # "Baseline", "Debt Avalanche", "Debt Snowball"
    description: str
    months_to_debt_free: int
    total_interest_paid: float
    total_principal_paid: float
    total_amount_paid: float
    interest_saved_vs_baseline: float
    months_saved_vs_baseline: int
    loan_milestones: List[LoanPayoffMilestone]
    trajectory: List[MonthlyDebtTrajectoryPoint]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class DebtOptimizerResult:
    total_initial_debt: float
    total_monthly_contractual_emi: float
    extra_monthly_prepayment: float
    lump_sum_prepayment: float
    baseline: DebtStrategyResult
    avalanche: DebtStrategyResult
    snowball: DebtStrategyResult
    recommended_strategy: str
    financial_savings_diff_avalanche_vs_snowball: float
    executive_summary: str
    actionable_takeaways: List[str]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _simulate_single_strategy(
    liabilities: List[LiabilityInput],
    strategy_name: str,
    extra_monthly_prepayment: float = 0.0,
    lump_sum_prepayment: float = 0.0,
    baseline_interest: Optional[float] = None,
    baseline_months: Optional[int] = None,
    max_months_cap: int = 360,
) -> DebtStrategyResult:
    """
    Simulates monthly debt amortization under a specific priority ordering.
    """
    if not liabilities:
        return DebtStrategyResult(
            strategy_name=strategy_name,
            description="No active liabilities found.",
            months_to_debt_free=0,
            total_interest_paid=0.0,
            total_principal_paid=0.0,
            total_amount_paid=0.0,
            interest_saved_vs_baseline=0.0,
            months_saved_vs_baseline=0,
            loan_milestones=[],
            trajectory=[],
        )

    # Active loan state
    loans = []
    for l in liabilities:
        if l.principal > 0:
            loans.append({
                "id": l.id,
                "name": l.name,
                "balance": float(l.principal),
                "initial_principal": float(l.principal),
                "annual_rate": float(l.annual_interest_rate),
                "monthly_rate": float(l.annual_interest_rate) / 12.0,
                "contractual_emi": float(l.monthly_payment),
                "interest_paid": 0.0,
                "payoff_month": None,
            })

    total_initial_principal = sum(item["balance"] for item in loans)

    # Apply immediate lump-sum prepayment in Month 0 if any
    if lump_sum_prepayment > 0:
        remaining_lump = lump_sum_prepayment
        if strategy_name == "Debt Avalanche":
            # Target highest rate
            sorted_for_lump = sorted(loans, key=lambda x: -x["annual_rate"])
        elif strategy_name == "Debt Snowball":
            # Target lowest balance
            sorted_for_lump = sorted(loans, key=lambda x: x["balance"])
        else:
            sorted_for_lump = sorted(loans, key=lambda x: -x["annual_rate"])

        for l in sorted_for_lump:
            if remaining_lump <= 0:
                break
            pay = min(l["balance"], remaining_lump)
            l["balance"] -= pay
            remaining_lump -= pay

    month = 0
    milestones: List[LoanPayoffMilestone] = []
    trajectory: List[MonthlyDebtTrajectoryPoint] = []
    total_interest_accum = 0.0
    total_principal_accum = 0.0

    while month < max_months_cap:
        active_loans = [l for l in loans if l["balance"] > 0.01]
        if not active_loans:
            break

        month += 1
        month_interest_paid = 0.0
        month_principal_paid = 0.0

        # 1. Accrue monthly interest on all active loans
        for l in active_loans:
            interest_accrued = l["balance"] * l["monthly_rate"]
            l["interest_paid"] += interest_accrued
            month_interest_paid += interest_accrued
            total_interest_accum += interest_accrued

        # 2. Minimum contractual payment on each active loan
        freed_up_snowball_cash = 0.0
        for l in active_loans:
            interest_accrued = l["balance"] * l["monthly_rate"]
            # Payment covers accrued interest first, remainder goes to principal
            payment = min(l["contractual_emi"], l["balance"] + interest_accrued)
            principal_paid = max(0.0, payment - interest_accrued)
            
            l["balance"] = max(0.0, l["balance"] + interest_accrued - payment)
            month_principal_paid += principal_paid
            total_principal_accum += principal_paid

            if l["balance"] <= 0.01 and l["payoff_month"] is None:
                l["payoff_month"] = month
                milestones.append(
                    LoanPayoffMilestone(
                        liability_id=l["id"],
                        liability_name=l["name"],
                        initial_principal=round(l["initial_principal"], 2),
                        annual_interest_rate=round(l["annual_rate"], 4),
                        payoff_month=month,
                        total_interest_paid=round(l["interest_paid"], 2),
                    )
                )

        # 3. Allocate extra prepayment + snowballed freed cash from already closed loans
        # In Snowball/Avalanche, freed-up EMIs from paid-off loans roll into acceleration pool
        if strategy_name != "Baseline":
            paid_off_loans = [l for l in loans if l["balance"] <= 0.01]
            snowball_rollover = sum(l["contractual_emi"] for l in paid_off_loans)
            extra_available = extra_monthly_prepayment + snowball_rollover
        else:
            extra_available = 0.0

        if extra_available > 0:
            remaining_active = [l for l in loans if l["balance"] > 0.01]
            if remaining_active:
                if strategy_name == "Debt Avalanche":
                    # Priority: highest interest rate first
                    target_loan = max(remaining_active, key=lambda x: (x["annual_rate"], -x["balance"]))
                else:  # Debt Snowball
                    # Priority: lowest outstanding balance first
                    target_loan = min(remaining_active, key=lambda x: (x["balance"], -x["annual_rate"]))

                extra_pay = min(target_loan["balance"], extra_available)
                target_loan["balance"] -= extra_pay
                month_principal_paid += extra_pay
                total_principal_accum += extra_pay

                if target_loan["balance"] <= 0.01 and target_loan["payoff_month"] is None:
                    target_loan["payoff_month"] = month
                    milestones.append(
                        LoanPayoffMilestone(
                            liability_id=target_loan["id"],
                            liability_name=target_loan["name"],
                            initial_principal=round(target_loan["initial_principal"], 2),
                            annual_interest_rate=round(target_loan["annual_rate"], 4),
                            payoff_month=month,
                            total_interest_paid=round(target_loan["interest_paid"], 2),
                        )
                    )

        rem_balance = sum(l["balance"] for l in loans if l["balance"] > 0.01)
        active_count = len([l for l in loans if l["balance"] > 0.01])
        trajectory.append(
            MonthlyDebtTrajectoryPoint(
                month=month,
                total_remaining_balance=round(rem_balance, 2),
                total_interest_paid_cumulative=round(total_interest_accum, 2),
                total_principal_paid_cumulative=round(total_principal_accum, 2),
                active_loans_count=active_count,
            )
        )

    # Any remaining loans cap out at max_months_cap
    for l in loans:
        if l["payoff_month"] is None:
            l["payoff_month"] = month
            milestones.append(
                LoanPayoffMilestone(
                    liability_id=l["id"],
                    liability_name=l["name"],
                    initial_principal=round(l["initial_principal"], 2),
                    annual_interest_rate=round(l["annual_rate"], 4),
                    payoff_month=month,
                    total_interest_paid=round(l["interest_paid"], 2),
                )
            )

    desc_map = {
        "Baseline": "Standard contractual minimum EMIs with zero prepayment acceleration.",
        "Debt Avalanche": "Aggressively targets the loan with the highest interest rate first to minimize total interest friction.",
        "Debt Snowball": "Aggressively eliminates the loan with the smallest balance first for rapid psychological wins.",
    }

    tot_interest = round(total_interest_accum, 2)
    interest_saved = max(0.0, round((baseline_interest or tot_interest) - tot_interest, 2)) if baseline_interest else 0.0
    months_saved = max(0, (baseline_months or month) - month) if baseline_months else 0

    return DebtStrategyResult(
        strategy_name=strategy_name,
        description=desc_map.get(strategy_name, ""),
        months_to_debt_free=month,
        total_interest_paid=tot_interest,
        total_principal_paid=round(total_initial_principal, 2),
        total_amount_paid=round(total_initial_principal + tot_interest, 2),
        interest_saved_vs_baseline=interest_saved,
        months_saved_vs_baseline=months_saved,
        loan_milestones=milestones,
        trajectory=trajectory,
    )


def compare_debt_acceleration(
    liabilities: List[Dict[str, Any]],
    extra_monthly_prepayment: float = 0.0,
    lump_sum_prepayment: float = 0.0,
) -> DebtOptimizerResult:
    """
    Compares Baseline, Avalanche, and Snowball strategies for a set of personal loans.
    """
    if extra_monthly_prepayment < 0:
        raise ValueError("Extra monthly prepayment cannot be negative.")
    if lump_sum_prepayment < 0:
        raise ValueError("Lump-sum prepayment cannot be negative.")

    parsed: List[LiabilityInput] = []
    for idx, l in enumerate(liabilities):
        p = float(l.get("principal") or l.get("principal_remaining") or 0.0)
        rate = float(l.get("annual_interest_rate") or l.get("interest_rate") or 0.0)
        emi = float(l.get("monthly_payment") or 0.0)
        name = str(l.get("name") or f"Loan {idx+1}")
        lid = l.get("id")
        if p > 0:
            parsed.append(
                LiabilityInput(
                    id=lid,
                    name=name,
                    principal=p,
                    annual_interest_rate=rate,
                    monthly_payment=emi,
                )
            )

    tot_principal = sum(item.principal for item in parsed)
    tot_contractual_emi = sum(item.monthly_payment for item in parsed)

    # 1. Baseline simulation
    baseline_res = _simulate_single_strategy(
        liabilities=parsed,
        strategy_name="Baseline",
        extra_monthly_prepayment=0.0,
        lump_sum_prepayment=0.0,
    )

    # 2. Avalanche simulation
    avalanche_res = _simulate_single_strategy(
        liabilities=parsed,
        strategy_name="Debt Avalanche",
        extra_monthly_prepayment=extra_monthly_prepayment,
        lump_sum_prepayment=lump_sum_prepayment,
        baseline_interest=baseline_res.total_interest_paid,
        baseline_months=baseline_res.months_to_debt_free,
    )

    # 3. Snowball simulation
    snowball_res = _simulate_single_strategy(
        liabilities=parsed,
        strategy_name="Debt Snowball",
        extra_monthly_prepayment=extra_monthly_prepayment,
        lump_sum_prepayment=lump_sum_prepayment,
        baseline_interest=baseline_res.total_interest_paid,
        baseline_months=baseline_res.months_to_debt_free,
    )

    savings_diff = round(snowball_res.total_interest_paid - avalanche_res.total_interest_paid, 2)
    
    # Recommendation logic
    if not parsed:
        recommended = "None"
        summary = "You currently have no active debt liabilities."
    elif savings_diff > 10000.0:
        recommended = "Debt Avalanche"
        summary = (
            f"Debt Avalanche is strongly recommended. It saves ₹{avalanche_res.interest_saved_vs_baseline:,.0f} in interest "
            f"(₹{savings_diff:,.0f} more than Snowball) and makes you debt-free {avalanche_res.months_saved_vs_baseline} months earlier."
        )
    elif snowball_res.months_to_debt_free < avalanche_res.months_to_debt_free:
        recommended = "Debt Snowball"
        summary = (
            f"Debt Snowball is recommended for earlier full debt payoff ({snowball_res.months_to_debt_free} months vs "
            f"{avalanche_res.months_to_debt_free} months) with negligible interest difference (₹{savings_diff:,.0f})."
        )
    else:
        recommended = "Debt Avalanche"
        summary = (
            f"Debt Avalanche provides maximum financial efficiency, saving ₹{avalanche_res.interest_saved_vs_baseline:,.0f} "
            f"in total interest and completing debt elimination in {avalanche_res.months_to_debt_free} months."
        )

    takeaways: List[str] = []
    if extra_monthly_prepayment > 0:
        takeaways.append(
            f"Prepayment Impact: An extra ₹{extra_monthly_prepayment:,.0f}/month cuts {avalanche_res.months_saved_vs_baseline} months "
            f"({round(avalanche_res.months_saved_vs_baseline / 12, 1)} years) off your repayment horizon."
        )
    if savings_diff > 0:
        takeaways.append(
            f"Strategy Difference: Choosing Avalanche over Snowball saves an additional ₹{savings_diff:,.0f} in pure interest."
        )
    if lump_sum_prepayment > 0:
        takeaways.append(
            f"Lump-Sum Injection: Applying ₹{lump_sum_prepayment:,.0f} upfront immediately reduces interest drag across all active accounts."
        )

    return DebtOptimizerResult(
        total_initial_debt=round(tot_principal, 2),
        total_monthly_contractual_emi=round(tot_contractual_emi, 2),
        extra_monthly_prepayment=round(extra_monthly_prepayment, 2),
        lump_sum_prepayment=round(lump_sum_prepayment, 2),
        baseline=baseline_res,
        avalanche=avalanche_res,
        snowball=snowball_res,
        recommended_strategy=recommended,
        financial_savings_diff_avalanche_vs_snowball=savings_diff,
        executive_summary=summary,
        actionable_takeaways=takeaways,
    )
