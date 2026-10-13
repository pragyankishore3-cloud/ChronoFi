"""
ChronoFi Liquidity Shock & Emergency Runway Stress Testing Engine.

Deterministically simulates adverse financial life events (job loss, medical emergency,
salary reduction) against a user's balance sheet, computing exact liquid survival runway,
asset liquidation order, insolvency timing, and recovery requirements.
"""

from dataclasses import asdict, dataclass
import json
import math
from typing import Any, Dict, List, Optional


@dataclass
class AssetTierLiquidation:
    tier_name: str
    asset_names: List[str]
    initial_value: float
    liquidated_value: float
    remaining_value: float
    friction_cost: float  # Penalties, taxes, or fire-sale haircuts


@dataclass
class MonthlyStressPoint:
    month: int
    income: float
    expenses: float
    debt_service: float
    net_deficit: float
    remaining_total_wealth: float
    tier1_liquid_remaining: float
    tier2_fd_remaining: float
    tier3_debt_remaining: float
    tier4_equity_remaining: float
    is_solvent: bool


@dataclass
class StressTestResult:
    scenario_name: str
    shock_type: str
    shock_duration_months: int
    monthly_burn_rate: float
    net_monthly_deficit: float
    total_shock_cost: float
    survival_runway_months: float
    is_solvent_through_shock: bool
    insolvency_month: Optional[int]
    total_wealth_before: float
    total_wealth_after: float
    total_friction_losses: float
    recommended_emergency_fund: float
    emergency_fund_gap: float
    resilience_rating: str  # "Robust", "Adequate", "Vulnerable", "Critical"
    asset_tiers: List[AssetTierLiquidation]
    monthly_trajectory: List[MonthlyStressPoint]
    actionable_advice: List[str]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def run_liquidity_stress_test(
    monthly_expenses: float,
    base_salary: float,
    existing_liabilities_monthly: float,
    assets: List[Dict[str, Any]],
    shock_type: str = "job_loss",
    shock_duration_months: int = 6,
    income_reduction_pct: float = 100.0,
    lump_sum_cost: float = 0.0,
) -> StressTestResult:
    """
    Simulates a liquidity shock against the user's balance sheet.
    
    Tiers of Liquidation:
    Tier 1: Cash, Savings, Ultra-Short Liquid (0% friction)
    Tier 2: Fixed Deposits / Term Deposits (1% premature exit penalty)
    Tier 3: Debt Mutual Funds / Bonds (2% friction / exit load)
    Tier 4: Equity & Risk Assets (15% fire-sale & volatility haircut)
    """
    if monthly_expenses < 0:
        raise ValueError("Monthly expenses cannot be negative.")
    if base_salary < 0:
        raise ValueError("Base salary cannot be negative.")
    if existing_liabilities_monthly < 0:
        raise ValueError("Liabilities cannot be negative.")
    if shock_duration_months <= 0:
        raise ValueError("Shock duration must be at least 1 month.")

    burn_rate = monthly_expenses + existing_liabilities_monthly
    recommended_emergency_fund = round(burn_rate * 6.0, 2)

    # 1. Bucket assets into tiers
    tier1_names, tier2_names, tier3_names, tier4_names = [], [], [], []
    t1_val, t2_val, t3_val, t4_val = 0.0, 0.0, 0.0, 0.0

    for a in assets:
        val = float(a.get("current_value", 0.0))
        atype = (a.get("asset_type") or "").lower()
        name = a.get("name", "Asset")

        if any(t in atype for t in ["cash", "liquid", "savings", "checking", "wallet"]):
            tier1_names.append(name)
            t1_val += val
        elif any(t in atype for t in ["fd", "fixed deposit", "recurring", "term"]):
            tier2_names.append(name)
            t2_val += val
        elif any(t in atype for t in ["debt", "bond", "liquid fund", "g-sec", "treasury"]):
            tier3_names.append(name)
            t3_val += val
        else:
            tier4_names.append(name)
            t4_val += val

    initial_total_wealth = t1_val + t2_val + t3_val + t4_val
    emergency_fund_gap = max(0.0, round(recommended_emergency_fund - t1_val, 2))

    # 2. Simulate month by month
    curr_t1, curr_t2, curr_t3, curr_t4 = t1_val, t2_val, t3_val, t4_val
    total_friction = 0.0
    t1_drained, t2_drained, t3_drained, t4_drained = 0.0, 0.0, 0.0, 0.0

    simulation_horizon = max(24, shock_duration_months + 12)
    trajectory: List[MonthlyStressPoint] = []
    insolvency_month: Optional[int] = None
    survived_months = 0.0

    # Handle upfront lump sum cost in Month 1 (e.g. medical emergency)
    pending_lump_sum = lump_sum_cost

    total_shock_outflow = 0.0

    for m in range(1, simulation_horizon + 1):
        if m <= shock_duration_months:
            reduced_factor = max(0.0, min(1.0, 1.0 - (income_reduction_pct / 100.0)))
            month_income = base_salary * reduced_factor
        else:
            month_income = base_salary  # Income restored

        month_outflow = burn_rate + (pending_lump_sum if m == 1 else 0.0)
        net_deficit = max(0.0, month_outflow - month_income)
        total_shock_outflow += net_deficit

        needed = net_deficit
        # Liquidation waterfall
        # Tier 1: Cash (0% friction)
        if needed > 0 and curr_t1 > 0:
            take = min(curr_t1, needed)
            curr_t1 -= take
            t1_drained += take
            needed -= take

        # Tier 2: FD (1% premature withdrawal penalty)
        if needed > 0 and curr_t2 > 0:
            gross_needed = needed / 0.99
            take_gross = min(curr_t2, gross_needed)
            net_received = take_gross * 0.99
            friction = take_gross * 0.01

            curr_t2 -= take_gross
            t2_drained += take_gross
            total_friction += friction
            needed -= net_received

        # Tier 3: Debt Mutual Funds (2% friction)
        if needed > 0 and curr_t3 > 0:
            gross_needed = needed / 0.98
            take_gross = min(curr_t3, gross_needed)
            net_received = take_gross * 0.98
            friction = take_gross * 0.02

            curr_t3 -= take_gross
            t3_drained += take_gross
            total_friction += friction
            needed -= net_received

        # Tier 4: Equity / Other (15% fire-sale / market drop haircut)
        if needed > 0 and curr_t4 > 0:
            gross_needed = needed / 0.85
            take_gross = min(curr_t4, gross_needed)
            net_received = take_gross * 0.85
            friction = take_gross * 0.15

            curr_t4 -= take_gross
            t4_drained += take_gross
            total_friction += friction
            needed -= net_received

        remaining_wealth = curr_t1 + curr_t2 + curr_t3 + curr_t4
        is_solvent = (needed <= 0.01)

        if not is_solvent and insolvency_month is None:
            insolvency_month = m

        if is_solvent and insolvency_month is None:
            survived_months += 1.0

        trajectory.append(
            MonthlyStressPoint(
                month=m,
                income=round(month_income, 2),
                expenses=round(monthly_expenses, 2),
                debt_service=round(existing_liabilities_monthly, 2),
                net_deficit=round(net_deficit, 2),
                remaining_total_wealth=round(remaining_wealth, 2),
                tier1_liquid_remaining=round(curr_t1, 2),
                tier2_fd_remaining=round(curr_t2, 2),
                tier3_debt_remaining=round(curr_t3, 2),
                tier4_equity_remaining=round(curr_t4, 2),
                is_solvent=is_solvent,
            )
        )

    # Calculate resilience rating
    is_solvent_through_shock = (insolvency_month is None or insolvency_month > shock_duration_months)
    final_wealth = round(curr_t1 + curr_t2 + curr_t3 + curr_t4, 2)

    if insolvency_month is not None and insolvency_month <= 3:
        rating = "Critical"
    elif insolvency_month is not None and insolvency_month <= shock_duration_months:
        rating = "Vulnerable"
    elif t1_drained <= t1_val and t2_drained == 0:
        rating = "Robust"
    else:
        rating = "Adequate"

    # Actionable advice formulation
    advice = []
    if t1_val < recommended_emergency_fund:
        advice.append(
            f"Build Emergency Reserve: Your pure liquid cash (₹{t1_val:,.0f}) covers only "
            f"{(t1_val / burn_rate):.1f} months of burn. Target ₹{recommended_emergency_fund:,.0f} (6 months)."
        )
    if t4_drained > 0:
        advice.append(
            f"Avoid Equity Fire-Sales: The shock forced liquidating ₹{t4_drained:,.0f} of equity, "
            f"incurring ₹{(t4_drained * 0.15):,.0f} in market timing/haircut friction."
        )
    if existing_liabilities_monthly > (burn_rate * 0.40):
        advice.append(
            f"High Fixed Debt Burden: Monthly debt service (₹{existing_liabilities_monthly:,.0f}) is "
            f"{(existing_liabilities_monthly / burn_rate * 100):.1f}% of total burn rate. Prioritize paying down high-cost liabilities."
        )
    if is_solvent_through_shock:
        advice.append(
            f"Solvent Through Shock: Your balance sheet successfully withstands the {shock_duration_months}-month shock "
            f"with ₹{final_wealth:,.0f} remaining net worth."
        )
    else:
        advice.append(
            f"Insolvency Risk at Month {insolvency_month}: Liquid reserves expire before shock completion. "
            f"Immediate debt restructuring or credit line buffer recommended."
        )

    asset_tiers = [
        AssetTierLiquidation(
            tier_name="Tier 1: Cash & Liquid",
            asset_names=tier1_names,
            initial_value=round(t1_val, 2),
            liquidated_value=round(t1_drained, 2),
            remaining_value=round(curr_t1, 2),
            friction_cost=0.0,
        ),
        AssetTierLiquidation(
            tier_name="Tier 2: Fixed Deposits",
            asset_names=tier2_names,
            initial_value=round(t2_val, 2),
            liquidated_value=round(t2_drained, 2),
            remaining_value=round(curr_t2, 2),
            friction_cost=round(t2_drained * 0.01, 2),
        ),
        AssetTierLiquidation(
            tier_name="Tier 3: Debt Mutual Funds",
            asset_names=tier3_names,
            initial_value=round(t3_val, 2),
            liquidated_value=round(t3_drained, 2),
            remaining_value=round(curr_t3, 2),
            friction_cost=round(t3_drained * 0.02, 2),
        ),
        AssetTierLiquidation(
            tier_name="Tier 4: Equity & Growth",
            asset_names=tier4_names,
            initial_value=round(t4_val, 2),
            liquidated_value=round(t4_drained, 2),
            remaining_value=round(curr_t4, 2),
            friction_cost=round(t4_drained * 0.15, 2),
        ),
    ]

    return StressTestResult(
        scenario_name=f"{shock_type.replace('_', ' ').title()} ({shock_duration_months} Months)",
        shock_type=shock_type,
        shock_duration_months=shock_duration_months,
        monthly_burn_rate=round(burn_rate, 2),
        net_monthly_deficit=round(burn_rate - (base_salary * (1.0 - income_reduction_pct/100.0)), 2),
        total_shock_cost=round(total_shock_outflow, 2),
        survival_runway_months=survived_months,
        is_solvent_through_shock=is_solvent_through_shock,
        insolvency_month=insolvency_month,
        total_wealth_before=round(initial_total_wealth, 2),
        total_wealth_after=final_wealth,
        total_friction_losses=round(total_friction, 2),
        recommended_emergency_fund=recommended_emergency_fund,
        emergency_fund_gap=emergency_fund_gap,
        resilience_rating=rating,
        asset_tiers=asset_tiers,
        monthly_trajectory=trajectory,
        actionable_advice=advice,
    )
