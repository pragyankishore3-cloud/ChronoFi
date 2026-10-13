"""
ChronoFi India Capital Gains (LTCG / STCG) Goal Drawdown Engine.

Implements deterministic capital gains taxation under India Finance Act 2024 / Budget 2024
rules for financial goal maturities and portfolio liquidations:
- Equity / Equity Mutual Funds:
    - Long-Term Capital Gains (LTCG > 12 months): 12.5% on gains exceeding ₹1.25 Lakh exemption limit.
    - Short-Term Capital Gains (STCG <= 12 months): 20%.
- Specified Debt Mutual Funds & Fixed Deposits:
    - Gains taxed at investor's marginal slab rate (e.g. 30% + 4% cess under Section 50AA).
- Tax-Exempt Instruments (PPF, SSY, EPF within limit, SGB till maturity):
    - 0% tax on principal and gains (EEE status).
- 4% Health and Education Cess applied across all taxes.
"""

from dataclasses import asdict, dataclass, field
import math
from typing import Any, Dict, List, Optional


# Budget 2024 Constants (Finance Act 2024)
BUDGET_2024_EQUITY_LTCG_RATE = 0.125  # 12.5%
BUDGET_2024_EQUITY_LTCG_EXEMPTION = 125000.0  # ₹1.25 Lakh per financial year
BUDGET_2024_EQUITY_STCG_RATE = 0.200  # 20.0%
HEALTH_AND_EDUCATION_CESS = 0.04  # 4%
DEFAULT_DEBT_MARGINAL_RATE = 0.300  # 30% default marginal slab rate


@dataclass
class AssetClassTaxBreakdown:
    asset_class: str  # "Equity", "Debt", "Tax-Exempt"
    allocation_pct: float
    allocated_corpus: float
    allocated_principal: float
    capital_gains: float
    tax_rate_applied_pct: float
    exemption_claimed: float
    taxable_gains: float
    tax_payable: float
    net_post_tax_corpus: float


@dataclass
class GoalTaxReport:
    goal_name: str
    target_years: int
    target_amount_nominal: float
    inflated_target_corpus: float
    inflation_rate: float
    gross_projected_corpus: float
    total_principal_invested: float
    total_capital_gains: float
    
    # Taxes
    equity_tax: float
    debt_tax: float
    cess_amount: float
    total_tax_liability: float
    effective_tax_rate_on_gains_pct: float
    effective_tax_drag_on_corpus_pct: float
    
    # Post-tax outcomes
    net_post_tax_corpus: float
    gross_funding_ratio_pct: float
    net_funding_ratio_pct: float
    is_fully_funded_gross: bool
    is_fully_funded_post_tax: bool
    post_tax_shortfall: float
    required_gross_target_to_meet_net: float
    
    # Asset breakdown
    asset_breakdowns: List[AssetClassTaxBreakdown]
    
    # Actionable optimization advice
    annual_tax_harvesting_benefit: float
    actionable_tax_tips: List[str]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def get_default_allocation_for_risk(risk_tolerance: str) -> Dict[str, float]:
    """
    Returns asset allocation weights (Equity, Debt, Tax-Exempt) based on risk tolerance profile.
    """
    normalized = (risk_tolerance or "Moderate").strip().lower()
    if "aggress" in normalized:
        return {"equity": 0.80, "debt": 0.15, "exempt": 0.05}
    elif "conserv" in normalized:
        return {"equity": 0.25, "debt": 0.55, "exempt": 0.20}
    else:  # Moderate
        return {"equity": 0.60, "debt": 0.30, "exempt": 0.10}


def calculate_goal_tax_drawdown(
    goal_name: str,
    target_years: int,
    target_amount: float,
    gross_projected_corpus: float,
    total_principal_invested: float,
    equity_pct: Optional[float] = None,
    debt_pct: Optional[float] = None,
    exempt_pct: Optional[float] = None,
    risk_tolerance: str = "Moderate",
    marginal_tax_rate: float = DEFAULT_DEBT_MARGINAL_RATE,
    inflation_rate: float = 0.06,
    annual_harvesting_practiced: bool = False,
) -> GoalTaxReport:
    """
    Calculates post-tax maturity value and capital gains tax liabilities for a specific goal
    under Indian taxation laws.
    """
    if target_years < 1:
        raise ValueError("Target years must be at least 1 year.")
    if target_amount < 0:
        raise ValueError("Target amount cannot be negative.")
    if gross_projected_corpus < 0:
        raise ValueError("Projected corpus cannot be negative.")
    if total_principal_invested < 0:
        raise ValueError("Principal invested cannot be negative.")

    # 1. Resolve asset weights
    if equity_pct is not None and debt_pct is not None:
        eq_w = max(0.0, float(equity_pct))
        db_w = max(0.0, float(debt_pct))
        ex_w = max(0.0, float(exempt_pct if exempt_pct is not None else (1.0 - eq_w - db_w)))
        total_w = eq_w + db_w + ex_w
        if total_w > 0:
            eq_w, db_w, ex_w = eq_w / total_w, db_w / total_w, ex_w / total_w
        else:
            eq_w, db_w, ex_w = 0.60, 0.30, 0.10
    else:
        alloc = get_default_allocation_for_risk(risk_tolerance)
        eq_w = alloc["equity"]
        db_w = alloc["debt"]
        ex_w = alloc["exempt"]

    # 2. Total Gains and Principal breakdown
    # If principal exceeds corpus (market loss), capital gains is 0
    total_gains = max(0.0, round(gross_projected_corpus - total_principal_invested, 2))
    
    eq_corpus = round(gross_projected_corpus * eq_w, 2)
    db_corpus = round(gross_projected_corpus * db_w, 2)
    ex_corpus = round(gross_projected_corpus * ex_w, 2)

    eq_principal = round(total_principal_invested * eq_w, 2)
    db_principal = round(total_principal_invested * db_w, 2)
    ex_principal = round(total_principal_invested * ex_w, 2)

    eq_gains = max(0.0, round(eq_corpus - eq_principal, 2))
    db_gains = max(0.0, round(db_corpus - db_principal, 2))
    ex_gains = max(0.0, round(ex_corpus - ex_principal, 2))

    # 3. Equity Tax Calculation
    # Horizon > 1 yr is LTCG under Section 112A
    is_equity_ltcg = target_years > 1
    if is_equity_ltcg:
        # Budget 2024: ₹1.25L exemption
        eq_exemption = min(eq_gains, BUDGET_2024_EQUITY_LTCG_EXEMPTION)
        taxable_eq_gains = max(0.0, round(eq_gains - eq_exemption, 2))
        base_eq_tax = taxable_eq_gains * BUDGET_2024_EQUITY_LTCG_RATE
        applied_eq_rate_pct = BUDGET_2024_EQUITY_LTCG_RATE * 100.0
    else:
        # STCG under Section 111A: 20% flat, no exemption
        eq_exemption = 0.0
        taxable_eq_gains = eq_gains
        base_eq_tax = taxable_eq_gains * BUDGET_2024_EQUITY_STCG_RATE
        applied_eq_rate_pct = BUDGET_2024_EQUITY_STCG_RATE * 100.0

    eq_cess = base_eq_tax * HEALTH_AND_EDUCATION_CESS
    eq_tax_total = round(base_eq_tax + eq_cess, 2)

    # 4. Debt Tax Calculation
    # Under Section 50AA, debt mutual fund gains are taxed at investor's marginal slab rate
    taxable_db_gains = db_gains
    base_db_tax = taxable_db_gains * marginal_tax_rate
    db_cess = base_db_tax * HEALTH_AND_EDUCATION_CESS
    db_tax_total = round(base_db_tax + db_cess, 2)

    # 5. Tax-Exempt Instruments (PPF, SSY, EPF within limit)
    ex_tax_total = 0.0

    # 6. Aggregates
    total_tax_liability = round(eq_tax_total + db_tax_total + ex_tax_total, 2)
    total_cess = round(eq_cess + db_cess, 2)
    net_post_tax_corpus = round(gross_projected_corpus - total_tax_liability, 2)

    inflated_target = round(target_amount * math.pow(1.0 + inflation_rate, target_years), 2)
    gross_funding_ratio = round((gross_projected_corpus / inflated_target * 100.0), 1) if inflated_target > 0 else 100.0
    net_funding_ratio = round((net_post_tax_corpus / inflated_target * 100.0), 1) if inflated_target > 0 else 100.0

    is_funded_gross = gross_projected_corpus >= inflated_target
    is_funded_net = net_post_tax_corpus >= inflated_target
    post_tax_shortfall = max(0.0, round(inflated_target - net_post_tax_corpus, 2))

    effective_gain_tax_rate = round((total_tax_liability / total_gains * 100.0), 2) if total_gains > 0 else 0.0
    tax_drag_pct = round((total_tax_liability / gross_projected_corpus * 100.0), 2) if gross_projected_corpus > 0 else 0.0

    # Gross-up estimation: what gross corpus is required to yield the inflated target after taxes?
    if tax_drag_pct > 0 and tax_drag_pct < 100:
        required_gross = round(inflated_target / (1.0 - (tax_drag_pct / 100.0)), 2)
    else:
        required_gross = inflated_target

    # Asset breakdowns
    breakdowns = [
        AssetClassTaxBreakdown(
            asset_class="Equity (Mutual Funds / Stocks)",
            allocation_pct=round(eq_w * 100.0, 1),
            allocated_corpus=eq_corpus,
            allocated_principal=eq_principal,
            capital_gains=eq_gains,
            tax_rate_applied_pct=round(applied_eq_rate_pct * (1 + HEALTH_AND_EDUCATION_CESS), 2),
            exemption_claimed=eq_exemption,
            taxable_gains=taxable_eq_gains,
            tax_payable=eq_tax_total,
            net_post_tax_corpus=round(eq_corpus - eq_tax_total, 2),
        ),
        AssetClassTaxBreakdown(
            asset_class="Debt (Fixed Income / Debt MFs)",
            allocation_pct=round(db_w * 100.0, 1),
            allocated_corpus=db_corpus,
            allocated_principal=db_principal,
            capital_gains=db_gains,
            tax_rate_applied_pct=round(marginal_tax_rate * (1 + HEALTH_AND_EDUCATION_CESS) * 100.0, 2),
            exemption_claimed=0.0,
            taxable_gains=taxable_db_gains,
            tax_payable=db_tax_total,
            net_post_tax_corpus=round(db_corpus - db_tax_total, 2),
        ),
        AssetClassTaxBreakdown(
            asset_class="Tax-Exempt (PPF / SSY / EEE)",
            allocation_pct=round(ex_w * 100.0, 1),
            allocated_corpus=ex_corpus,
            allocated_principal=ex_principal,
            capital_gains=ex_gains,
            tax_rate_applied_pct=0.0,
            exemption_claimed=ex_gains,
            taxable_gains=0.0,
            tax_payable=0.0,
            net_post_tax_corpus=ex_corpus,
        ),
    ]

    # Actionable optimization advice
    annual_harvesting_benefit = round(BUDGET_2024_EQUITY_LTCG_EXEMPTION * BUDGET_2024_EQUITY_LTCG_RATE * (1 + HEALTH_AND_EDUCATION_CESS), 2)
    advice: List[str] = []

    if eq_tax_total > 0 and target_years > 2:
        advice.append(
            f"Tax Harvesting Opportunity: By realizing up to ₹1,25,000 in equity gains each financial year and reinvesting, "
            f"you can save up to ₹{annual_harvesting_benefit:,.0f}/year in capital gains tax over {target_years} years."
        )

    if db_tax_total > eq_tax_total and db_w >= 0.25:
        advice.append(
            "Debt Tax Drag Mitigation: Debt funds are taxed at your marginal slab rate (Section 50AA). "
            "Consider shifting debt allocation to Arbitrage Funds (equity taxation with low volatility) "
            "or Tax-Free Sovereign Gold Bonds / PPF to cut tax drag significantly."
        )

    if is_funded_gross and not is_funded_net:
        advice.append(
            f"Tax Gap Alert: Your gross projected corpus of ₹{gross_projected_corpus:,.0f} meets the target, "
            f"but after taxes of ₹{total_tax_liability:,.0f}, you face a post-tax shortfall of ₹{post_tax_shortfall:,.0f}. "
            f"Increase monthly SIP by ~{round((post_tax_shortfall / (target_years * 12)), 0):,.0f} to cover tax friction."
        )

    if not advice:
        advice.append("Tax structure is highly optimized with low friction drag.")

    return GoalTaxReport(
        goal_name=goal_name,
        target_years=target_years,
        target_amount_nominal=round(target_amount, 2),
        inflated_target_corpus=inflated_target,
        inflation_rate=round(inflation_rate, 4),
        gross_projected_corpus=round(gross_projected_corpus, 2),
        total_principal_invested=round(total_principal_invested, 2),
        total_capital_gains=total_gains,
        equity_tax=eq_tax_total,
        debt_tax=db_tax_total,
        cess_amount=total_cess,
        total_tax_liability=total_tax_liability,
        effective_tax_rate_on_gains_pct=effective_gain_tax_rate,
        effective_tax_drag_on_corpus_pct=tax_drag_pct,
        net_post_tax_corpus=net_post_tax_corpus,
        gross_funding_ratio_pct=gross_funding_ratio,
        net_funding_ratio_pct=net_funding_ratio,
        is_fully_funded_gross=is_funded_gross,
        is_fully_funded_post_tax=is_funded_net,
        post_tax_shortfall=post_tax_shortfall,
        required_gross_target_to_meet_net=required_gross,
        asset_breakdowns=breakdowns,
        annual_tax_harvesting_benefit=annual_harvesting_benefit,
        actionable_tax_tips=advice,
    )


def calculate_portfolio_goals_tax(
    goals: List[Dict[str, Any]],
    risk_tolerance: str = "Moderate",
    marginal_tax_rate: float = DEFAULT_DEBT_MARGINAL_RATE,
) -> Dict[str, Any]:
    """
    Evaluates multi-goal portfolio capital gains tax impact across all user goals.
    """
    reports: List[GoalTaxReport] = []
    total_gross = 0.0
    total_net = 0.0
    total_tax = 0.0

    for g in goals:
        name = g.get("name", "Goal")
        target_yrs = max(1, int(g.get("target_years", 5)))
        target_amt = float(g.get("target_amount", 0.0))
        gross_corpus = float(g.get("gross_projected_corpus") or (target_amt * 1.10))
        invested_prin = float(g.get("total_principal_invested") or (gross_corpus * 0.60))

        rpt = calculate_goal_tax_drawdown(
            goal_name=name,
            target_years=target_yrs,
            target_amount=target_amt,
            gross_projected_corpus=gross_corpus,
            total_principal_invested=invested_prin,
            equity_pct=g.get("equity_pct"),
            debt_pct=g.get("debt_pct"),
            exempt_pct=g.get("exempt_pct"),
            risk_tolerance=risk_tolerance,
            marginal_tax_rate=marginal_tax_rate,
            inflation_rate=float(g.get("inflation_rate", 0.06)),
        )
        reports.append(rpt)
        total_gross += rpt.gross_projected_corpus
        total_net += rpt.net_post_tax_corpus
        total_tax += rpt.total_tax_liability

    return {
        "total_gross_corpus": round(total_gross, 2),
        "total_net_post_tax_corpus": round(total_net, 2),
        "total_tax_liability": round(total_tax, 2),
        "overall_tax_drag_pct": round((total_tax / total_gross * 100.0), 2) if total_gross > 0 else 0.0,
        "goals_evaluated": [r.to_dict() for r in reports],
    }
