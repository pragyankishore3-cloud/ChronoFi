"""
LangChain Scenario Tools for ChronoFi.
Exposes typed tools for executing What-If financial simulations against the calculation engine.
"""

from typing import Any, Dict, Optional
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import tool
from pydantic import BaseModel, Field
from sqlalchemy import select
from sqlalchemy.orm import selectinload

from engine.scenario_engine import ScenarioAssumption, project_retirement_simulation
import models.database as db_models
from models.database import ScenarioRecord, User


class RunFinancialScenarioInput(BaseModel):
    scenario_type: str = Field(
        default="vehicle_loan",
        description="Type of financial scenario (e.g. vehicle_loan, home_loan, personal_loan, expense_change)",
    )
    loan_principal: float = Field(
        default=0.0,
        description="Loan principal amount in INR (e.g. 1000000 for ₹10 Lakh car loan)",
    )
    annual_interest_rate: float = Field(
        default=0.09,
        description="Annual interest rate as a decimal (e.g. 0.09 for 9% APR)",
    )
    loan_tenure_months: int = Field(
        default=60,
        description="Loan repayment tenure in months (e.g. 60 for 5 years)",
    )
    down_payment: float = Field(
        default=0.0,
        description="Upfront down payment contributed from existing cash or savings in INR",
    )
    monthly_vehicle_expenses: float = Field(
        default=0.0,
        description="Estimated ongoing monthly vehicle running costs (fuel, insurance, maintenance)",
    )
    monthly_savings_change: float = Field(
        default=0.0,
        description="Direct change to monthly discretionary investment savings",
    )
    notes: Optional[str] = Field(
        default=None,
        description="Optional context or user remarks regarding the scenario",
    )


@tool
async def run_financial_scenario(
    scenario_type: str = "vehicle_loan",
    loan_principal: float = 0.0,
    annual_interest_rate: float = 0.09,
    loan_tenure_months: int = 60,
    down_payment: float = 0.0,
    monthly_vehicle_expenses: float = 0.0,
    monthly_savings_change: float = 0.0,
    notes: Optional[str] = None,
    user_id: Optional[int] = None,
    config: RunnableConfig = None,
) -> Dict[str, Any]:
    """
    Simulates a What-If financial scenario against the user's authorized financial profile.
    Calculates exact EMI, cash flow changes, baseline vs modified retirement corpus,
    retirement age delay, goal completion probability, and actionable alternatives.
    Enforces server-side user authorization.
    """
    if user_id is None and config and "configurable" in config:
        user_id = config["configurable"].get("user_id")

    if not user_id:
        return {
            "status": "error",
            "error": "Unauthorized: No user_id in execution context. Cannot access financial profile.",
        }

    # Input validation
    if loan_principal < 0:
        return {"status": "error", "error": "Loan principal cannot be negative."}
    if annual_interest_rate < 0 or annual_interest_rate > 1.0:
        return {"status": "error", "error": "Annual interest rate must be between 0.0 and 1.0."}
    if loan_tenure_months <= 0 or loan_tenure_months > 360:
        return {"status": "error", "error": "Loan tenure must be between 1 and 360 months."}

    async with db_models.AsyncSessionLocal() as db:
        result = await db.execute(
            select(User)
            .where(User.id == user_id)
            .options(
                selectinload(User.assets),
                selectinload(User.liabilities),
                selectinload(User.goals),
            )
        )
        user = result.scalars().first()
        if not user:
            return {"status": "error", "error": f"User profile not found for user_id={user_id}."}

        # Resolve server-side profile data
        current_savings = sum(a.current_value for a in user.assets) if user.assets else 0.0
        existing_liabilities_monthly = sum(l.monthly_payment for l in user.liabilities) if user.liabilities else 0.0

        if user.assets and current_savings > 0:
            weighted_return = sum(a.current_value * (a.expected_return_rate or 0.10) for a in user.assets) / current_savings
        else:
            weighted_return = 0.105

        # Retirement goal resolution
        ret_goal = next((g for g in user.goals if "retire" in g.name.lower()), None)
        if not ret_goal and user.goals:
            ret_goal = max(user.goals, key=lambda g: g.target_years)

        if ret_goal:
            target_amount = ret_goal.target_amount
            target_years = ret_goal.target_years
            inflation_rate = ret_goal.inflation_rate or 0.06
        else:
            target_years = max(5, 60 - user.age)
            target_amount = (user.base_salary * 12.0) * 20.0
            inflation_rate = 0.06

        # Documented assumptions
        assumptions = [
            ScenarioAssumption("loan_principal", loan_principal, "user_provided" if loan_principal > 0 else "explicit_default", "Principal requested"),
            ScenarioAssumption("annual_interest_rate", annual_interest_rate, "user_provided", "Annual loan interest rate"),
            ScenarioAssumption("loan_tenure_months", loan_tenure_months, "explicit_default" if loan_tenure_months == 60 else "user_provided", "Repayment tenure"),
            ScenarioAssumption("down_payment", down_payment, "user_provided" if down_payment > 0 else "explicit_default", "Down payment from assets"),
            ScenarioAssumption("expected_annual_return", round(weighted_return, 4), "profile_retrieved", "Portfolio expected annual return"),
            ScenarioAssumption("inflation_rate", round(inflation_rate, 4), "profile_retrieved", "Expected long-term annual inflation"),
        ]

        # Execute Engine Simulation
        sim_res = project_retirement_simulation(
            current_age=user.age,
            base_salary=user.base_salary,
            existing_liabilities_monthly=existing_liabilities_monthly,
            current_savings=current_savings,
            target_amount=target_amount,
            target_years=target_years,
            loan_principal=loan_principal,
            annual_interest_rate=annual_interest_rate,
            loan_tenure_months=loan_tenure_months,
            down_payment=down_payment,
            monthly_vehicle_expenses=monthly_vehicle_expenses,
            expected_annual_return=weighted_return,
            inflation_rate=inflation_rate,
            num_simulations=500,
            seed=42,
            scenario_type=scenario_type,
            assumptions_list=assumptions,
        )

        return sim_res.to_dict()
