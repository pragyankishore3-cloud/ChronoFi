"""
LangChain Scenario & Memory Tools for ChronoFi.
Exposes typed tools for:
1. run_financial_scenario: Executing What-If financial simulations against the calculation engine
2. retrieve_previous_scenarios: Retrieving historical scenarios and stored simulation results
3. confirm_and_record_liability: Permanently recording a confirmed real liability after explicit user confirmation
"""

from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
import uuid

from langchain_core.runnables import RunnableConfig
from langchain_core.tools import tool
from pydantic import BaseModel, Field
from sqlalchemy import desc, select
from sqlalchemy.orm import selectinload

from engine.scenario_engine import ScenarioAssumption, project_retirement_simulation
import models.database as db_models
from models.database import Liability, ScenarioRecord, User


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
    parent_scenario_id: Optional[str] = Field(
        default=None,
        description="Optional ID of prior scenario being updated or versioned",
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
    parent_scenario_id: Optional[str] = None,
    notes: Optional[str] = None,
    user_id: Optional[int] = None,
    config: RunnableConfig = None,
) -> Dict[str, Any]:
    """
    Simulates a What-If financial scenario against the user's authorized financial profile.
    Calculates exact EMI, cash flow changes, baseline vs modified retirement corpus,
    retirement age delay, goal completion probability, and actionable alternatives.
    Enforces server-side user authorization. Persists scenario record with versioning.
    """
    if user_id is None and config and "configurable" in config:
        user_id = config["configurable"].get("user_id")

    thread_id = None
    if config and "configurable" in config:
        thread_id = config["configurable"].get("thread_id")

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

        # Versioning check
        version = 1
        if parent_scenario_id:
            parent_res = await db.execute(
                select(ScenarioRecord).where(
                    ScenarioRecord.id == parent_scenario_id,
                    ScenarioRecord.user_id == user_id,
                )
            )
            parent = parent_res.scalars().first()
            if parent:
                version = parent.version + 1

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
            estimated_monthly_expenses=user.monthly_expenses if (user.monthly_expenses and user.monthly_expenses > 0) else None,
            expected_annual_return=weighted_return,
            inflation_rate=inflation_rate,
            num_simulations=500,
            seed=42,
            scenario_type=scenario_type,
            assumptions_list=assumptions,
        )

        res_dict = sim_res.to_dict()
        res_dict["version"] = version
        res_dict["parent_scenario_id"] = parent_scenario_id

        # Persist ScenarioRecord for audit & memory retrieval
        scenario_id = str(uuid.uuid4())
        res_dict["scenario_id"] = scenario_id
        now_str = datetime.now(timezone.utc).isoformat()
        name_str = f"{scenario_type.replace('_', ' ').title()}: ₹{loan_principal/100000:.1f}L @ {annual_interest_rate*100:.1f}%"
        
        record = ScenarioRecord(
            id=scenario_id,
            user_id=user_id,
            thread_id=thread_id,
            name=name_str,
            scenario_type=scenario_type,
            parameters={
                "loan_principal": loan_principal,
                "annual_interest_rate": annual_interest_rate,
                "loan_tenure_months": loan_tenure_months,
                "down_payment": down_payment,
                "monthly_vehicle_expenses": monthly_vehicle_expenses,
                "monthly_savings_change": monthly_savings_change,
            },
            assumptions=[{"field": a.field, "value": a.value, "source": a.source, "description": a.description} for a in assumptions],
            baseline_id=res_dict.get("baseline_metrics", {}).get("projection_id", "baseline"),
            results_summary=res_dict.get("impact"),
            version=version,
            parent_scenario_id=parent_scenario_id,
            status="simulated",
            engine_version="1.0.0-crn",
            created_at=now_str,
            updated_at=now_str,
        )
        db.add(record)
        await db.commit()

        return res_dict


@tool
async def retrieve_previous_scenarios(
    search_query: Optional[str] = None,
    scenario_type: Optional[str] = None,
    scenario_id: Optional[str] = None,
    limit: int = 5,
    user_id: Optional[int] = None,
    config: RunnableConfig = None,
) -> Dict[str, Any]:
    """
    Retrieves previous financial scenarios, simulations, and assumptions
    from persistent storage for the authenticated user.
    Enforces server-side authorization: never returns scenarios of other users.
    Returns structured parameters, previous results, creation time, version, and status.
    """
    if user_id is None and config and "configurable" in config:
        user_id = config["configurable"].get("user_id")

    if not user_id:
        return {
            "status": "error",
            "error": "Unauthorized: No user_id in execution context. Cannot retrieve scenario history.",
            "scenarios": [],
        }

    async with db_models.AsyncSessionLocal() as db:
        query = select(ScenarioRecord).where(ScenarioRecord.user_id == user_id)

        if scenario_id:
            query = query.where(ScenarioRecord.id == scenario_id)
        if scenario_type:
            query = query.where(ScenarioRecord.scenario_type.ilike(f"%{scenario_type}%"))

        query = query.order_by(desc(ScenarioRecord.created_at)).limit(max(1, limit))
        result = await db.execute(query)
        records = result.scalars().all()

        scenarios_out: List[Dict[str, Any]] = []
        q_lower = (search_query or "").lower()

        for rec in records:
            # If search query provided, filter on type, name, or parameters text
            if q_lower:
                name_match = rec.name and q_lower in rec.name.lower()
                type_match = rec.scenario_type and q_lower in rec.scenario_type.lower()
                param_str = str(rec.parameters or {}).lower()
                param_match = any(token in param_str for token in q_lower.split())
                is_car = any(w in q_lower for w in ["car", "vehicle", "auto"]) and "vehicle" in (rec.scenario_type or "").lower()
                if not (name_match or type_match or param_match or is_car):
                    continue

            params = rec.parameters or {}
            scenarios_out.append({
                "scenario_id": rec.id,
                "user_id": rec.user_id,
                "thread_id": rec.thread_id,
                "name": rec.name or f"{rec.scenario_type} Scenario",
                "scenario_type": rec.scenario_type,
                "parameters": params,
                "assumptions": rec.assumptions or [],
                "baseline_id": rec.baseline_id,
                "results_summary": rec.results_summary or {},
                "version": rec.version,
                "parent_scenario_id": rec.parent_scenario_id,
                "status": rec.status,
                "engine_version": rec.engine_version,
                "created_at": rec.created_at,
                "updated_at": rec.updated_at,
                "is_current": True,
            })

        return {
            "status": "success",
            "count": len(scenarios_out),
            "scenarios": scenarios_out,
        }


@tool
async def confirm_and_record_liability(
    scenario_id: str,
    name: Optional[str] = "Car Loan",
    principal: Optional[float] = None,
    annual_interest_rate: Optional[float] = None,
    loan_tenure_months: Optional[int] = None,
    monthly_payment: Optional[float] = None,
    user_id: Optional[int] = None,
    config: RunnableConfig = None,
) -> Dict[str, Any]:
    """
    Confirms and records an actual financial liability in the user's permanent profile.
    Only invoked after the user explicitly confirms that a transaction actually took place
    (not merely decided to explore or proceed).
    Updates database Liability, sets scenario status to 'confirmed_liability',
    and returns the new liability ID, updated monthly surplus, and confirmation message.
    """
    if user_id is None and config and "configurable" in config:
        user_id = config["configurable"].get("user_id")

    if not user_id:
        return {
            "status": "error",
            "error": "Unauthorized: Cannot commit liability without verified user_id.",
        }

    async with db_models.AsyncSessionLocal() as db:
        # 1. Fetch the scenario record and verify ownership
        rec_res = await db.execute(
            select(ScenarioRecord).where(
                ScenarioRecord.id == scenario_id,
                ScenarioRecord.user_id == user_id,
            )
        )
        rec = rec_res.scalars().first()

        params = rec.parameters if rec else {}
        p_val = principal or params.get("loan_principal", 1000000.0)
        r_val = annual_interest_rate or params.get("annual_interest_rate", 0.09)
        t_val = loan_tenure_months or params.get("loan_tenure_months", 60)

        # Calculate exact EMI if not supplied
        if monthly_payment is None or monthly_payment <= 0:
            if r_val > 0 and t_val > 0:
                monthly_r = r_val / 12.0
                emi_calc = p_val * (monthly_r * ((1.0 + monthly_r) ** t_val)) / (((1.0 + monthly_r) ** t_val) - 1.0)
                monthly_payment = round(emi_calc, 2)
            else:
                monthly_payment = round(p_val / max(1, t_val), 2)

        # 2. Insert new Liability into permanent user profile
        liability = Liability(
            user_id=user_id,
            name=name or "Car Loan",
            principal_remaining=float(p_val),
            interest_rate=float(r_val),
            monthly_payment=float(monthly_payment),
        )
        db.add(liability)

        # 3. Update ScenarioRecord status to confirmed_liability
        now_str = datetime.now(timezone.utc).isoformat()
        if rec:
            rec.status = "confirmed_liability"
            rec.updated_at = now_str

        await db.commit()
        await db.refresh(liability)

        # 4. Fetch updated user context to compute new net monthly surplus
        user_res = await db.execute(
            select(User)
            .where(User.id == user_id)
            .options(
                selectinload(User.assets),
                selectinload(User.liabilities),
                selectinload(User.goals),
            )
        )
        user = user_res.scalars().first()
        total_monthly_liabilities = sum(l.monthly_payment for l in user.liabilities)
        new_surplus = max(0.0, user.base_salary - 70000.0 - total_monthly_liabilities)

        return {
            "status": "success",
            "liability_id": liability.id,
            "name": liability.name,
            "principal_remaining": liability.principal_remaining,
            "interest_rate": liability.interest_rate,
            "monthly_payment": liability.monthly_payment,
            "scenario_status": "confirmed_liability",
            "updated_total_monthly_emis": total_monthly_liabilities,
            "updated_monthly_surplus": new_surplus,
            "message": f"Successfully recorded {liability.name} of ₹{liability.principal_remaining:,.2f} at {liability.interest_rate*100:.1f}% APR with monthly EMI ₹{liability.monthly_payment:,.2f}.",
        }
