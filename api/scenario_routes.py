"""
ChronoFi What-If Financial Scenarios API.
Exposes versioned endpoints for simulating hypothetical financial decisions
against the authenticated user's authorized baseline plan.
"""

from datetime import datetime, timezone
import uuid
from typing import List, Optional

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from api import deps
from engine.scenario_engine import (
    ScenarioAssumption,
    project_retirement_simulation,
)
from models import database, schemas

from sqlalchemy.orm import selectinload

router = APIRouter()


@router.post(
    "/simulate",
    response_model=schemas.SimulateScenarioResponse,
    status_code=status.HTTP_200_OK,
    summary="Simulate What-If Financial Scenario",
)
async def simulate_financial_scenario(
    request: schemas.SimulateScenarioRequest,
    current_user: database.User = Depends(deps.get_current_user),
    db: AsyncSession = Depends(deps.get_db),
):
    """
    Executes a high-precision deterministic and paired Monte Carlo simulation
    comparing the authenticated user's baseline financial plan with requested changes.
    Enforces user isolation, returns structured math results, and generates actionable alternatives.
    """
    # 1. Fetch user with eager relationships to ensure isolation
    result = await db.execute(
        select(database.User)
        .where(database.User.id == current_user.id)
        .options(
            selectinload(database.User.assets),
            selectinload(database.User.liabilities),
            selectinload(database.User.goals),
        )
    )
    user = result.scalars().first()
    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User financial profile not found.",
        )

    # 2. Check Idempotency if provided
    scenario_uuid = str(uuid.uuid4())
    if request.idempotency_key:
        idempotency_id = f"idem_{current_user.id}_{request.idempotency_key}"
        existing = await db.execute(
            select(database.ScenarioRecord).where(database.ScenarioRecord.id == idempotency_id)
        )
        cached_record = existing.scalars().first()
        if cached_record and cached_record.results_summary:
            return cached_record.results_summary

    # 3. Resolve User Financial Baseline server-side
    current_savings = sum(a.current_value for a in user.assets) if user.assets else 0.0
    existing_liabilities_monthly = sum(l.monthly_payment for l in user.liabilities) if user.liabilities else 0.0

    # Calculate weighted expected return from user's assets
    if user.assets and current_savings > 0:
        weighted_return = sum(a.current_value * (a.expected_return_rate or 0.10) for a in user.assets) / current_savings
    else:
        weighted_return = 0.105

    # Find retirement goal or fallback
    retirement_goal = next((g for g in user.goals if "retire" in g.name.lower()), None)
    if not retirement_goal and user.goals:
        # Pick the longest horizon goal
        retirement_goal = max(user.goals, key=lambda g: g.target_years)

    if retirement_goal:
        target_amount = retirement_goal.target_amount
        target_years = retirement_goal.target_years
        goal_inflation = retirement_goal.inflation_rate or 0.06
        goal_source = "profile_retrieved"
    else:
        # Default retirement target if no goal set
        target_years = max(5, 60 - user.age)
        target_amount = (user.base_salary * 12.0) * 20.0  # 20x annual income rule of thumb
        goal_inflation = 0.06
        goal_source = "explicit_default"

    # 4. Handle Assumption Overrides if provided
    overrides = request.assumptions_override or schemas.AssumptionOverride()
    final_return = overrides.expected_annual_return if overrides.expected_annual_return is not None else weighted_return
    final_inflation = overrides.inflation_rate if overrides.inflation_rate is not None else goal_inflation
    final_expenses = overrides.estimated_monthly_expenses

    # 5. Extract & Validate Changes
    changes = request.changes
    if changes.loan_principal < 0:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="loan_principal cannot be negative.",
        )
    if changes.annual_interest_rate < 0 or changes.annual_interest_rate > 1.0:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="annual_interest_rate must be between 0.0 and 1.0 (e.g. 0.09 for 9%).",
        )
    if changes.loan_tenure_months <= 0 or changes.loan_tenure_months > 360:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="loan_tenure_months must be between 1 and 360.",
        )

    # 6. Build Documented Assumptions
    assumptions_list = [
        ScenarioAssumption(
            field="loan_principal",
            value=changes.loan_principal,
            source="user_provided" if changes.loan_principal > 0 else "explicit_default",
            description="Principal amount requested for vehicle financing",
        ),
        ScenarioAssumption(
            field="annual_interest_rate",
            value=changes.annual_interest_rate,
            source="user_provided",
            description="Annual vehicle loan interest rate (APR)",
        ),
        ScenarioAssumption(
            field="loan_tenure_months",
            value=changes.loan_tenure_months,
            source="user_provided" if changes.loan_tenure_months != 60 else "explicit_default",
            description="Loan repayment duration in months",
        ),
        ScenarioAssumption(
            field="down_payment",
            value=changes.down_payment,
            source="user_provided" if changes.down_payment > 0 else "explicit_default",
            description="Upfront capital contributed from current savings",
        ),
        ScenarioAssumption(
            field="expected_annual_return",
            value=round(final_return, 4),
            source="profile_retrieved" if overrides.expected_annual_return is None else "user_override",
            description="Portfolio weighted compound annual growth rate",
        ),
        ScenarioAssumption(
            field="inflation_rate",
            value=round(final_inflation, 4),
            source=goal_source if overrides.inflation_rate is None else "user_override",
            description="Annual cost-of-living inflation rate",
        ),
    ]

    # 7. Execute Simulation via Financial Engine
    sim_settings = request.simulation_settings
    horizon = sim_settings.horizon_years or target_years

    user_goals_data = [
        {
            "id": g.id,
            "name": g.name,
            "target_amount": g.target_amount,
            "target_years": g.target_years,
            "priority": g.priority.value if hasattr(g.priority, "value") else str(g.priority),
            "inflation_rate": g.inflation_rate or 0.06,
        }
        for g in user.goals
    ] if user.goals else None

    sim_result = project_retirement_simulation(
        current_age=user.age,
        base_salary=user.base_salary,
        existing_liabilities_monthly=existing_liabilities_monthly,
        current_savings=current_savings,
        target_amount=target_amount,
        target_years=horizon,
        loan_principal=changes.loan_principal,
        annual_interest_rate=changes.annual_interest_rate,
        loan_tenure_months=changes.loan_tenure_months,
        down_payment=changes.down_payment,
        monthly_vehicle_expenses=changes.monthly_vehicle_expenses,
        estimated_monthly_expenses=final_expenses,
        expected_annual_return=final_return,
        inflation_rate=final_inflation,
        num_simulations=sim_settings.num_simulations,
        seed=sim_settings.seed,
        scenario_id=scenario_uuid,
        scenario_type=request.scenario_type,
        assumptions_list=assumptions_list,
        user_goals=user_goals_data,
    )

    result_dict = sim_result.to_dict()

    # 8. Persist Scenario Record for Auditing & History (Read-only for actual financial assets)
    record_id = f"idem_{current_user.id}_{request.idempotency_key}" if request.idempotency_key else scenario_uuid
    scenario_record = database.ScenarioRecord(
        id=record_id,
        user_id=current_user.id,
        scenario_type=request.scenario_type,
        parameters=request.changes.model_dump(),
        baseline_id=f"base_{current_user.id}",
        results_summary=result_dict,
        created_at=datetime.now(timezone.utc).isoformat(),
    )
    db.add(scenario_record)
    await db.commit()

    return result_dict


@router.get(
    "/user-scenarios",
    response_model=List[schemas.SimulateScenarioResponse],
    summary="Get User's Past Scenario Simulations",
)
async def get_user_scenarios(
    current_user: database.User = Depends(deps.get_current_user),
    db: AsyncSession = Depends(deps.get_db),
):
    """
    Returns recent scenario simulations executed by the authenticated user.
    """
    result = await db.execute(
        select(database.ScenarioRecord)
        .where(database.ScenarioRecord.user_id == current_user.id)
        .order_by(database.ScenarioRecord.created_at.desc())
        .limit(10)
    )
    records = result.scalars().all()
    return [r.results_summary for r in records if r.results_summary]


@router.post(
    "/confirm-liability",
    response_model=schemas.ConfirmLiabilityResponse,
    status_code=status.HTTP_200_OK,
    summary="Confirm Scenario and Record Real Liability",
)
async def confirm_liability_endpoint(
    request: schemas.ConfirmLiabilityRequest,
    current_user: database.User = Depends(deps.get_current_user),
    db: AsyncSession = Depends(deps.get_db),
):
    """
    Confirms an approved hypothetical scenario and creates an official liability
    in the user's permanent profile. Updates scenario status to 'confirmed_liability'.
    """
    # 1. Fetch scenario and verify ownership
    rec_res = await db.execute(
        select(database.ScenarioRecord).where(
            database.ScenarioRecord.id == request.scenario_id,
            database.ScenarioRecord.user_id == current_user.id,
        )
    )
    rec = rec_res.scalars().first()
    if not rec:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Scenario record not found or access denied.",
        )

    params = rec.parameters or {}
    p_val = request.principal or params.get("loan_principal", 1000000.0)
    r_val = request.interest_rate or params.get("annual_interest_rate", 0.09)
    t_val = request.tenure_months or params.get("loan_tenure_months", 60)

    # 2. Compute exact EMI if missing
    emi_val = request.emi
    if emi_val is None or emi_val <= 0:
        if r_val > 0 and t_val > 0:
            monthly_r = r_val / 12.0
            emi_calc = p_val * (monthly_r * ((1.0 + monthly_r) ** t_val)) / (((1.0 + monthly_r) ** t_val) - 1.0)
            emi_val = round(emi_calc, 2)
        else:
            emi_val = round(p_val / max(1, t_val), 2)

    # 3. Create permanent Liability
    liability = database.Liability(
        user_id=current_user.id,
        name=request.name or rec.name or "Car Loan",
        principal_remaining=float(p_val),
        interest_rate=float(r_val),
        monthly_payment=float(emi_val),
    )
    db.add(liability)

    # 4. Mark scenario as confirmed
    rec.status = "confirmed_liability"
    rec.updated_at = datetime.now(timezone.utc).isoformat()
    await db.commit()
    await db.refresh(liability)

    return schemas.ConfirmLiabilityResponse(
        success=True,
        liability_id=liability.id,
        name=liability.name,
        principal=liability.principal_remaining,
        interest_rate=liability.interest_rate,
        monthly_payment=liability.monthly_payment,
        scenario_status=rec.status,
        message=f"Successfully confirmed and added {liability.name} of ₹{liability.principal_remaining:,.2f} to liabilities.",
    )


@router.get(
    "/history/records",
    response_model=List[schemas.HistoricalScenarioResponse],
    summary="Get Detailed Scenario History with Versions",
)
async def get_scenario_history(
    current_user: database.User = Depends(deps.get_current_user),
    db: AsyncSession = Depends(deps.get_db),
):
    """
    Retrieves full scenario history including versions, status, and parameter snapshots.
    """
    result = await db.execute(
        select(database.ScenarioRecord)
        .where(database.ScenarioRecord.user_id == current_user.id)
        .order_by(database.ScenarioRecord.created_at.desc())
    )
    records = result.scalars().all()
    return [
        schemas.HistoricalScenarioResponse(
            scenario_id=r.id,
            user_id=r.user_id,
            thread_id=r.thread_id,
            name=r.name,
            scenario_type=r.scenario_type,
            parameters=r.parameters or {},
            assumptions=r.assumptions or [],
            baseline_id=r.baseline_id,
            results_summary=r.results_summary,
            version=r.version,
            parent_scenario_id=r.parent_scenario_id,
            status=r.status,
            engine_version=r.engine_version,
            created_at=r.created_at,
            updated_at=r.updated_at,
            is_current=True,
        )
        for r in records
    ]


@router.get(
    "/{scenario_id}",
    response_model=schemas.SimulateScenarioResponse,
    summary="Get Specific Scenario Simulation by ID",
)
async def get_scenario_by_id(
    scenario_id: str,
    current_user: database.User = Depends(deps.get_current_user),
    db: AsyncSession = Depends(deps.get_db),
):
    """
    Retrieves a specific scenario simulation record. Enforces user authorization.
    """
    result = await db.execute(
        select(database.ScenarioRecord).where(
            database.ScenarioRecord.id == scenario_id,
            database.ScenarioRecord.user_id == current_user.id,
        )
    )
    record = result.scalars().first()
    if not record or not record.results_summary:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Scenario record not found or unauthorized.",
        )
    return record.results_summary
