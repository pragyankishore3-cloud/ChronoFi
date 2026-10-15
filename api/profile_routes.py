from typing import Any
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select
from sqlalchemy.orm import selectinload

from api.deps import get_current_user, get_db
from models import database, schemas

router = APIRouter()

# In-memory preference store to preserve user currency choice, occupation, and DOB without altering existing DB schemas
_user_currency_preferences: dict[int, str] = {}
_user_occupations: dict[int, str] = {}
_user_dobs: dict[int, str] = {}


def _compute_profile_kpis(user: database.User) -> dict[str, float]:
    total_assets = sum(a.current_value for a in user.assets) if user.assets else 0.0
    total_liabilities = sum(l.principal_remaining for l in user.liabilities) if user.liabilities else 0.0
    net_worth = total_assets - total_liabilities

    total_emis = sum(l.monthly_payment for l in user.liabilities) if user.liabilities else 0.0
    expenses = user.monthly_expenses if user.monthly_expenses and user.monthly_expenses > 0 else (user.base_salary * 0.40)
    surplus = max(0.0, user.base_salary - total_emis - expenses)

    # Liquid assets: Cash, Savings, FD
    liquid_assets = sum(
        a.current_value for a in user.assets 
        if any(t in (a.asset_type or "").lower() for t in ["cash", "liquid", "savings", "fd"])
    ) if user.assets else 0.0
    burn_rate = expenses + total_emis
    months_runway = round(liquid_assets / burn_rate, 1) if burn_rate > 0 else 99.0

    return {
        "net_worth": round(net_worth, 2),
        "monthly_investable_surplus": round(surplus, 2),
        "months_of_runway": months_runway,
    }


@router.get("/me", response_model=schemas.UserProfileResponse)
async def get_my_profile(
    current_user: database.User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """Retrieves authenticated user profile with computed financial KPIs."""
    res = await db.execute(
        select(database.User)
        .where(database.User.id == current_user.id)
        .options(
            selectinload(database.User.assets),
            selectinload(database.User.liabilities),
            selectinload(database.User.goals),
        )
    )
    user = res.scalars().first()
    if not user:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="User not found")

    kpis = _compute_profile_kpis(user)
    
    return schemas.UserProfileResponse(
        id=user.id,
        name=user.name,
        email=user.email,
        age=user.age,
        risk_tolerance=user.risk_tolerance,
        base_salary=user.base_salary,
        monthly_expenses=user.monthly_expenses or 0.0,
        currency=_user_currency_preferences.get(user.id, "INR"),
        occupation=_user_occupations.get(user.id, "Professional / Tech"),
        date_of_birth=_user_dobs.get(user.id, "2001-01-15"),
        assets=[schemas.AssetResponse.model_validate(a) for a in user.assets],
        liabilities=[schemas.LiabilityResponse.model_validate(l) for l in user.liabilities],
        goals=[schemas.GoalResponse.model_validate(g) for g in user.goals],
        net_worth=kpis["net_worth"],
        monthly_investable_surplus=kpis["monthly_investable_surplus"],
        months_of_runway=kpis["months_of_runway"],
    )


@router.put("/me", response_model=schemas.UserProfileResponse)
async def update_my_profile(
    update_data: schemas.UserProfileUpdate,
    current_user: database.User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """Updates user demographic, salary, or living expense settings."""
    res = await db.execute(
        select(database.User)
        .where(database.User.id == current_user.id)
        .options(
            selectinload(database.User.assets),
            selectinload(database.User.liabilities),
            selectinload(database.User.goals),
        )
    )
    user = res.scalars().first()
    if not user:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="User not found")

    if update_data.name is not None:
        user.name = update_data.name
    if update_data.age is not None:
        user.age = update_data.age
    if update_data.risk_tolerance is not None:
        user.risk_tolerance = update_data.risk_tolerance
    if update_data.base_salary is not None:
        user.base_salary = update_data.base_salary
    if update_data.monthly_expenses is not None:
        user.monthly_expenses = update_data.monthly_expenses
    if update_data.currency is not None:
        _user_currency_preferences[user.id] = update_data.currency.upper().strip()
    if update_data.occupation is not None:
        _user_occupations[user.id] = update_data.occupation.strip()
    if update_data.date_of_birth is not None:
        _user_dobs[user.id] = update_data.date_of_birth.strip()

    await db.commit()
    await db.refresh(user)

    kpis = _compute_profile_kpis(user)

    return schemas.UserProfileResponse(
        id=user.id,
        name=user.name,
        email=user.email,
        age=user.age,
        risk_tolerance=user.risk_tolerance,
        base_salary=user.base_salary,
        monthly_expenses=user.monthly_expenses or 0.0,
        currency=_user_currency_preferences.get(user.id, "INR"),
        occupation=_user_occupations.get(user.id, "Professional / Tech"),
        date_of_birth=_user_dobs.get(user.id, "2001-01-15"),
        assets=[schemas.AssetResponse.model_validate(a) for a in user.assets],
        liabilities=[schemas.LiabilityResponse.model_validate(l) for l in user.liabilities],
        goals=[schemas.GoalResponse.model_validate(g) for g in user.goals],
        net_worth=kpis["net_worth"],
        monthly_investable_surplus=kpis["monthly_investable_surplus"],
        months_of_runway=kpis["months_of_runway"],
    )



# --- ASSET CRUD ---
@router.post("/assets", response_model=schemas.AssetResponse)
async def create_asset(
    asset_in: schemas.AssetCreate,
    current_user: database.User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """Adds a new financial asset to user profile."""
    new_asset = database.Asset(
        user_id=current_user.id,
        name=asset_in.name,
        asset_type=asset_in.asset_type,
        current_value=asset_in.current_value,
        expected_return_rate=asset_in.expected_return_rate,
    )
    db.add(new_asset)
    await db.commit()
    await db.refresh(new_asset)
    return new_asset


@router.put("/assets/{asset_id}", response_model=schemas.AssetResponse)
async def update_asset(
    asset_id: int,
    asset_in: schemas.AssetUpdate,
    current_user: database.User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """Updates an existing asset belonging to the authenticated user."""
    res = await db.execute(
        select(database.Asset).where(
            database.Asset.id == asset_id,
            database.Asset.user_id == current_user.id
        )
    )
    asset = res.scalars().first()
    if not asset:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Asset not found.")

    if asset_in.name is not None:
        asset.name = asset_in.name
    if asset_in.asset_type is not None:
        asset.asset_type = asset_in.asset_type
    if asset_in.current_value is not None:
        asset.current_value = asset_in.current_value
    if asset_in.expected_return_rate is not None:
        asset.expected_return_rate = asset_in.expected_return_rate

    await db.commit()
    await db.refresh(asset)
    return asset


@router.delete("/assets/{asset_id}")
async def delete_asset(
    asset_id: int,
    current_user: database.User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """Removes an asset belonging to the authenticated user."""
    res = await db.execute(
        select(database.Asset).where(
            database.Asset.id == asset_id,
            database.Asset.user_id == current_user.id
        )
    )
    asset = res.scalars().first()
    if not asset:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Asset not found.")
    await db.delete(asset)
    await db.commit()
    return {"status": "success", "message": f"Asset {asset_id} deleted."}


# --- LIABILITY CRUD ---
@router.post("/liabilities", response_model=schemas.LiabilityResponse)
async def create_liability(
    liability_in: schemas.LiabilityCreate,
    current_user: database.User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """Adds a new liability or loan to user profile."""
    new_liab = database.Liability(
        user_id=current_user.id,
        name=liability_in.name,
        principal_remaining=liability_in.principal_remaining,
        interest_rate=liability_in.interest_rate,
        monthly_payment=liability_in.monthly_payment,
    )
    db.add(new_liab)
    await db.commit()
    await db.refresh(new_liab)
    return new_liab


@router.put("/liabilities/{liability_id}", response_model=schemas.LiabilityResponse)
async def update_liability(
    liability_id: int,
    liability_in: schemas.LiabilityUpdate,
    current_user: database.User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """Updates an existing liability belonging to the authenticated user."""
    res = await db.execute(
        select(database.Liability).where(
            database.Liability.id == liability_id,
            database.Liability.user_id == current_user.id
        )
    )
    liab = res.scalars().first()
    if not liab:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Liability not found.")

    if liability_in.name is not None:
        liab.name = liability_in.name
    if liability_in.principal_remaining is not None:
        liab.principal_remaining = liability_in.principal_remaining
    if liability_in.interest_rate is not None:
        liab.interest_rate = liability_in.interest_rate
    if liability_in.monthly_payment is not None:
        liab.monthly_payment = liability_in.monthly_payment

    await db.commit()
    await db.refresh(liab)
    return liab


@router.delete("/liabilities/{liability_id}")
async def delete_liability(
    liability_id: int,
    current_user: database.User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """Removes a liability belonging to the authenticated user."""
    res = await db.execute(
        select(database.Liability).where(
            database.Liability.id == liability_id,
            database.Liability.user_id == current_user.id
        )
    )
    liab = res.scalars().first()
    if not liab:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Liability not found.")
    await db.delete(liab)
    await db.commit()
    return {"status": "success", "message": f"Liability {liability_id} deleted."}


# --- GOAL CRUD ---
@router.post("/goals", response_model=schemas.GoalResponse)
async def create_goal(
    goal_in: schemas.GoalCreate,
    current_user: database.User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """Adds a new financial life goal to user profile."""
    new_goal = database.Goal(
        user_id=current_user.id,
        name=goal_in.name,
        target_amount=goal_in.target_amount,
        target_years=goal_in.target_years,
        priority=goal_in.priority,
        inflation_rate=goal_in.inflation_rate,
    )
    db.add(new_goal)
    await db.commit()
    await db.refresh(new_goal)
    return new_goal


@router.put("/goals/{goal_id}", response_model=schemas.GoalResponse)
async def update_goal(
    goal_id: int,
    goal_in: schemas.GoalUpdate,
    current_user: database.User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """Updates an existing life goal belonging to the authenticated user."""
    res = await db.execute(
        select(database.Goal).where(
            database.Goal.id == goal_id,
            database.Goal.user_id == current_user.id
        )
    )
    goal = res.scalars().first()
    if not goal:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Goal not found.")

    if goal_in.name is not None:
        goal.name = goal_in.name
    if goal_in.target_amount is not None:
        goal.target_amount = goal_in.target_amount
    if goal_in.target_years is not None:
        goal.target_years = goal_in.target_years
    if goal_in.priority is not None:
        goal.priority = goal_in.priority
    if goal_in.inflation_rate is not None:
        goal.inflation_rate = goal_in.inflation_rate

    await db.commit()
    await db.refresh(goal)
    return goal


@router.delete("/goals/{goal_id}")
async def delete_goal(
    goal_id: int,
    current_user: database.User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """Removes a goal belonging to the authenticated user."""
    res = await db.execute(
        select(database.Goal).where(
            database.Goal.id == goal_id,
            database.Goal.user_id == current_user.id
        )
    )
    goal = res.scalars().first()
    if not goal:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Goal not found.")
    await db.delete(goal)
    await db.commit()
    return {"status": "success", "message": f"Goal {goal_id} deleted."}
