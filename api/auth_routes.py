from datetime import timedelta

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordRequestForm
from pydantic import BaseModel
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from api.deps import get_db
from core.security import (
    ACCESS_TOKEN_EXPIRE_MINUTES,
    create_access_token,
    verify_password,
)
from models.database import User

router = APIRouter()

# 1. Define a strict response model
class TokenResponse(BaseModel):
    access_token: str
    token_type: str
    user_id: int  # Adjust to str if you use UUIDs

@router.post("/token", response_model=TokenResponse)
async def login_for_access_token(
    form_data: OAuth2PasswordRequestForm = Depends(),
    db: AsyncSession = Depends(get_db)
):
    # 2. Use .where() instead of .filter() for modern SQLAlchemy syntax
    result = await db.execute(select(User).where(User.email == form_data.username))
    user = result.scalars().first()

    # 3. Mitigate timing attacks by separating the checks and using a dummy verification
    if not user:
        # Run a dummy verification against a static hash to equalize response times
        verify_password(form_data.password, "$2b$12$rsuUyzdMcSNI.8oBXDdep.tYPoBI2XmE68nbJm/Xf/zZkGBL0DKje") 
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect email or password",
            headers={"WWW-Authenticate": "Bearer"},
        )

    if not verify_password(form_data.password, user.hashed_password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect email or password",
            headers={"WWW-Authenticate": "Bearer"},
        )

    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
    data={"sub": str(user.id)}, 
    expires_delta=access_token_expires
    )

    return {
        "access_token": access_token, 
        "token_type": "bearer", 
        "user_id": user.id
    }
