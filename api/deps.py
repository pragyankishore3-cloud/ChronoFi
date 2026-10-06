import os
from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from jose import JWTError, jwt
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select  # Updated import for SQLAlchemy 2.0
from pydantic import BaseModel, ValidationError

from models.database import AsyncSessionLocal, User
from core.security import SECRET_KEY, ALGORITHM

# Note: Ensure this URL matches your actual auth route prefix (e.g., "/api/token" or "/token")
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="api/auth/token")

# Optional but recommended: Add a Pydantic model for token validation
class TokenData(BaseModel):
    user_id: int # or str, depending on your DB ID type

async def get_db():
    # async with automatically closes the session, no need for try/finally
    async with AsyncSessionLocal() as db:
        yield db

async def get_current_user(
    token: str = Depends(oauth2_scheme), 
    db: AsyncSession = Depends(get_db)
):
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        user_id_str: str = payload.get("sub")
        
        if user_id_str is None:
            raise credentials_exception
            
        token_data = TokenData(user_id=int(user_id_str))
    except (JWTError, ValidationError, ValueError):
        raise credentials_exception

    # Query using the user ID instead of the email, and use .where()
    result = await db.execute(select(User).where(User.id == token_data.user_id))
    user = result.scalars().first()

    if user is None:
        raise credentials_exception
        
    return user
