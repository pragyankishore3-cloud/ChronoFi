from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select
from models.database import User, Goal, Asset, Liability

class UserRepository:
    def __init__(self, session: AsyncSession):
        self.session = session

    async def get_by_id(self, user_id: int) -> User:
        result = await self.session.execute(select(User).where(User.id == user_id))
        return result.scalars().first()

    async def create(self, user_data: dict) -> User:
        user = User(**user_data)
        self.session.add(user)
        await self.session.commit()
        await self.session.refresh(user)
        return user

class GoalRepository:
    def __init__(self, session: AsyncSession):
        self.session = session

    async def get_by_user(self, user_id: int):
        result = await self.session.execute(select(Goal).where(Goal.user_id == user_id))
        return result.scalars().all()
