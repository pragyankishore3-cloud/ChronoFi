from fastapi import FastAPI, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select
from models import database, schemas
from ai.agent import get_financial_agent
from seed.seed_db import seed_database
import contextlib

@contextlib.asynccontextmanager
async def lifespan(app: FastAPI):
    # Run once on startup
    async with database.AsyncSessionLocal() as db:
        # Auto-seed if empty
        result = await db.execute(select(database.User))
        first_user = result.scalars().first()
        if not first_user:
            await seed_database()
    yield

app = FastAPI(title="Chronofi Backend API", description="Enterprise Financial Planning Engine", lifespan=lifespan)

async def get_db():
    async with database.AsyncSessionLocal() as db:
        try:
            yield db
        finally:
            await db.close()

@app.get("/api/profile/{user_id}", response_model=schemas.UserProfileResponse)
async def get_profile(user_id: int, db: AsyncSession = Depends(get_db)):
    result = await db.execute(select(database.User).filter(database.User.id == user_id))
    user = result.scalars().first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    return user

@app.post("/api/chat", response_model=schemas.ChatResponse)
async def chat_with_chronofi(request: schemas.ChatRequest, db: AsyncSession = Depends(get_db)):
    result = await db.execute(select(database.User).filter(database.User.id == request.user_id))
    user = result.scalars().first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
        
    # Build financial context
    context = f"User Profile: {user.name}, Age: {user.age}, Salary: ₹{user.base_salary}. "
    assets_val = sum(a.current_value for a in user.assets)
    context += f"Total Assets: ₹{assets_val}. "
    goals = [f"{g.name}: ₹{g.target_amount} in {g.target_years} years" for g in user.goals]
    context += f"Goals: {', '.join(goals)}. "
    
    full_prompt = f"Context: {context}\nUser Query: {request.message}"
    
    agent = get_financial_agent()
    
    try:
        # We run the synchronous LLM call in a thread to prevent event loop blocking
        import asyncio
        loop = asyncio.get_running_loop()
        agent_result = await loop.run_in_executor(None, lambda: agent.invoke({"messages": [("user", full_prompt)]}))
        response_text = agent_result["messages"][-1].content
    except Exception as e:
        response_text = f"I'm sorry, I encountered an error analyzing that scenario. ({str(e)})"
    
    return schemas.ChatResponse(
        response=response_text, 
        reasoning_logs=[{"action": "Agent evaluated context and invoked deterministic tools."}]
    )

@app.post("/admin/reset")
async def reset_database():
    """Hackathon God-Mode endpoint: Resets DB and seeds personas instantly."""
    await seed_database()
    return {"message": "Database reset and seeded with Arjun and Priya personas."}
