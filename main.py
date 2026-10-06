import json
import asyncio
import contextlib
from fastapi import FastAPI, Depends, HTTPException
from fastapi.responses import StreamingResponse
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select

from models import database, schemas
from api import auth_routes, deps
from ai.agent import get_financial_agent
from seed.seed_db import seed_database

@contextlib.asynccontextmanager
async def lifespan(app: FastAPI):
    # Run once on startup
    async with database.AsyncSessionLocal() as db:
        result = await db.execute(select(database.User))
        first_user = result.scalars().first()
        if not first_user:
            await seed_database()
    yield

app = FastAPI(title="Chronofi Backend API", description="Enterprise Financial Planning Engine", lifespan=lifespan)

# Mount Auth Routes
app.include_router(auth_routes.router, prefix="/api/auth", tags=["auth"])

@app.get("/api/profile/me", response_model=schemas.UserProfileResponse)
async def get_profile(current_user: database.User = Depends(deps.get_current_user)):
    return current_user

@app.post("/api/chat")
async def chat_with_chronofi(
    request: schemas.ChatRequest, 
    current_user: database.User = Depends(deps.get_current_user),
    db: AsyncSession = Depends(deps.get_db)
):
    # Ensure relationships are loaded properly if not eager loaded, but we used lazy='selectin'
    # Build financial context
    context = f"User Profile: {current_user.name}, Age: {current_user.age}, Salary: ₹{current_user.base_salary}. "
    
    # Need to manually fetch the user with relationships loaded just in case it wasn't
    result = await db.execute(
        select(database.User).filter(database.User.id == current_user.id)
    )
    user_with_rels = result.scalars().first()
    
    assets_val = sum(a.current_value for a in user_with_rels.assets)
    context += f"Total Assets: ₹{assets_val}. "
    goals = [f"{g.name}: ₹{g.target_amount} in {g.target_years} years" for g in user_with_rels.goals]
    context += f"Goals: {', '.join(goals)}. "
    
    full_prompt = f"Context: {context}\nUser Query: {request.message}"
    
    agent = get_financial_agent()

    async def event_stream():
        try:
            # astream_events is the modern LangGraph streaming method
            async for event in agent.astream_events(
                {"messages": [("user", full_prompt)]}, 
                version="v1"
            ):
                kind = event["event"]
                
                # Stream the final AI response token-by-token
                if kind == "on_chat_model_stream":
                    content = event["data"]["chunk"].content
                    if content:
                        # Server-Sent Event (SSE) format
                        yield f"data: {json.dumps({'type': 'token', 'content': content})}\n\n"
                        
                # Optionally stream tool execution logs to the frontend
                elif kind == "on_tool_start":
                    tool_name = event["name"]
                    yield f"data: {json.dumps({'type': 'log', 'content': f'Invoking mathematical tool: {tool_name}'})}\n\n"
                    
        except Exception as e:
            yield f"data: {json.dumps({'type': 'error', 'content': str(e)})}\n\n"
            
        yield "data: [DONE]\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")

@app.post("/admin/reset")
async def reset_database():
    """Hackathon God-Mode endpoint: Resets DB and seeds personas instantly."""
    await seed_database()
    return {"message": "Database reset and seeded with Arjun and Priya personas."}
