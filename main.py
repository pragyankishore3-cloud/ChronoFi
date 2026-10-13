import contextlib
import json

from fastapi import Depends, FastAPI
from fastapi.responses import StreamingResponse
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select

from ai.supervisor import supervisor_app
from api import auth_routes, deps, tax_routes, scenario_routes
from models import database, schemas
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

# Mount Routes
app.include_router(auth_routes.router, prefix="/api/auth", tags=["auth"])
app.include_router(tax_routes.router, prefix="/api/tax", tags=["tax"])
app.include_router(scenario_routes.router, prefix="/api/v1/financial-scenarios", tags=["scenarios"])
app.include_router(scenario_routes.router, prefix="/api/scenarios", tags=["scenarios"])

@app.get("/api/profile/me", response_model=schemas.UserProfileResponse)
async def get_profile(current_user: database.User = Depends(deps.get_current_user)):
    return current_user

@app.get("/api/chat")
async def chat_with_chronofi_get(
    message: str,
    current_user: database.User = Depends(deps.get_current_user),
    db: AsyncSession = Depends(deps.get_db)
):
    request = schemas.ChatRequest(user_id=current_user.id, message=message)
    return await chat_with_chronofi(request, current_user, db)

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
    
    agent = supervisor_app
    config = {'configurable': {'user_id': current_user.id}}

    async def event_stream():
        streamed_tokens = False
        try:
            # astream_events is the modern LangGraph streaming method
            async for event in agent.astream_events(
                {"messages": [("user", full_prompt)], "user_query": request.message, "user_id": current_user.id}, config=config, 
                version="v1"
            ):
                kind = event["event"]
                
                # Stream the final AI response token-by-token
                if kind == "on_chat_model_stream":
                    content = event["data"]["chunk"].content
                    if content:
                        streamed_tokens = True
                        # Server-Sent Event (SSE) format
                        yield f"data: {json.dumps({'type': 'token', 'content': content})}\n\n"
                        
                # Stream tool execution logs to the frontend
                elif kind == "on_tool_start":
                    tool_name = event.get("name")
                    yield f"data: {json.dumps({'type': 'log', 'content': f'Invoking mathematical tool: {tool_name}'})}\n\n"

                # Stream structured scenario result when financial engine finishes
                elif kind == "on_tool_end":
                    tool_name = event.get("name")
                    tool_output = event.get("data", {}).get("output")
                    if tool_name == "run_financial_scenario" and isinstance(tool_output, dict):
                        yield f"data: {json.dumps({'type': 'scenario_result', 'data': tool_output})}\n\n"

                # Stream final output if generated deterministically by presentation node
                elif kind == "on_chain_end" and not streamed_tokens:
                    chain_name = event.get("name")
                    if chain_name in ["Scenario_Agent", "present_scenario"]:
                        out = event.get("data", {}).get("output", {})
                        if isinstance(out, dict) and "messages" in out and out["messages"]:
                            content = out["messages"][-1].content
                            if content:
                                yield f"data: {json.dumps({'type': 'token', 'content': content})}\n\n"
                                streamed_tokens = True
                    
        except Exception as e:
            yield f"data: {json.dumps({'type': 'error', 'content': str(e)})}\n\n"
            
        yield "data: [DONE]\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")

@app.post("/admin/reset")
async def reset_database():
    """Hackathon God-Mode endpoint: Resets DB and seeds personas instantly."""
    await seed_database()
    return {"message": "Database reset and seeded with Arjun and Priya personas."}

