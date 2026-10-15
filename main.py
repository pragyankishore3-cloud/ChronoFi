import contextlib
from datetime import datetime, timezone
import json
from typing import List, Optional
import uuid

from pathlib import Path
from fastapi import Depends, FastAPI, HTTPException, status
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select

from api import (
    auth_routes,
    debt_routes,
    deps,
    document_routes,
    health_routes,
    profile_routes,
    rates_routes,
    scenario_routes,
    stress_routes,
    tax_routes,
)
from core import checkpointer
from models import database, schemas
from seed.seed_db import seed_database


@contextlib.asynccontextmanager
async def lifespan(app: FastAPI):
    # Ensure tables exist on startup
    async with database.engine.begin() as conn:
        await conn.run_sync(database.Base.metadata.create_all)

    # Initialize durable checkpointer
    await checkpointer.init_checkpointer()

    # Seed if database is empty
    async with database.AsyncSessionLocal() as db:
        result = await db.execute(select(database.User))
        first_user = result.scalars().first()
        if not first_user:
            await seed_database()

    yield

    # Clean up checkpointer on shutdown
    await checkpointer.close_checkpointer()


app = FastAPI(title="Chronofi Backend API", description="Enterprise Financial Planning Engine", lifespan=lifespan)

STATIC_DIR = Path(__file__).resolve().parent / "static"
if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


@app.get("/", response_class=HTMLResponse)
async def serve_index():
    """Serves the ChronoFi Copilot frontend landing page."""
    index_file = STATIC_DIR / "index.html"
    if index_file.exists():
        return HTMLResponse(content=index_file.read_text(encoding="utf-8"), status_code=200)
    return HTMLResponse(content="<html><body><h1>ChronoFi Copilot</h1></body></html>", status_code=200)


# Mount Routes
app.include_router(auth_routes.router, prefix="/api/auth", tags=["auth"])
app.include_router(profile_routes.router, prefix="/api/profile", tags=["profile"])
app.include_router(tax_routes.router, prefix="/api/tax", tags=["tax"])
app.include_router(stress_routes.router, prefix="/api/stress", tags=["stress"])
app.include_router(debt_routes.router, prefix="/api/debt", tags=["debt"])
app.include_router(health_routes.router, prefix="/api/health", tags=["health"])
app.include_router(document_routes.router, prefix="/api/documents", tags=["documents"])
app.include_router(rates_routes.router, prefix="/api/rates", tags=["rates"])
app.include_router(scenario_routes.router, prefix="/api/v1/financial-scenarios", tags=["scenarios"])


@app.get("/api/v1/threads", response_model=List[schemas.ConversationThreadResponse])
async def list_user_threads(
    current_user: database.User = Depends(deps.get_current_user),
    db: AsyncSession = Depends(deps.get_db),
):
    """Lists all active conversation threads belonging to the authenticated user."""
    result = await db.execute(
        select(database.ConversationThread)
        .where(database.ConversationThread.user_id == current_user.id)
        .order_by(database.ConversationThread.updated_at.desc())
    )
    return result.scalars().all()


@app.get("/api/v1/threads/{thread_id}", response_model=schemas.ConversationThreadResponse)
async def get_thread_info(
    thread_id: str,
    current_user: database.User = Depends(deps.get_current_user),
    db: AsyncSession = Depends(deps.get_db),
):
    """Retrieves metadata for a specific thread. Enforces user isolation."""
    result = await db.execute(
        select(database.ConversationThread).where(database.ConversationThread.id == thread_id)
    )
    thread = result.scalars().first()
    if not thread:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Thread not found.")
    if thread.user_id != current_user.id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Access denied: thread belongs to another user.",
        )
    return thread


@app.get("/api/chat")
async def chat_with_chronofi_get(
    message: str,
    thread_id: Optional[str] = None,
    current_user: database.User = Depends(deps.get_current_user),
    db: AsyncSession = Depends(deps.get_db),
):
    request = schemas.ChatRequest(user_id=current_user.id, message=message, thread_id=thread_id)
    return await chat_with_chronofi(request, current_user, db)


@app.post("/api/chat")
async def chat_with_chronofi(
    request: schemas.ChatRequest, 
    current_user: database.User = Depends(deps.get_current_user),
    db: AsyncSession = Depends(deps.get_db)
):
    # 1. Resolve and authenticate thread ownership
    now_str = datetime.now(timezone.utc).isoformat()
    thread_id = request.thread_id

    if thread_id:
        t_res = await db.execute(
            select(database.ConversationThread).where(database.ConversationThread.id == thread_id)
        )
        thread_obj = t_res.scalars().first()
        if thread_obj:
            if thread_obj.user_id != current_user.id:
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail="Access denied: requested conversation thread belongs to another user.",
                )
            thread_obj.updated_at = now_str
            await db.commit()
        else:
            # Register newly created thread ID for this user
            new_thread = database.ConversationThread(
                id=thread_id,
                user_id=current_user.id,
                title=request.message[:40],
                created_at=now_str,
                updated_at=now_str,
            )
            db.add(new_thread)
            await db.commit()
    else:
        thread_id = str(uuid.uuid4())
        new_thread = database.ConversationThread(
            id=thread_id,
            user_id=current_user.id,
            title=request.message[:40],
            created_at=now_str,
            updated_at=now_str,
        )
        db.add(new_thread)
        await db.commit()

    # 2. Build financial context
    context = f"User Profile: {current_user.name}, Age: {current_user.age}, Salary: ₹{current_user.base_salary}. "
    
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
    config = {'configurable': {'user_id': current_user.id, 'thread_id': thread_id}}

    async def event_stream():
        streamed_tokens = False
        try:
            # Immediately notify frontend of active thread_id for conversation persistence
            yield f"data: {json.dumps({'type': 'thread_id', 'thread_id': thread_id})}\n\n"

            # astream_events is the modern LangGraph streaming method
            async for event in agent.astream_events(
                {
                    "messages": [("user", full_prompt)],
                    "user_query": request.message,
                    "user_id": current_user.id,
                    "thread_id": thread_id,
                },
                config=config, 
                version="v1"
            ):
                kind = event["event"]
                
                # Stream the final AI response token-by-token
                if kind == "on_chat_model_stream":
                    content = event["data"]["chunk"].content
                    if content:
                        streamed_tokens = True
                        yield f"data: {json.dumps({'type': 'token', 'content': content})}\n\n"
                        
                # Stream tool execution logs to the frontend
                elif kind == "on_tool_start":
                    tool_name = event.get("name")
                    yield f"data: {json.dumps({'type': 'log', 'content': f'Invoking tool: {tool_name}'})}\n\n"

                # Stream structured scenario result when financial engine finishes
                elif kind == "on_tool_end":
                    tool_name = event.get("name")
                    tool_output = event.get("data", {}).get("output")
                    if tool_name == "run_financial_scenario" and isinstance(tool_output, dict):
                        yield f"data: {json.dumps({'type': 'scenario_result', 'data': tool_output})}\n\n"
                    elif tool_name == "confirm_and_record_liability" and isinstance(tool_output, dict):
                        yield f"data: {json.dumps({'type': 'liability_confirmed', 'data': tool_output})}\n\n"

                # Stream final output if generated deterministically by presentation node
                elif kind == "on_chain_end" and not streamed_tokens:
                    chain_name = event.get("name")
                    if chain_name in ["Scenario_Agent", "present_scenario", "clarification_node", "execute_confirmation"]:
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
