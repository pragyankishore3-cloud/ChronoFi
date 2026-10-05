from fastapi import FastAPI, Depends, HTTPException
from sqlalchemy.orm import Session
from models import database, schemas
from ai.agent import get_financial_agent
from seed.seed_db import seed_database

app = FastAPI(title="Chronofi Backend API", description="Chronofi Financial Planning Engine")

def get_db():
    db = database.SessionLocal()
    try:
        yield db
    finally:
        db.close()

@app.on_event("startup")
def startup_event():
    database.init_db()
    db = database.SessionLocal()
    if db.query(database.User).count() == 0:
        seed_database()
    db.close()

@app.get("/api/profile/{user_id}", response_model=schemas.UserProfileResponse)
def get_profile(user_id: int, db: Session = Depends(get_db)):
    user = db.query(database.User).filter(database.User.id == user_id).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    return user

@app.post("/api/chat", response_model=schemas.ChatResponse)
def chat_with_chronofi(request: schemas.ChatRequest, db: Session = Depends(get_db)):
    user = db.query(database.User).filter(database.User.id == request.user_id).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
        
    # Build financial context for the AI prompt
    context = f"User Profile: {user.name}, Age: {user.age}, Salary: ₹{user.base_salary}. "
    assets_val = sum(a.current_value for a in user.assets)
    context += f"Total Assets: ₹{assets_val}. "
    goals = [f"{g.name}: ₹{g.target_amount} in {g.target_years} years" for g in user.goals]
    context += f"Goals: {', '.join(goals)}. "
    
    full_prompt = f"Context: {context}\nUser Query: {request.message}"
    
    agent = get_financial_agent()
    
    try:
        # LangGraph invoke structure requires a messages list
        result = agent.invoke({"messages": [("user", full_prompt)]})
        # The AI's final response is the last message content
        response_text = result["messages"][-1].content
    except Exception as e:
        response_text = f"I'm sorry, I encountered an error analyzing that scenario. ({str(e)})"
    
    return schemas.ChatResponse(
        response=response_text, 
        reasoning_logs=[{"action": "Agent evaluated context and invoked deterministic tools."}]
    )

@app.post("/admin/reset")
def reset_database():
    """Hackathon God-Mode endpoint: Resets DB and seeds personas instantly."""
    seed_database()
    return {"message": "Database reset and seeded with Arjun and Priya personas."}
