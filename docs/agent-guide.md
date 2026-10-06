# ChronoFi: Agent Development Guide

## Project Purpose
To provide institutional-grade financial modeling to retail investors through a conversational AI interface backed by rigorous PyTorch deep learning models.

## Repository Structure
- `/api`: FastAPI route handlers (auth, profile, dependencies).
- `/core`: Security functions (JWT, bcrypt).
- `/models`: SQLAlchemy models and Pydantic schemas.
- `/ml`: PyTorch model definition (`model.py`), training script (`train.py`), weights, and scalers.
- `/engine`: The execution layer bridging the ML model (`ml_forecaster.py`) and the Agent.
- `/ai`: LangGraph definition (`agent.py`) housing the `simulate_goal_probability_tool`.
- `/seed`: Database initialization.
- `/docs`: This documentation directory.
- `../chronofi-frontend`: The Next.js UI (kept as a parallel directory, but conceptually part of the system).

## How the Chatbot Works
The chatbot is powered by **Gemini 2.5 Pro**, orchestrated via **LangGraph**.
**CRITICAL RULE:** The LLM is strictly forbidden from performing mathematical calculations. It must *always* invoke the `simulate_goal_probability_tool`, which executes the deterministic PyTorch LSTM script.

## Coding Conventions
- **Async First:** All FastAPI routes and SQLAlchemy queries must use `async`/`await`.
- **Type Hints:** Enforced across all Python functions.
- **Pydantic Validation:** All external data must flow through Pydantic schemas.
- **UI Aesthetics:** The frontend must use a strict `zinc-950` monochrome palette. Absolutely no glassmorphism or neon "AI" gradients.

## Common Mistakes to Avoid
1. **Schema Mismatch:** Ensure frontend TypeScript interfaces map exactly to Pydantic JSON keys (e.g., `current_value` vs `value`).
2. **Synchronous DB Calls:** Never use `SessionLocal` in an async route. Use `AsyncSessionLocal`.
3. **Passlib Dependency:** Do not use `passlib` for bcrypt. It is deprecated and breaks on modern Python. Use raw `bcrypt.hashpw`.
