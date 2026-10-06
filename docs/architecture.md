# ChronoFi: Architecture

## System Overview
ChronoFi is a monolithic backend application built with Python FastAPI, tightly coupled with a Next.js (App Router) frontend proxy. 

## Major Components
1. **Frontend (Next.js 14, Tailwind, Framer Motion)**
   - Proxies `/api/*` to the FastAPI backend to bypass CORS.
   - Renders a responsive, server-state-driven Dashboard.
2. **Backend API (FastAPI)**
   - Exposes RESTful endpoints for Auth (`/api/auth/token`) and Profile (`/api/profile/me`).
   - Exposes Server-Sent Events (SSE) endpoint for AI Chat (`/api/chat`).
3. **Database (Async SQLite -> PostgreSQL ready)**
   - Uses `sqlalchemy.ext.asyncio` for non-blocking DB operations.
   - Schema: `User`, `Asset`, `Liability`, `Goal`.
4. **AI Reasoning Engine (LangGraph + Gemini 2.5 Pro)**
   - Responsible strictly for NLU, intent routing, and contextual data formatting.
5. **Quantitative Math Engine (PyTorch LSTM)**
   - A deep learning model (`chronofi_lstm.pth`) trained on 30 years of S&P 500 historical data.
   - Uses Monte Carlo Dropout during inference to generate 500 potential market trajectories.

## Data Flow (Chat Interaction)
1. User types query in UI -> `fetchEventSource` (SSE) with JWT -> FastAPI `/api/chat`.
2. FastAPI extracts user's Assets/Liabilities/Goals from SQLite.
3. LangGraph Agent reads context and triggers `simulate_goal_probability_tool`.
4. Tool runs PyTorch LSTM model locally.
5. Tool returns math JSON -> LangGraph formats text -> Yields text tokens to UI.
6. UI intercepts JSON data payload and renders Recharts graph.
