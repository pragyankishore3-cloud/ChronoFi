# ChronoFi — System Architecture

## 1. High-Level Architecture Overview

ChronoFi is an AI-powered life financial planning platform that pairs natural language conversational intelligence with strict, verifiable deterministic mathematical financial modeling.

```mermaid
flowchart TD
    subgraph Client["Frontend Layer (Next.js 14 / Tailwind CSS)"]
        UI_Dash[Dashboard & KPI Gauges]
        UI_Sim[What-If Scenario Simulator Modal]
        UI_Chat[LangGraph Agent Chat & Reasoning]
        UI_Tax[Section 115BAC Tax Optimizer]
    end

    subgraph API["FastAPI Application Services (Python 3.14 / Uvicorn)"]
        API_Auth["/api/auth (JWT Auth & Passlib)"]
        API_Profile["/api/profile (Cash-Flow KPIs & CRUD)"]
        API_Scenario["/api/scenarios (What-If Execution & Storage)"]
        API_Chat["/api/chat (LangGraph Supervisor & Tools)"]
        API_Tax["/api/tax (New vs Old Regime Calculation)"]
    end

    subgraph CoreEngine["Quantitative Financial Engine"]
        Engine_Loan[Loan Amortization & EMI Engine]
        Engine_MC[Paired Monte Carlo Simulator (CRN)]
        Engine_MultiGoal[Multi-Goal Waterfall & Conflict Resolver]
        Engine_Tax[Indian Income & Capital Gains Tax Rules]
    end

    subgraph AgentLayer["AI & State Management (LangGraph)"]
        Graph_Sup[Financial Supervisor]
        Graph_Tools[Scenario Tools & Checkpointer Bridge]
        Graph_Checkpointer[(AsyncSqliteSaver Checkpointer)]
    end

    subgraph Data["Persistence Layer"]
        DB_SQLite[(SQLite / chronofi.db)]
        DB_Checkpoints[(checkpoints.db)]
    end

    Client --> API
    API --> CoreEngine
    API --> AgentLayer
    AgentLayer --> Graph_Checkpointer
    CoreEngine --> DB_SQLite
    API --> DB_SQLite
```

## 2. Core Principles & Safety Invariants

1. **Zero Arithmetic Hallucination**:
   - The LLM never computes EMIs, future corpuses, probabilities, or taxes directly.
   - All financial numbers originate from verified, deterministic Python functions in `engine/scenario_engine.py` and `ai/tools.py`.
2. **Paired Monte Carlo (Common Random Numbers - CRN)**:
   - When evaluating a scenario against baseline, both paths are generated using identical pseudo-random market return shocks (`np.random.seed(seed)`).
   - This isolates the true financial impact of the decision from Monte Carlo sample variance.
3. **Strict Separation of Hypothetical Scenarios and Real Records**:
   - Running simulations creates read-only `ScenarioRecord` entries without modifying actual `User`, `Asset`, `Liability`, or `Goal` tables.
   - Real commitments require explicit user confirmation and dedicated CRUD endpoints.
4. **Data Isolation**:
   - Every database query strictly filters by `user_id == current_user.id` resolved server-side from signed JWTs.
