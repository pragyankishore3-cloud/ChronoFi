# ChronoFi — Implementation Backlog

This backlog catalogs every verified, feasible improvement required for ChronoFi, prioritized by impact, risk, and architectural dependencies.

---

## Task Index

| Task ID | Tier | Title | Status | Primary Component |
| :--- | :---: | :--- | :---: | :--- |
| **CF-001** | **P0** | Living Cash-Flow Profiling & Dynamic Expenses | **Completed** | Database, API, Engine, UI |
| **CF-002** | **P0** | Multi-Goal Priority Waterfall & Conflict Resolution | **Completed** | Engine, API, AI Workflow, UI |
| **CF-003** | **P1** | Emergency Runway & Liquidity Shock Stress Tester | **Completed** | `engine/stress_engine.py`, API, UI |
| **CF-004** | **P1** | India Capital Gains (LTCG/STCG) Goal Drawdown | **Completed** | `engine/tax_drawdown.py`, Engine, API |
| **CF-005** | **P1** | Debt Avalanche vs Snowball Acceleration Engine | **Completed** | `engine/debt_engine.py`, API, UI |
| **CF-006** | **P2** | Actual vs Projected Variance & Financial Health Monitor | **Completed** | `engine/health_engine.py`, API, UI |
| **CF-007** | **P2** | Local In-Memory Fallback & Document Intelligence RAG | **Completed** | `engine/vector_store.py`, `api/document_routes.py` |
| **CF-008** | **P2** | LangGraph Multi-Tool Agent Routing & Checkpoint Persistence | **Completed** | `ai/supervisor.py`, `core/checkpointer.py` |
| **CF-009** | **P3** | Interactive Stress Testing & Debt Payoff UI Integration | **Completed** | Next.js Dashboard Components (`StressTester`, `DebtPayoffOptimizer`, `FinancialHealthMonitor`) |

---

## Implemented & Verified Specifications

### [CF-001] Living Cash-Flow Profiling & Dynamic Expenses (P0) — COMPLETED
- Added `monthly_expenses` to Database schema, `User` model, and API schemas.
- Profile CRUD endpoints `/api/profile` with live calculation of Net Worth, Monthly Investable Surplus, and Months of Emergency Runway.
- Full UI integration in `FinancialProfile.tsx`. 4/4 tests passing in `tests/test_profile_crud.py`.

### [CF-002] Multi-Goal Priority Waterfall & Conflict Resolution (P0) — COMPLETED
- Multi-tier priority allocation (`Essential` $\rightarrow$ `Important` $\rightarrow$ `Aspirational`).
- Deficit starvation tracking and post-loan payoff recovery compounding.
- Unified into `/api/v1/financial-scenarios/simulate` with `goal_conflicts` response.
- Integrated into `ScenarioComparison.tsx`. 5/5 tests passing in `tests/test_multi_goal_waterfall.py`.

### [CF-003] Emergency Runway & Liquidity Shock Stress Tester (P1) — COMPLETED
- Implemented `engine/stress_engine.py` with 4-tier liquidation waterfall: Cash (0% penalty) $\rightarrow$ FD (1% premature penalty) $\rightarrow$ Debt (2% exit friction) $\rightarrow$ Equity (15% fire-sale haircut).
- Mounted `/api/stress/simulate` endpoint.
- Interactive Next.js component `StressTester.tsx` integrated into dashboard. 5/5 tests passing in `tests/test_stress_engine.py`.

### [CF-004] India Capital Gains (LTCG/STCG) Goal Drawdown (P1) — COMPLETED
- Implemented `engine/tax_drawdown.py` following Finance Act / Budget 2024 rules:
  - Equity LTCG: 12.5% on gains exceeding ₹1.25 Lakh annual exemption limit.
  - Equity STCG: 20%.
  - Debt / Section 50AA: Taxed at investor's marginal slab rate.
  - EEE Tax-Exempt instruments (PPF, EPF within statutory cap): 0% tax.
  - 4% Health & Education cess.
  - Gross-up withdrawal formula and tax-loss harvesting recommendation.
- Mounted `/api/tax/goal-drawdown` and `/api/tax/portfolio-drawdown`. 6/6 tests passing in `tests/test_tax_drawdown.py`.

### [CF-005] Debt Avalanche vs Snowball Acceleration Engine (P1) — COMPLETED
- Implemented `engine/debt_engine.py` comparing Baseline vs Avalanche (highest rate first) vs Snowball (smallest balance first).
- Supports extra monthly prepayments and lump-sum bonus injections.
- Mounted `/api/debt/optimize`.
- Interactive Next.js component `DebtPayoffOptimizer.tsx` integrated into dashboard. 5/5 tests passing in `tests/test_debt_engine.py`.

### [CF-006] Actual vs Projected Variance & Financial Health Monitor (P2) — COMPLETED
- Implemented `engine/health_engine.py` computing 0-100 composite Financial Health Score:
  - Emergency Runway (0-25 pts)
  - Debt-to-Income DTI (0-25 pts)
  - Savings & Surplus Rate (0-25 pts)
  - Goal Funding Progress (0-25 pts)
- Milestone tracking variance (Ahead / On Track / Behind).
- Mounted `/api/health/score` and `/api/health/evaluate`.
- Interactive Next.js component `FinancialHealthMonitor.tsx` integrated into dashboard. 8/8 tests passing in `tests/test_health_engine.py`.

### [CF-007] Local In-Memory Fallback & Document Intelligence RAG (P2) — COMPLETED
- Offline-safe try/except fallback for embedding models in `engine/vector_store.py`.
- Full document management and processing routes in `api/document_routes.py` (`POST /api/documents/upload`, `GET /api/documents/`, `DELETE /api/documents/{id}`).
- All document intelligence and vector store tests unskipped and verified (100% passing across 8 test suites).

### [CF-008] LangGraph Multi-Tool Agent Routing & Checkpoint Persistence (P2) — COMPLETED
- Fixed supervisor node routing and unskipped `tests/test_langgraph_rag.py` (3/3 passing).
- Durable checkpointer persistence verified across reboots and isolated threads.

### [CF-009] Interactive Stress Testing & Debt Payoff UI Integration (P3) — COMPLETED
- Added multi-tab navigation to `src/app/dashboard/page.tsx` with What-If Simulator, Liquidity Stress Tester, Debt Payoff Accelerator, Financial Health Score, and Quant Agent Chat.
- Verified TypeScript type check (`npx tsc --noEmit` $\rightarrow$ 0 errors) and Next.js production build (`npm run build` $\rightarrow$ 9/9 static routes generated cleanly).
