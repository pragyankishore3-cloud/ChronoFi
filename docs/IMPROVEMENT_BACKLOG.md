# ChronoFi — Improvement Backlog

Tracking document for continuous research, development, and iterative delivery of the ChronoFi AI Life Financial Planner.

**Priority Order:**
1. Critical correctness, security, and data-integrity issues
2. Broken existing functionality
3. Missing tests and reliability gaps
4. High-value improvements to existing features
5. New differentiating features
6. Cosmetic improvements and low-priority refactoring

---

## Backlog Items

| Task ID | Title | Priority | Status | Impact Area |
| :--- | :--- | :---: | :---: | :--- |
| **CF-001** | Living Cash-Flow Profiling & Dynamic Living Expenses | P0 | **Completed** | Profile Engine, Math Engine, APIs, UI |
| **CF-002** | Multi-Goal Waterfall Allocation & Pareto Conflict Resolution | P0 | **Completed** | Scenario Engine, Multi-Goal Math, APIs, UI |
| **CF-003** | Emergency Runway & Liquidity Shock Stress Tester | P1 | **Pending** | Profile Engine, Health Monitor |
| **CF-004** | India LTCG/STCG Tax-Aware Goal Decumulation & Drawdown | P1 | **Pending** | Tax Engine, Scenario Engine |
| **CF-005** | Retire US S&P 500 LSTM $\rightarrow$ Indian Multi-Asset Portfolio Matrix | P2 | **Pending** | Quant Engine, Risk Modeling |
| **CF-006** | Automated Debt Avalanche vs Snowball Optimizer | P2 | **Pending** | Liability Planning, AI Tools |
| **CF-007** | LangGraph Session Checkpoint Snapshot Pruner & Exporter | P2 | **Pending** | Memory & Checkpointer |

---

## Detailed Task Specifications

### [CF-001] Living Cash-Flow Profiling & Dynamic Living Expenses
- **Status**: `Completed`
- **Problem**: Baseline and scenario calculations used hardcoded `salary * 0.50` or `₹70,000` expenses heuristic, disconnecting simulations from the user's real budget. Furthermore, users could not edit expenses or perform CRUD on assets/liabilities/goals.
- **Evidence**: `scenario_engine.py` line 254 (`baseline_surplus = max(1000.0, (base_salary - existing_liabilities_monthly) * 0.50)`).
- **Delivered**:
  - Added `monthly_expenses` to `User` DB model and Pydantic schemas.
  - Implemented `/api/profile/me` with KPIs (`net_worth`, `monthly_investable_surplus`, `months_of_runway`) and full CRUD for assets, liabilities, and goals.
  - Integrated dynamic expenses into `scenario_tools.py`, `scenario_workflow.py`, and `scenario_engine.py`.
  - Added interactive expense editor and item CRUD modals in `FinancialProfile.tsx`.
- **Tests**: 4 comprehensive tests in `tests/test_profile_crud.py` (47/47 suite passing).

---

### [CF-002] Multi-Goal Waterfall Allocation & Pareto Conflict Resolution
- **Status**: `Completed`
- **Problem**: When a user runs a scenario (e.g. Taking a ₹10 Lakh car loan), ChronoFi currently only projects a single goal (Retirement or the longest-horizon goal), ignoring intermediate goals (Home down payment, Child education, Emergency fund). The user cannot see which intermediate goals are starved of funds, delayed, or put at risk.
- **Evidence**: `ai/scenario_tools.py` line 128: selects a single `ret_goal`, discarding all other user goals from the simulation.
- **Delivered**:
  - Implemented deterministic `calculate_multi_goal_waterfall` in `engine/scenario_engine.py` allocating monthly surplus across `Essential` $\rightarrow$ `Important` $\rightarrow$ `Aspirational` tiers.
  - Modeled post-loan recovery compounding for goals extending past loan tenure.
  - Added `GoalConflictItem` and `GoalConflictItemSchema` with status badges, funding ratio deltas, corpus shortfall, and timeline delays.
  - Integrated `user_goals` data into `/api/v1/financial-scenarios/simulate` and LangGraph tool execution.
  - Added Multi-Goal Life Impact Analysis narrative generation in `ai/scenario_workflow.py`.
  - Implemented rich Multi-Goal Conflict & Waterfall visualization in `ScenarioComparison.tsx`.
- **Tests**: 5 comprehensive tests in `tests/test_multi_goal_waterfall.py` (52/52 non-skipped backend tests passing).

---

### [CF-003] Multi-Goal Trade-Off Visualizer & Alternative Selector
- **Status**: `Pending`
- **Problem**: The frontend What-If Scenario Modal only shows Retirement corpus and age diffs. Users cannot visually compare all their life goals or toggle alternative adjustments directly in the UI.
- **Dependencies**: `CF-002`.
- **Acceptance Criteria**: Dashboard UI displays goal impact cards showing baseline vs modified target years, completion probability, and trade-off badges.

---

### [CF-004] Emergency Runway & Liquidity Shock Stress Tester
- **Status**: `Pending`
- **Problem**: Users cannot simulate catastrophic life events (sudden job loss for 6 months, major medical emergency of ₹5L, income reduction of 30%).
- **Dependencies**: `CF-001`.
- **Acceptance Criteria**: Calculates runway depletion under 0 income, identifies which liquid assets are liquidated first, and calculates recovery timeline.

---

### [CF-005] India LTCG/STCG Tax-Aware Goal Decumulation & Drawdown
- **Status**: `Pending`
- **Problem**: ChronoFi's tax engine calculates income tax (Section 115BAC), but scenario projections assume tax-free investment growth and zero withdrawal friction. Under Indian tax laws, equity withdrawals incur 12.5% LTCG above ₹1.25 Lakh, and debt funds are taxed at slab rates.
- **Dependencies**: `CF-002`.
- **Acceptance Criteria**: Integrates post-tax net corpus calculations for mutual funds, EPF/PPF, and fixed deposits during goal maturity.
