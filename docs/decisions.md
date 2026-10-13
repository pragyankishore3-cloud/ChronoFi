# ChronoFi — Architecture Decision Records (ADRs)

## ADR-001: Separation of LLM Reasoning and Deterministic Financial Mathematics
- **Status**: Accepted
- **Context**: Financial calculations (compound interest, loan amortization, stochastic return paths) are mathematically sensitive. LLMs frequently hallucinate small arithmetic errors that compound drastically over 20-30 year horizons.
- **Decision**: Treat the LLM strictly as an intent parser, parameter extractor, and explanatory agent. All quantitative outputs must be computed by dedicated Python modules (`engine/scenario_engine.py`) and returned to the LLM via tool results.
- **Consequences**: Zero arithmetic hallucination. All calculations are unit-testable and verifiable against standard financial benchmarks.

## ADR-002: Paired Monte Carlo with Common Random Numbers (CRN)
- **Status**: Accepted
- **Context**: When simulating "Take ₹10L loan vs Keep baseline", running independent Monte Carlo simulations introduces simulation variance (e.g. baseline might experience a market crash in run 1 while scenario experiences a bull market in run 2, skewing the comparison).
- **Decision**: Use Common Random Numbers (CRN) by syncing the pseudo-random generator across both runs.
- **Consequences**: Differences between baseline and scenario reflect 100% of the decision's causal effect, not statistical noise.

## ADR-003: Durable LangGraph Persistence with AsyncSqliteSaver
- **Status**: Accepted
- **Context**: In-memory checkpointers (`MemorySaver`) lose conversational and scenario state across server restarts or multi-pod deployments.
- **Decision**: Use `AsyncSqliteSaver` pointing to a local/persistent SQLite file for thread state persistence.
- **Consequences**: Users can resume conversations and retrieve previously simulated scenarios seamlessly across sessions.

## ADR-004: Living Profile with Dynamic Cash-Flow Profiling
- **Status**: Accepted
- **Context**: Baseline simulations previously hardcoded 50% salary or ₹70k living expenses, disconnecting simulations from actual user finances.
- **Decision**: Add `monthly_expenses` to the user profile and compute real-time KPIs (`net_worth`, `monthly_investable_surplus`, `months_of_runway`) with full user CRUD.
- **Consequences**: Every scenario simulation accurately models the user's real disposable surplus.

## ADR-005: Multi-Goal Priority Waterfall Allocation
- **Status**: Accepted
- **Context**: In real life, users have multiple competing goals (`Essential` like Retirement & Emergency Fund, `Important` like Child Education, `Aspirational` like Luxury Travel). When surplus changes, money does not get allocated equally; it follows a priority waterfall.
- **Decision**: Implement a deterministic priority-weighted waterfall allocation in the scenario engine that tracks every goal's funding status, completion probability, and timeline delta.
- **Consequences**: The simulator provides holistic life planning rather than siloed retirement-only projections.
