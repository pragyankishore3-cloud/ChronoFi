# ChronoFi: Progress & Status

## Completed Work
- [x] Phase 1: Core Database & FastAPI skeleton (`User`, `Asset`, `Liability`, `Goal`).
- [x] Phase 2: PyTorch LSTM Model definition and training pipeline (`ml/train.py`).
- [x] Phase 3: LangGraph Agent architecture with strict math tool invocation.
- [x] Phase 4: Authentication & Security (JWT, bcrypt).
- [x] Phase 5: Next.js Frontend Dashboard (Responsive, Dark Mode, Recharts, SSE Streaming).
- [x] DevOps: Dockerfile, Docker Compose, Pytest suite, GitHub Actions CI/CD.

## Current Status
The application is **feature-complete for MVP**. The backend correctly processes probabilistic ML queries, and the frontend accurately renders the trajectory graphs and chat interface.

## Known Bugs
- None currently active. The `EPERM` Next.js file locking bug and the Pydantic schema mismatches were resolved.

## Remaining Work / Next Recommended Tasks
1. Implement a user registration page (currently relies on demo accounts from `seed_db.py`).
2. Add CRUD endpoints for users to manually add/edit their Assets, Liabilities, and Goals in the UI.
3. Migrate from SQLite to PostgreSQL for production deployment.
