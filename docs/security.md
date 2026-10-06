# ChronoFi: Security Posture

## Authentication
- Handled via standard OAuth2 Password Flow.
- JWT tokens are signed using the `HS256` algorithm.
- `SECRET_KEY` must be injected via `.env` in production.
- Token expires in 30 minutes.

## Authorization
- Endpoints `/api/profile/me` and `/api/chat` strictly require a valid Bearer token.
- User data isolation is enforced at the SQL layer (e.g., `user_id == current_user.id`).

## Data Protection
- Passwords are salted and hashed using `bcrypt`.
- The application does not store plain-text passwords anywhere.
- SQLite is used for local development, meaning DB files should be explicitly added to `.gitignore`.

## LLM Guardrails
- **Prompt Injection:** Mitigated by placing the system message in the immutable `state_modifier` of the LangGraph agent, ensuring user inputs are treated strictly as conversational messages, not system directives.
- **Data Leakage:** The agent is only passed the context of the *currently authenticated user*. It physically cannot access or leak data belonging to other users.
