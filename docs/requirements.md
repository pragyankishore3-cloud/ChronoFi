# ChronoFi: Requirements

## Functional Requirements
1. **User Authentication:** System must support secure login via JWT.
2. **Financial Profile Retrieval:** Users must be able to view their Assets, Liabilities, and Goals.
3. **Conversational AI:** Users must be able to chat with an AI via natural language.
4. **Quantitative Forecasting:** The system must run deep learning ML predictions on financial portfolios upon user request.
5. **Real-Time UI Updates:** The frontend must stream tokens natively and plot the resulting statistical metrics on a chart dynamically.

## Non-Functional Requirements
1. **Performance:** The SSE stream must begin returning text tokens in under 2 seconds. The ML forecast must resolve in under 5 seconds.
2. **Security:** Passwords must be bcrypt hashed. Routes must be protected by JWT authorization.
3. **Responsiveness:** The Next.js frontend must adapt seamlessly from 1080p desktop monitors down to mobile viewports.

## Explicitly Excluded Features
- Real-time stock trading.
- Plaid/Bank API syncing (MVP uses manual database entries).
