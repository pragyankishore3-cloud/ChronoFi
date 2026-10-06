# ChronoFi: Assumptions Log

## Assumption 1
- **Assumption:** Users will understand the difference between CVaR (Conditional Value at Risk) and standard expected returns.
- **Why it was made:** We are building an institutional-grade platform, assuming a slightly higher baseline financial literacy than generic apps.
- **Impact if wrong:** Users might get scared by the CVaR numbers (which represent severe market crashes) and abandon their financial plans.
- **Status:** Active. Mitigated by strictly instructing the Gemini LLM to always explain CVaR in plain English.

## Assumption 2
- **Assumption:** An LSTM trained on S&P 500 historical data is sufficient for modeling all equity portfolios.
- **Why it was made:** To restrict MVP scope while still proving the deep learning pipeline works.
- **Impact if wrong:** Users with highly diverse portfolios (e.g., heavy crypto or real estate) will receive mathematically inaccurate risk profiles because the S&P 500 drift/volatility does not match their assets.
- **Status:** Active. Must be addressed in V2 by training specialized ML models for specific asset classes.
