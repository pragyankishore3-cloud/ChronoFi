# ChronoFi: Decision Log

## Decision 1: PyTorch LSTM over standard Monte Carlo (Geometric Brownian Motion)
- **Context:** Financial forecasting often uses GBM (stochastic calculus).
- **Options considered:** GBM vs Deep Learning.
- **Chosen approach:** PyTorch LSTM trained on S&P 500.
- **Reason:** GBM assumes normal distribution of returns, failing to account for "fat tails" (frequent market crashes). An LSTM learns the actual historical volatility patterns.
- **Trade-offs:** Slower inference time (~3000ms vs ~100ms for numpy), higher CPU overhead.

## Decision 2: Replacing Passlib with raw Bcrypt
- **Context:** The backend needed secure password hashing.
- **Options considered:** `passlib` vs `bcrypt` direct integration.
- **Chosen approach:** Raw `bcrypt`.
- **Reason:** `passlib` is unmaintained and causes fatal `AttributeError` exceptions when used with `bcrypt >= 4.0`.

## Decision 3: Server-Sent Events (SSE) for AI Streaming
- **Context:** The frontend needs to stream AI text token-by-token.
- **Options considered:** WebSockets vs SSE.
- **Chosen approach:** SSE via `StreamingResponse` in FastAPI.
- **Reason:** SSE natively leverages HTTP/1.1 without the handshake complexity of WebSockets, making it perfectly suited for unidirectional text streaming from an LLM. WebSockets were deemed unnecessary overhead.
