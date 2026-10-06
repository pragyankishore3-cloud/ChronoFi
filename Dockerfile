# syntax=docker/dockerfile:1
FROM python:3.11-slim as builder

# Install uv for blazingly fast dependency installation
COPY --from=ghcr.io/astral-sh/uv:latest /uv /bin/uv

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# Copy dependency files
COPY pyproject.toml uv.lock ./

# Create virtual environment and install dependencies natively
RUN uv venv /app/.venv
ENV PATH="/app/.venv/bin:$PATH"
RUN uv sync --no-dev --locked

# Copy application code
COPY . .

# Final stage (distroless/minimal)
FROM python:3.11-slim

WORKDIR /app

# Copy the venv and app from builder
COPY --from=builder /app /app
ENV PATH="/app/.venv/bin:$PATH"

# Run Uvicorn without root privileges
RUN useradd -m appuser
USER appuser

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
