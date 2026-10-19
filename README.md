# ChronoFi Backend — Multi-Agent Financial Intelligence Platform

![Python](https://img.shields.io/badge/Python-3.11%20%7C%203.12-3776AB?style=flat-square&logo=python)
![FastAPI](https://img.shields.io/badge/FastAPI-0.110+-009688?style=flat-square&logo=fastapi)
![LangGraph](https://img.shields.io/badge/LangGraph-Multi--Agent-FF6F00?style=flat-square)
![PyTorch](https://img.shields.io/badge/PyTorch-LSTM%20Forecaster-EE4C2C?style=flat-square&logo=pytorch)
![Google Gemini](https://img.shields.io/badge/Google%20Gemini-Flash%203.5%20%7C%203.6%20%7C%203.8-4285F4?style=flat-square&logo=google)

**ChronoFi Backend** is the computational and agentic reasoning backbone of ChronoFi. Built on **FastAPI**, **LangGraph**, and **PyTorch**, it orchestrates specialized autonomous agents (Tax, Quant, and Budgeting) along with deep probabilistic Monte Carlo dropout simulations to deliver institutional-grade financial intelligence.

---

## 🏛️ System Architecture

```
                      +-----------------------------+
                      |   ChronoFi-UI / Frontend    |
                      +--------------+--------------+
                                     |
                          SSE Streaming & REST
                                     |
                                     v
                      +-----------------------------+
                      |     FastAPI Gateway         |
                      |   (Auth, CORS, Security)    |
                      +--------------+--------------+
                                     |
              +----------------------+----------------------+
              |                      |                      |
              v                      v                      v
     +-----------------+    +-----------------+    +-----------------+
     |   Tax Agent     |    |   Quant Agent   |    | Budgeting Agent |
     | (Finance Act 24)|    | (PyTorch LSTM)  |    | (Runway/Debts)  |
     +-----------------+    +-----------------+    +-----------------+
              |                      |                      |
              +----------------------+----------------------+
                                     |
                       LangGraph Supervisor Engine
                                     |
                                     v
              +----------------------+----------------------+
              |  Google Gemini AI    |  SQLite / PostgreSQL |
              |  (Flash 3.5 / 3.8)   |   Durable Checkpoint |
              +----------------------+----------------------+
```

---

## 🚀 Key Features

### 1. LangGraph Multi-Agent Supervisor
- **Tax Agent**: Implements the Indian Finance Act 2024 provisions (New Regime standard deduction ₹75,000, Section 87A rebate up to ₹12 Lakhs, equity LTCG harvesting limits).
- **Quant Agent**: Executes probabilistic deep learning forecasts with 500-iteration Monte Carlo dropout to predict goal achievement probabilities.
- **Budgeting Agent**: Enforces strict user tenant isolation while analyzing cash flow, runway, and debt amortization.

### 2. PyTorch LSTM Forecaster & Monte Carlo Engine
- Computes inflated target amounts, expected median corpus (50th percentile), worst-case scenario (10th percentile), and Conditional Value at Risk (CVaR).

### 3. Google Gemini Model Agility
- Integrates Google Gemini Flash via official SDKs.
- Configurable model fallback supporting `gemini-3.5-flash`, `gemini-3.6-flash`, `gemini-3.5-flash-lite`, and `gemini-3.8-flash`.

---

## 🛠️ Quick Start

### 1. Prerequisites
- Python 3.11+
- Virtual environment tool (`venv` or `uv`)

### 2. Installation
```bash
git clone https://github.com/piyushx502/ChronoFi.git
cd ChronoFi

# Create and activate virtual environment
python -m venv .venv
# On Windows:
.\.venv\Scripts\activate
# On Linux/macOS:
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt  # or uv sync
```

### 3. Environment Configuration
Copy `.env.example` to `.env`:
```bash
cp .env.example .env
```
Populate your environment variables:
```ini
GEMINI_API_KEY=your_gemini_api_key_here
GEMINI_MODEL=gemini-3.5-flash
SECRET_KEY=adf92afde4779256b69569807041e79052493156930fcce512a87d111ae3067c
JWT_SECRET_KEY=adf92afde4779256b69569807041e79052493156930fcce512a87d111ae3067c
DATABASE_URL=sqlite+aiosqlite:///./chronofi.db
ENVIRONMENT=development
PORT=8000
```

### 4. Database Seeding & Startup
```bash
# Start FastAPI server
uvicorn main:app --reload --port 8000
```
Interactive Swagger API documentation is available at [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs).

---

## 🧪 Testing

Run automated tests:
```bash
pytest -v
```

---

## 📄 License
MIT License. Built for quantitative financial planning and research.
