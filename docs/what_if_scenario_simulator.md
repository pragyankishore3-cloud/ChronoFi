# What-If Scenario Simulator: Architecture & Technical Reference

## 1. Overview
The **What-If Scenario Simulator** is an enterprise-grade financial forecasting subsystem integrated into ChronoFi's AI Life Financial Planner. It empowers users to ask natural-language questions regarding major life and financial decisions (e.g. taking a ₹10,00,000 vehicle loan at 9% interest) and observe the simulated impact on their retirement date, corpus trajectory, monthly investable surplus, and goal probability.

---

## 2. Mathematical Engine Specifications

### 2.1 Loan Amortization Formula
Standard reducing-balance monthly amortization is calculated as:
$$EMI = \frac{P \cdot r \cdot (1 + r)^n}{(1 + r)^n - 1}$$
Where:
- $P$ = Loan principal (e.g. ₹10,00,000)
- $r$ = Monthly interest rate ($\frac{\text{Annual Rate}}{12}$)
- $n$ = Loan tenure in months (e.g. 60)

For edge cases where $r = 0$, $EMI = \frac{P}{n}$.
Total payment and total interest are computed with exact precision:
$$\text{Total Payment} = \text{Round}(EMI, 2) \times n$$
$$\text{Total Interest} = \text{Total Payment} - P$$

### 2.2 Cash Flow & Projection Engine
Month-by-month cash flow simulation tracks:
1. **Gross Monthly Surplus**:
   $$\text{Surplus}_{\text{baseline}} = \text{Monthly Income} - \text{Monthly Expenses} - \text{Existing EMIs}$$
   $$\text{Surplus}_{\text{scenario}} = \text{Surplus}_{\text{baseline}} - EMI_{\text{new}} - \text{Maintenance / Operational Costs}$$
2. **Corpus Accumulation**:
   For each month $m$ from current age to retirement age ($60$):
   $$\text{Corpus}_{m} = \text{Corpus}_{m-1} \cdot (1 + r_{\text{invest}}) + \text{Surplus}_{m}$$
   Where $r_{\text{invest}} = (1 + \text{Annual Return})^{1/12} - 1$.
3. **Retirement Delay Metric**:
   Evaluates the month when the scenario corpus reaches the baseline corpus at age 60:
   $$\Delta \text{Years} = \frac{\text{Months to reach Target Corpus} - \text{Baseline Horizon Months}}{12}$$

### 2.3 Paired Monte Carlo Simulation with Common Random Numbers (CRN)
To isolate the true scenario impact from stochastic noise:
- Both baseline and scenario paths are evaluated across $N = 500$ stochastic iterations.
- A single seed generates identical pseudo-random return shocks for each corresponding trial:
  $$R_{t, i} \sim \mathcal{N}(\mu, \sigma^2)$$
  $$\text{Seed} = 42 \quad (\text{or request-configurable})$$
- Metrics calculated:
  - Median final corpus ($P_{50}$)
  - Conservative percentile ($P_{10}$)
  - Optimistic percentile ($P_{90}$)
  - Conditional Value at Risk ($CVaR_{0.05}$)
  - Goal Success Probability: $\frac{1}{N} \sum_{i=1}^{N} \mathbb{I}(\text{Corpus}_{i} \ge \text{Target})$

---

## 3. LangGraph Workflow & Tool Orchestration

### 3.1 State Graph Nodes
1. `understand_request`: Classifies user intent into `create`, `modify`, `compare`, `recalculate`, or `explain`.
2. `load_financial_context`: Queries authenticated user profile, active assets, liabilities, and retirement goals.
3. `extract_scenario_parameters`: Regex & LLM-based typed parameter extraction with INR currency parsing (Lakh/Crore conversions).
4. `validate_scenario`: Guards numerical bounds (principal > 0, rate in [0, 100], tenure in [1, 480]). Emits clarification if material fields are missing.
5. `build_scenario`: Synthesizes validated payload with assumption overrides and tracking metadata.
6. `execute_scenario_tool`: Calls the `@tool` `run_financial_scenario` which calls the math engine with server-enforced `user_id`.
7. `validate_results`: Validates result status, schema consistency, non-negative bounds, and metric integrity.
8. `interpret_results`: Generates factual explanation grounded strictly in computed metrics.
9. `present_scenario`: Formats structured payload for both SSE streaming and frontend presentation.

---

## 4. API Endpoints

### `POST /api/v1/financial-scenarios/simulate`
- **Auth**: Bearer JWT (`deps.get_current_user`)
- **Headers**: `Idempotency-Key` (Optional UUID)
- **Request Body**:
```json
{
  "scenario_type": "vehicle_loan",
  "changes": {
    "loan_principal": 1000000.0,
    "annual_interest_rate": 0.09,
    "loan_tenure_months": 60,
    "down_payment": 0.0,
    "additional_monthly_expenses": 5000.0
  },
  "assumptions": [
    {
      "field": "loan_tenure_months",
      "value": 60,
      "source": "industry_standard_vehicle_loan_tenure"
    }
  ]
}
```
- **Response**: `SimulateScenarioResponse` with loan metrics, baseline vs scenario metrics, delta impact, 4 evaluated alternatives, and 26-year trajectory data.

---

## 5. Security & Isolation Guarantees
- **No LLM Privilege Escalation**: The scenario tool and routes extract `user_id` strictly from authenticated JWT / DB session, never allowing LLM prompt overrides.
- **Read-Only Invariant**: All calculations run purely in-memory. User liabilities and assets are never modified during simulation.
- **Audit Logging**: Completed runs are persisted to `scenario_records` table with engine version, timestamp, and input assumptions hash.
