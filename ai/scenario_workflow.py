"""
ChronoFi What-If Scenario Simulator: LangGraph Stateful Workflow.

Orchestrates intent understanding, context resolution, parameter extraction,
validation & clarification, tool execution, result validation, and
grounded financial trade-off interpretation.
"""

from typing import Annotated, Any, Dict, List, Literal, Optional, TypedDict
import json
import os
import re

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from sqlalchemy import select
from sqlalchemy.orm import selectinload

from ai.scenario_tools import run_financial_scenario
import models.database as db_models
from models.database import User


class ScenarioWorkflowState(TypedDict):
    messages: Annotated[List[BaseMessage], add_messages]
    user_query: str
    user_id: int
    request_intent: str
    financial_context: Dict[str, Any]
    extracted_params: Dict[str, Any]
    param_sources: Dict[str, str]
    validation_status: str  # "valid", "needs_clarification", "invalid"
    clarification_prompt: Optional[str]
    scenario_payload: Optional[Dict[str, Any]]
    simulation_result: Optional[Dict[str, Any]]
    interpretation: Optional[str]
    final_presentation: Optional[Dict[str, Any]]
    errors: List[str]


# --- NODE 1: Understand Request ---
async def understand_request(state: ScenarioWorkflowState, config: RunnableConfig) -> Dict[str, Any]:
    query = state.get("user_query", "")
    if not query and state.get("messages"):
        query = state["messages"][-1].content

    q_lower = query.lower()
    intent = "new_scenario"
    if any(w in q_lower for w in ["compare", "versus", "vs", "which is better"]):
        intent = "compare_scenarios"
    elif any(w in q_lower for w in ["modify", "change tenure", "change interest", "adjust"]):
        intent = "modify_scenario"
    elif any(w in q_lower for w in ["explain", "why", "breakdown", "clarify"]):
        intent = "explain_scenario"

    return {
        "user_query": query,
        "request_intent": intent,
        "errors": state.get("errors", []),
    }


# --- NODE 2: Load Financial Context ---
async def load_financial_context(state: ScenarioWorkflowState, config: RunnableConfig) -> Dict[str, Any]:
    user_id = state.get("user_id") or config.get("configurable", {}).get("user_id")
    if not user_id:
        return {
            "validation_status": "invalid",
            "errors": ["Unauthorized: user_id is missing from execution context."],
        }

    async with db_models.AsyncSessionLocal() as db:
        res = await db.execute(
            select(User)
            .where(User.id == user_id)
            .options(
                selectinload(User.assets),
                selectinload(User.liabilities),
                selectinload(User.goals),
            )
        )
        user = res.scalars().first()
        if not user:
            return {
                "validation_status": "invalid",
                "errors": [f"User profile with id={user_id} not found."],
            }

        current_savings = sum(a.current_value for a in user.assets) if user.assets else 0.0
        existing_debt_monthly = sum(l.monthly_payment for l in user.liabilities) if user.liabilities else 0.0

        # Weighted return
        if user.assets and current_savings > 0:
            weighted_ret = sum(a.current_value * (a.expected_return_rate or 0.10) for a in user.assets) / current_savings
        else:
            weighted_ret = 0.105

        ret_goal = next((g for g in user.goals if "retire" in g.name.lower()), None)
        if not ret_goal and user.goals:
            ret_goal = max(user.goals, key=lambda g: g.target_years)

        context = {
            "user_id": user.id,
            "name": user.name,
            "age": user.age,
            "base_salary": user.base_salary,
            "current_savings": current_savings,
            "existing_debt_monthly": existing_debt_monthly,
            "expected_annual_return": round(weighted_ret, 4),
            "retirement_goal_amount": ret_goal.target_amount if ret_goal else (user.base_salary * 12 * 20),
            "retirement_target_years": ret_goal.target_years if ret_goal else max(5, 60 - user.age),
            "inflation_rate": ret_goal.inflation_rate if ret_goal else 0.06,
        }

        return {
            "user_id": user_id,
            "financial_context": context,
        }


# --- NODE 3: Extract Scenario Parameters ---
def _parse_currency_amount(text: str) -> Optional[float]:
    """Parses Indian currency denominations (e.g. ₹10 Lakh, 10L, 50k, 1 crore)."""
    text_clean = text.replace(",", "")
    
    # Matches "10 lakh", "10lakh", "10 lakh", "10 l"
    lakh_match = re.search(r'(?:₹|rs\.?|inr)?\s*(\d+(?:\.\d+)?)\s*(?:lakh|lac|l)\b', text_clean, re.IGNORECASE)
    if lakh_match:
        return float(lakh_match.group(1)) * 100000.0

    # Matches "1 crore", "1cr", "1.5 crore"
    cr_match = re.search(r'(?:₹|rs\.?|inr)?\s*(\d+(?:\.\d+)?)\s*(?:crore|cr)\b', text_clean, re.IGNORECASE)
    if cr_match:
        return float(cr_match.group(1)) * 10000000.0

    # Matches "50k", "50 thousand"
    k_match = re.search(r'(?:₹|rs\.?|inr)?\s*(\d+(?:\.\d+)?)\s*(?:k|thousand)\b', text_clean, re.IGNORECASE)
    if k_match:
        return float(k_match.group(1)) * 1000.0

    # Matches raw numbers like "₹1000000" or "1000000"
    num_match = re.search(r'(?:₹|rs\.?|inr)?\s*(\d{5,9})\b', text_clean, re.IGNORECASE)
    if num_match:
        return float(num_match.group(1))

    return None


def _parse_interest_rate(text: str) -> Optional[float]:
    """Parses interest rates like 9%, 9.5%, 9 percent."""
    match = re.search(r'(\d+(?:\.\d+)?)\s*(?:%|percent)', text, re.IGNORECASE)
    if match:
        rate_val = float(match.group(1))
        # If entered as 9, convert to decimal 0.09
        if rate_val > 1.0:
            return rate_val / 100.0
        return rate_val
    return None


def _parse_tenure_months(text: str) -> Optional[int]:
    """Parses loan tenure in years or months."""
    year_match = re.search(r'(\d+)\s*(?:years?|yrs?)\b', text, re.IGNORECASE)
    if year_match:
        return int(year_match.group(1)) * 12

    month_match = re.search(r'(\d+)\s*(?:months?|mos?)\b', text, re.IGNORECASE)
    if month_match:
        return int(month_match.group(1))

    return None


async def extract_scenario_parameters(state: ScenarioWorkflowState, config: RunnableConfig) -> Dict[str, Any]:
    query = state.get("user_query", "")
    params = {}
    sources = {}

    # 1. Principal
    principal = _parse_currency_amount(query)
    if principal is not None:
        params["loan_principal"] = principal
        sources["loan_principal"] = "user_provided"
    else:
        params["loan_principal"] = None
        sources["loan_principal"] = "missing"

    # 2. Interest Rate
    rate = _parse_interest_rate(query)
    if rate is not None:
        params["annual_interest_rate"] = rate
        sources["annual_interest_rate"] = "user_provided"
    else:
        params["annual_interest_rate"] = None
        sources["annual_interest_rate"] = "missing"

    # 3. Tenure
    tenure = _parse_tenure_months(query)
    if tenure is not None:
        params["loan_tenure_months"] = tenure
        sources["loan_tenure_months"] = "user_provided"
    else:
        params["loan_tenure_months"] = 60  # Default 5-year vehicle loan assumption
        sources["loan_tenure_months"] = "explicit_default"

    # 4. Down Payment
    # Check if down payment explicitly mentioned
    dp_match = re.search(r'down\s*payment\s*(?:of)?\s*(?:₹|rs\.?)?\s*(\d+(?:\.\d+)?\s*(?:lakh|lac|k|thousand)?)', query, re.IGNORECASE)
    if dp_match:
        parsed_dp = _parse_currency_amount(dp_match.group(0))
        params["down_payment"] = parsed_dp or 0.0
        sources["down_payment"] = "user_provided"
    else:
        params["down_payment"] = 0.0
        sources["down_payment"] = "explicit_default"

    # 5. Category
    q_lower = query.lower()
    if any(k in q_lower for k in ["car", "vehicle", "auto"]):
        params["scenario_type"] = "vehicle_loan"
    elif any(k in q_lower for k in ["home", "house", "mortgage"]):
        params["scenario_type"] = "home_loan"
    elif any(k in q_lower for k in ["personal"]):
        params["scenario_type"] = "personal_loan"
    else:
        params["scenario_type"] = "vehicle_loan"

    return {
        "extracted_params": params,
        "param_sources": sources,
    }


# --- NODE 4: Validate Scenario ---
async def validate_scenario(state: ScenarioWorkflowState, config: RunnableConfig) -> Dict[str, Any]:
    params = state.get("extracted_params", {})
    principal = params.get("loan_principal")
    rate = params.get("annual_interest_rate")

    # Check for missing material parameters
    if principal is None and rate is None:
        prompt = (
            "To simulate this decision accurately, could you clarify the loan amount and expected interest rate? "
            "(For example: '₹10 lakh loan at 9% interest for 5 years')."
        )
        return {
            "validation_status": "needs_clarification",
            "clarification_prompt": prompt,
        }

    if principal is None:
        prompt = "What is the principal amount you are planning to borrow for this vehicle?"
        return {
            "validation_status": "needs_clarification",
            "clarification_prompt": prompt,
        }

    if rate is None:
        # If rate is missing, we can ask for clarification or use prevailing market vehicle rate (e.g. 9.0%)
        params["annual_interest_rate"] = 0.09
        state["param_sources"]["annual_interest_rate"] = "explicit_default"

    # Numeric range validation
    if principal <= 0 or principal > 100000000.0:
        return {
            "validation_status": "invalid",
            "errors": [f"Loan principal of ₹{principal:,.0f} is outside feasible simulation range."],
        }

    if params["annual_interest_rate"] < 0 or params["annual_interest_rate"] > 1.0:
        return {
            "validation_status": "invalid",
            "errors": [f"Interest rate {params['annual_interest_rate']} is outside valid range (0.0 to 1.0)."],
        }

    return {
        "validation_status": "valid",
        "extracted_params": params,
        "clarification_prompt": None,
    }


# --- NODE 5: Build Scenario ---
async def build_scenario(state: ScenarioWorkflowState, config: RunnableConfig) -> Dict[str, Any]:
    params = state["extracted_params"]
    payload = {
        "scenario_type": params.get("scenario_type", "vehicle_loan"),
        "loan_principal": params["loan_principal"],
        "annual_interest_rate": params["annual_interest_rate"],
        "loan_tenure_months": params.get("loan_tenure_months", 60),
        "down_payment": params.get("down_payment", 0.0),
        "monthly_vehicle_expenses": params.get("monthly_vehicle_expenses", 0.0),
    }

    return {
        "scenario_payload": payload,
    }


# --- NODE 6: Execute Scenario Tool ---
async def execute_scenario_tool(state: ScenarioWorkflowState, config: RunnableConfig) -> Dict[str, Any]:
    payload = dict(state.get("scenario_payload", {}))
    user_id = state.get("user_id") or (config.get("configurable", {}).get("user_id") if config else None)
    payload["user_id"] = user_id

    tool_config = {"configurable": {"user_id": user_id}}
    try:
        sim_result = await run_financial_scenario.ainvoke(payload, config=tool_config)
        if "error" in sim_result:
            return {
                "errors": [sim_result["error"]],
                "simulation_result": None,
            }
        return {
            "simulation_result": sim_result,
        }
    except Exception as e:
        return {
            "errors": [f"Execution error invoking calculation tool: {str(e)}"],
            "simulation_result": None,
        }


# --- NODE 7: Validate Results ---
async def validate_results(state: ScenarioWorkflowState, config: RunnableConfig) -> Dict[str, Any]:
    result = state.get("simulation_result")
    if not result:
        return {
            "validation_status": "invalid",
            "errors": state.get("errors", []) or ["Simulation engine returned empty result."],
        }

    # Verify structural integrity
    required_keys = ["baseline", "modified", "impact", "loan_metrics", "alternatives"]
    for k in required_keys:
        if k not in result:
            return {
                "validation_status": "invalid",
                "errors": [f"Malformed engine result: missing key '{k}'."],
            }

    # Numerical consistency checks
    baseline = result["baseline"]
    modified = result["modified"]
    impact = result["impact"]

    if modified["monthly_investable_surplus"] > baseline["monthly_investable_surplus"] + 0.01:
        return {
            "validation_status": "invalid",
            "errors": ["Inconsistent math: loan obligation increased investable surplus."],
        }

    return {
        "validation_status": "valid",
    }


# --- NODE 8: Interpret Results ---
async def interpret_results(state: ScenarioWorkflowState, config: RunnableConfig) -> Dict[str, Any]:
    res = state["simulation_result"]
    loan = res["loan_metrics"]
    baseline = res["baseline"]
    modified = res["modified"]
    impact = res["impact"]
    alternatives = res.get("alternatives", [])
    assumptions = res.get("effective_assumptions", [])

    emi = loan["monthly_emi"]
    tenure_years = loan["tenure_months"] // 12
    total_interest = loan["total_interest"]
    cash_flow_diff = abs(impact["monthly_cash_flow_difference"])
    age_diff = impact["retirement_age_difference"]
    corpus_diff = abs(impact["corpus_difference"])
    prob_drop = abs(impact["success_probability_difference"])

    # Documented assumptions string
    assumptions_summary = []
    for a in assumptions:
        if a["source"] == "explicit_default":
            assumptions_summary.append(f"{a['description']}: {a['value']} ({a['source']})")

    # Format authoritative interpretation text
    interpretation_lines = [
        f"Under the assumptions you selected (₹{loan['principal']:,.0f} vehicle loan at {loan['annual_interest_rate']*100:.1f}% for {loan['tenure_months']} months), the car loan reduces your monthly investable surplus by ₹{cash_flow_diff:,.0f} (monthly EMI of ₹{emi:,.0f}).",
        f"\n**Retirement Impact:**",
        f"• **Projected Retirement Age:** Shifts from **{baseline['retirement_age']}** to **{modified['retirement_age']}** (+{age_diff} years delay).",
        f"• **Projected Retirement Corpus:** Changes from ₹{baseline['projected_corpus']/10000000:.2f} Cr to ₹{modified['projected_corpus']/10000000:.2f} Cr (a compounding deficit of ₹{corpus_diff/100000:.2f} Lakh).",
        f"• **Goal Success Probability:** Decreases from **{baseline['goal_success_probability']}%** to **{modified['goal_success_probability']}%** (-{prob_drop:.1f}%).",
        f"• **Total Financing Cost:** You will pay ₹{total_interest:,.0f} in cumulative interest over {loan['tenure_months']} months.",
    ]

    if assumptions_summary:
        interpretation_lines.append(f"\n*Explicit Assumptions Review:* {', '.join(assumptions_summary)}.")

    interpretation_lines.append("\n**Actionable Alternatives to Mitigate This Impact:**")
    for alt in alternatives[:3]:
        interpretation_lines.append(
            f"1. **{alt['name']}**: EMI ₹{alt['monthly_emi']:,.0f}/mo. {alt['trade_off_summary']} (Success probability: {alt['goal_success_probability']}%)."
        )

    interpretation = "\n".join(interpretation_lines)

    return {
        "interpretation": interpretation,
    }


# --- NODE 9: Present Scenario ---
async def present_scenario(state: ScenarioWorkflowState, config: RunnableConfig) -> Dict[str, Any]:
    interpretation = state.get("interpretation", "")
    sim_result = state.get("simulation_result", {})

    presentation = {
        "text": interpretation,
        "structured_data": sim_result,
        "assumptions": sim_result.get("effective_assumptions", []),
        "impact": sim_result.get("impact", {}),
        "alternatives": sim_result.get("alternatives", []),
        "trajectory": sim_result.get("trajectory", []),
    }

    ai_msg = AIMessage(content=interpretation)

    return {
        "final_presentation": presentation,
        "messages": [ai_msg],
    }


# --- Clarification Node ---
async def clarification_node(state: ScenarioWorkflowState, config: RunnableConfig) -> Dict[str, Any]:
    prompt = state.get("clarification_prompt", "Could you provide additional details about this loan?")
    return {
        "messages": [AIMessage(content=prompt)],
    }


# --- Error Handler Node ---
async def error_node(state: ScenarioWorkflowState, config: RunnableConfig) -> Dict[str, Any]:
    err_text = "; ".join(state.get("errors", ["An unexpected error occurred during scenario simulation."]))
    return {
        "messages": [AIMessage(content=f"Simulation Error: {err_text}")],
    }


# --- Graph Routing Logic ---
def route_after_validation(state: ScenarioWorkflowState) -> Literal["build_scenario", "clarification_node", "error_node"]:
    status = state.get("validation_status")
    if status == "needs_clarification":
        return "clarification_node"
    elif status == "invalid":
        return "error_node"
    return "build_scenario"


def route_after_tool(state: ScenarioWorkflowState) -> Literal["validate_results", "error_node"]:
    if state.get("errors") or not state.get("simulation_result"):
        return "error_node"
    return "validate_results"


# --- Assemble Workflow Graph ---
scenario_workflow = StateGraph(ScenarioWorkflowState)

scenario_workflow.add_node("understand_request", understand_request)
scenario_workflow.add_node("load_financial_context", load_financial_context)
scenario_workflow.add_node("extract_scenario_parameters", extract_scenario_parameters)
scenario_workflow.add_node("validate_scenario", validate_scenario)
scenario_workflow.add_node("build_scenario", build_scenario)
scenario_workflow.add_node("execute_scenario_tool", execute_scenario_tool)
scenario_workflow.add_node("validate_results", validate_results)
scenario_workflow.add_node("interpret_results", interpret_results)
scenario_workflow.add_node("present_scenario", present_scenario)
scenario_workflow.add_node("clarification_node", clarification_node)
scenario_workflow.add_node("error_node", error_node)

# Flow
scenario_workflow.add_edge(START, "understand_request")
scenario_workflow.add_edge("understand_request", "load_financial_context")
scenario_workflow.add_edge("load_financial_context", "extract_scenario_parameters")
scenario_workflow.add_edge("extract_scenario_parameters", "validate_scenario")

scenario_workflow.add_conditional_edges(
    "validate_scenario",
    route_after_validation,
    {
        "build_scenario": "build_scenario",
        "clarification_node": "clarification_node",
        "error_node": "error_node",
    },
)

scenario_workflow.add_edge("build_scenario", "execute_scenario_tool")
scenario_workflow.add_conditional_edges(
    "execute_scenario_tool",
    route_after_tool,
    {
        "validate_results": "validate_results",
        "error_node": "error_node",
    },
)

scenario_workflow.add_edge("validate_results", "interpret_results")
scenario_workflow.add_edge("interpret_results", "present_scenario")
scenario_workflow.add_edge("present_scenario", END)
scenario_workflow.add_edge("clarification_node", END)
scenario_workflow.add_edge("error_node", END)

scenario_workflow_app = scenario_workflow.compile()
