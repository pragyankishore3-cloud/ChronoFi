"""
ChronoFi What-If Scenario Simulator: LangGraph Stateful Workflow.

Orchestrates intent understanding, context resolution, historical memory retrieval,
parameter extraction, validation & clarification, tool execution, result validation,
and grounded financial trade-off interpretation with persistent checkpointers.
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

from ai.scenario_tools import confirm_and_record_liability, retrieve_previous_scenarios, run_financial_scenario
from core.checkpointer import get_checkpointer
import models.database as db_models
from models.database import User


class ScenarioWorkflowState(TypedDict):
    messages: Annotated[List[BaseMessage], add_messages]
    user_query: str
    user_id: int
    thread_id: Optional[str]
    active_scenario_id: Optional[str]
    retrieved_memory_refs: List[Dict[str, Any]]
    pending_confirmation: Optional[Dict[str, Any]]
    workflow_status: str  # "idle", "clarifying", "awaiting_confirmation", "simulated", "confirmed"
    request_intent: str   # "new_scenario", "resume_scenario", "confirm_transaction", "modify_scenario", "explain_scenario"
    financial_context: Dict[str, Any]
    extracted_params: Dict[str, Any]
    param_sources: Dict[str, str]
    validation_status: str  # "valid", "needs_clarification", "invalid", "cached_result"
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

    # Check for confirmation of pending transaction
    pending = state.get("pending_confirmation")
    if pending and any(w in q_lower for w in ["yes", "i took it", "already taken", "confirm", "record", "signed", "proceed"]):
        intent = "confirm_transaction"
    elif any(w in q_lower for w in ["remember", "last week", "we discussed", "earlier", "previous", "prior", "that car loan", "what did we decide", "decided to take"]):
        intent = "resume_scenario"
    elif any(w in q_lower for w in ["compare", "versus", "vs", "which is better"]):
        intent = "compare_scenarios"
    elif any(w in q_lower for w in ["modify", "change tenure", "change interest", "adjust"]):
        intent = "modify_scenario"
    elif any(w in q_lower for w in ["explain", "why", "breakdown", "clarify", "trade-off", "tradeoff", "trade off", "alternative"]):
        intent = "explain_scenario"

    thread_id = state.get("thread_id") or config.get("configurable", {}).get("thread_id")
    user_id = state.get("user_id") or config.get("configurable", {}).get("user_id")

    return {
        "user_query": query,
        "request_intent": intent,
        "thread_id": thread_id,
        "user_id": user_id,
        "errors": state.get("errors", []),
    }


# --- NODE 1B: Retrieve Memory ---
async def retrieve_memory(state: ScenarioWorkflowState, config: RunnableConfig) -> Dict[str, Any]:
    user_id = state.get("user_id") or config.get("configurable", {}).get("user_id")
    query = state.get("user_query", "")

    # Retrieve matching scenarios for this user
    res = await retrieve_previous_scenarios.ainvoke(
        {"search_query": query, "user_id": user_id, "limit": 5},
        config=config,
    )

    scenarios = res.get("scenarios", [])
    if not scenarios:
        # Fallback query without text filter
        res_all = await retrieve_previous_scenarios.ainvoke(
            {"user_id": user_id, "limit": 5},
            config=config,
        )
        scenarios = res_all.get("scenarios", [])

    if not scenarios:
        return {
            "validation_status": "needs_clarification",
            "clarification_prompt": "I couldn't find a record of that earlier loan discussion in your profile. Could you share the loan amount, interest rate, or tenure?",
            "retrieved_memory_refs": [],
        }

    matching = scenarios
    if len(matching) > 1:
        q_lower = query.lower()
        if "car" in q_lower or "vehicle" in q_lower:
            v_matches = [s for s in matching if "vehicle" in (s.get("scenario_type") or "").lower()]
            if v_matches:
                matching = v_matches

    if len(matching) > 1:
        options = ", ".join(f"'{s.get('name')}' (saved on {s.get('created_at', '')[:10]})" for s in matching[:3])
        return {
            "validation_status": "needs_clarification",
            "clarification_prompt": f"I found multiple previous scenarios: {options}. Which one would you like to continue with?",
            "retrieved_memory_refs": matching,
        }

    chosen = matching[0]
    return {
        "active_scenario_id": chosen["scenario_id"],
        "retrieved_memory_refs": [chosen],
        "validation_status": "valid",
    }


# --- NODE 1C: Resolve Intent and Freshness ---
async def resolve_intent_and_freshness(state: ScenarioWorkflowState, config: RunnableConfig) -> Dict[str, Any]:
    query = state.get("user_query", "").lower()
    scenario = state["retrieved_memory_refs"][0] if state.get("retrieved_memory_refs") else {}
    params = scenario.get("parameters", {})
    principal = params.get("loan_principal", 1000000.0)
    rate = params.get("annual_interest_rate", 0.09)
    tenure_mo = params.get("loan_tenure_months", 60)
    years = tenure_mo // 12
    lakhs = principal / 100000.0

    # Distinguish decision intention from confirmed financial transaction
    if any(k in query for k in ["decided to take", "taking it", "decided on taking", "going ahead", "i decided"]):
        prompt = (
            f"I found your previous car-loan scenario: ₹{lakhs:.0f} lakh at {rate*100:.0f}% interest over {years} years. "
            "Your earlier simulation showed its impact on your retirement plan. "
            "Have you already taken the loan, or would you like to update the plan assuming you will take it?"
        )
        return {
            "validation_status": "needs_clarification",
            "workflow_status": "awaiting_confirmation",
            "clarification_prompt": prompt,
            "pending_confirmation": {
                "action": "record_liability",
                "scenario_id": scenario.get("scenario_id"),
                "name": scenario.get("name", "Car Loan"),
                "principal": principal,
                "annual_interest_rate": rate,
                "loan_tenure_months": tenure_mo,
            },
        }

    # If user simply requests saved result
    if any(k in query for k in ["what did we", "show me", "what was", "result", "tell me about"]):
        summary = scenario.get("results_summary", {})
        corpus_diff = summary.get("retirement_corpus_diff", 0.0)
        delay = summary.get("retirement_age_delay_years", 0.0)
        msg = (
            f"Here is your saved scenario for the ₹{lakhs:.0f} lakh car loan at {rate*100:.0f}% interest over {years} years:\n"
            f"- Impact on Retirement Corpus: ₹{abs(corpus_diff):,.2f} reduction\n"
            f"- Retirement Milestone Delay: {delay:.2f} years\n"
            f"- Scenario Status: {scenario.get('status', 'simulated')}\n"
            "Would you like to adjust any assumptions or explore alternatives?"
        )
        return {
            "validation_status": "cached_result",
            "final_presentation": {
                "scenario_id": scenario.get("scenario_id"),
                "summary": msg,
                "is_cached": True,
            },
            "interpretation": msg,
        }

    return {"validation_status": "valid"}


# --- NODE 1D: Execute Confirmation ---
async def execute_confirmation(state: ScenarioWorkflowState, config: RunnableConfig) -> Dict[str, Any]:
    pending = state.get("pending_confirmation") or {}
    scenario_id = pending.get("scenario_id") or state.get("active_scenario_id")
    user_id = state.get("user_id") or config.get("configurable", {}).get("user_id")

    res = await confirm_and_record_liability.ainvoke(
        {
            "scenario_id": scenario_id,
            "name": pending.get("name", "Car Loan"),
            "principal": pending.get("principal"),
            "annual_interest_rate": pending.get("annual_interest_rate"),
            "loan_tenure_months": pending.get("loan_tenure_months"),
            "user_id": user_id,
        },
        config=config,
    )

    if res.get("status") == "error":
        return {
            "errors": [res.get("error", "Failed to confirm liability.")],
            "workflow_status": "error",
            "messages": [AIMessage(content=f"Error recording transaction: {res.get('error')}")],
        }

    msg = (
        f"I have officially recorded your **{res.get('name')}** of ₹{res.get('principal_remaining', 0):,.2f} "
        f"at {res.get('interest_rate', 0)*100:.1f}% APR with a monthly EMI of **₹{res.get('monthly_payment', 0):,.2f}** "
        "in your active financial profile. Your baseline financial roadmap has been updated with your new net "
        f"monthly surplus of **₹{res.get('updated_monthly_surplus', 0):,.2f}**."
    )

    return {
        "workflow_status": "confirmed",
        "pending_confirmation": None,
        "interpretation": msg,
        "final_presentation": {
            "action": "liability_recorded",
            "details": res,
            "summary": msg,
        },
        "messages": [AIMessage(content=msg)],
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
            context = {
                "user_id": user_id,
                "name": f"User {user_id}",
                "age": 35,
                "base_salary": 250000.0,
                "monthly_income": 250000.0,
                "monthly_expenses": 70000.0,
                "existing_liabilities": [],
                "total_existing_emis": 0.0,
                "assets": [],
                "total_savings": 2500000.0,
                "retirement_goal": {
                    "name": "Retirement",
                    "target_amount": 50000000.0,
                    "target_years": 25,
                    "inflation_rate": 0.06,
                },
            }
            return {
                "financial_context": context,
                "validation_status": "valid",
            }

        ret_goal = next((g for g in user.goals if "retire" in g.name.lower()), None)
        if not ret_goal and user.goals:
            ret_goal = max(user.goals, key=lambda g: g.target_years)

        context = {
            "user_id": user.id,
            "name": user.name,
            "age": user.age,
            "base_salary": user.base_salary,
            "monthly_income": user.base_salary,
            "monthly_expenses": user.monthly_expenses if user.monthly_expenses and user.monthly_expenses > 0 else (user.base_salary * 0.40),
            "existing_liabilities": [
                {"name": l.name, "monthly_payment": l.monthly_payment, "principal": l.principal_remaining}
                for l in user.liabilities
            ],
            "total_existing_emis": sum(l.monthly_payment for l in user.liabilities),
            "assets": [
                {"name": a.name, "type": a.asset_type, "value": a.current_value, "expected_return": a.expected_return_rate}
                for a in user.assets
            ],
            "total_savings": sum(a.current_value for a in user.assets),
            "retirement_goal": {
                "name": ret_goal.name if ret_goal else "Retirement",
                "target_amount": ret_goal.target_amount if ret_goal else (user.base_salary * 12.0) * 20.0,
                "target_years": ret_goal.target_years if ret_goal else max(5, 60 - user.age),
                "inflation_rate": ret_goal.inflation_rate if ret_goal else 0.06,
            },
        }

        return {
            "financial_context": context,
            "validation_status": "valid",
        }


# --- NODE 3: Extract Scenario Parameters ---
def _parse_inr_value(text: str) -> Optional[float]:
    lakh_match = re.search(r"(\d+(?:\.\d+)?)\s*(?:lakh|lakhs|lac|lacs|l)\b", text, re.IGNORECASE)
    if lakh_match:
        return float(lakh_match.group(1)) * 100000.0

    crore_match = re.search(r"(\d+(?:\.\d+)?)\s*(?:crore|crores|cr)\b", text, re.IGNORECASE)
    if crore_match:
        return float(crore_match.group(1)) * 10000000.0

    k_match = re.search(r"(\d+(?:\.\d+)?)\s*k\b", text, re.IGNORECASE)
    if k_match:
        return float(k_match.group(1)) * 1000.0

    rupee_match = re.search(r"(?:₹|rs\.?|inr)\s*([\d,]+(?:\.\d+)?)", text, re.IGNORECASE)
    if rupee_match:
        return float(rupee_match.group(1).replace(",", ""))

    return None


_parse_currency_amount = _parse_inr_value


def _parse_interest_rate(text: str) -> Optional[float]:
    rate_match = re.search(r"(\d+(?:\.\d+)?)\s*(?:%|percent)\s*(?:interest|apr|rate)?", text, re.IGNORECASE)
    if rate_match:
        return float(rate_match.group(1)) / 100.0
    return None


def _parse_tenure_months(text: str) -> Optional[int]:
    year_match = re.search(r"(\d+)\s*(?:year|years|yr|yrs)\b", text, re.IGNORECASE)
    if year_match:
        return int(year_match.group(1)) * 12

    month_match = re.search(r"(\d+)\s*(?:month|months|mo|mos)\b", text, re.IGNORECASE)
    if month_match:
        return int(month_match.group(1))

    return None


async def extract_scenario_parameters(state: ScenarioWorkflowState, config: RunnableConfig) -> Dict[str, Any]:
    query = state.get("user_query", "")
    params: Dict[str, Any] = {}
    sources: Dict[str, str] = {}

    # Check if continuing from retrieved scenario
    if state.get("retrieved_memory_refs"):
        prev = state["retrieved_memory_refs"][0].get("parameters", {})
        for k, v in prev.items():
            params[k] = v
            sources[k] = "retrieved_historical"

    # Category determination
    q_lower = query.lower()
    if any(k in q_lower for k in ["car", "vehicle", "auto"]):
        params["scenario_type"] = "vehicle_loan"
    elif any(k in q_lower for k in ["home", "mortgage", "house"]):
        params["scenario_type"] = "home_loan"
    elif any(k in q_lower for k in ["personal", "unsecured"]):
        params["scenario_type"] = "personal_loan"
    else:
        params.setdefault("scenario_type", "vehicle_loan")
    sources["scenario_type"] = "user_intent"

    # Principal extraction
    p_val = _parse_inr_value(query)
    if p_val is not None:
        params["loan_principal"] = p_val
        sources["loan_principal"] = "user_provided"
    elif "loan_principal" in params and params["loan_principal"] is not None:
        pass
    else:
        params["loan_principal"] = None
        sources["loan_principal"] = "missing"

    # Rate extraction
    r_val = _parse_interest_rate(query)
    if r_val is not None:
        params["annual_interest_rate"] = r_val
        sources["annual_interest_rate"] = "user_provided"
    elif "annual_interest_rate" not in params:
        params["annual_interest_rate"] = 0.09
        sources["annual_interest_rate"] = "assumed_default"

    # Tenure extraction
    t_val = _parse_tenure_months(query)
    if t_val is not None:
        params["loan_tenure_months"] = t_val
        sources["loan_tenure_months"] = "user_provided"
    elif "loan_tenure_months" not in params:
        params["loan_tenure_months"] = 60
        sources["loan_tenure_months"] = "assumed_default"

    # Down payment extraction
    dp_match = re.search(r"down\s*payment\s*(?:of)?\s*(?:₹|rs\.?)?\s*(\d+(?:\.\d+)?\s*(?:lakh|lac|cr)?)", query, re.IGNORECASE)
    if dp_match:
        dp_val = _parse_inr_value(dp_match.group(1))
        if dp_val is not None:
            params["down_payment"] = dp_val
            sources["down_payment"] = "user_provided"
    if "down_payment" not in params:
        params["down_payment"] = 0.0
        sources["down_payment"] = "assumed_zero"

    # Additional monthly vehicle running costs
    if "monthly_vehicle_expenses" not in params:
        params["monthly_vehicle_expenses"] = 0.0
        sources["monthly_vehicle_expenses"] = "assumed_zero"

    return {
        "extracted_params": params,
        "param_sources": sources,
    }


# --- NODE 4: Validate Scenario & Missing Parameters ---
async def validate_scenario(state: ScenarioWorkflowState, config: RunnableConfig) -> Dict[str, Any]:
    params = state.get("extracted_params", {})
    p = params.get("loan_principal")
    r = params.get("annual_interest_rate", 0.0)
    t = params.get("loan_tenure_months", 0)

    # Material missing parameter: principal amount
    if p is None or p <= 0:
        return {
            "validation_status": "needs_clarification",
            "workflow_status": "awaiting_clarification",
            "clarification_prompt": "Could you please clarify the loan amount you are considering? For example, ₹10 lakh or ₹20 lakh.",
        }

    errors = []
    if r < 0.0 or r > 1.0:
        errors.append(f"Interest rate {r*100}% is outside the realistic range (0% to 100%).")
    if t <= 0 or t > 480:
        errors.append(f"Loan tenure of {t} months must be between 1 and 480 months.")

    if errors:
        return {
            "validation_status": "invalid",
            "errors": errors,
        }

    return {
        "validation_status": "valid",
        "clarification_prompt": None,
    }


# --- NODE 5: Build Scenario Payload ---
async def build_scenario(state: ScenarioWorkflowState, config: RunnableConfig) -> Dict[str, Any]:
    params = state.get("extracted_params", {})
    sources = state.get("param_sources", {})

    assumptions = [
        {"field": "loan_tenure_months", "value": params.get("loan_tenure_months", 60), "source": sources.get("loan_tenure_months", "default")},
        {"field": "down_payment", "value": params.get("down_payment", 0.0), "source": sources.get("down_payment", "default")},
        {"field": "monthly_vehicle_expenses", "value": params.get("monthly_vehicle_expenses", 0.0), "source": sources.get("monthly_vehicle_expenses", "default")},
    ]

    payload = {
        "scenario_type": params.get("scenario_type", "vehicle_loan"),
        "loan_principal": params.get("loan_principal", 1000000.0),
        "annual_interest_rate": params.get("annual_interest_rate", 0.09),
        "loan_tenure_months": params.get("loan_tenure_months", 60),
        "down_payment": params.get("down_payment", 0.0),
        "monthly_vehicle_expenses": params.get("monthly_vehicle_expenses", 0.0),
        "parent_scenario_id": state.get("active_scenario_id"),
        "assumptions": assumptions,
    }

    return {"scenario_payload": payload}


# --- NODE 6: Execute Scenario Tool ---
async def execute_scenario_tool(state: ScenarioWorkflowState, config: RunnableConfig) -> Dict[str, Any]:
    payload = state.get("scenario_payload", {})
    user_id = state.get("user_id") or config.get("configurable", {}).get("user_id")

    try:
        tool_args = {
            "scenario_type": payload.get("scenario_type", "vehicle_loan"),
            "loan_principal": payload.get("loan_principal", 1000000.0),
            "annual_interest_rate": payload.get("annual_interest_rate", 0.09),
            "loan_tenure_months": payload.get("loan_tenure_months", 60),
            "down_payment": payload.get("down_payment", 0.0),
            "monthly_vehicle_expenses": payload.get("monthly_vehicle_expenses", 0.0),
            "parent_scenario_id": payload.get("parent_scenario_id"),
            "user_id": user_id,
        }

        result = await run_financial_scenario.ainvoke(tool_args, config=config)

        if result.get("status") == "error":
            return {
                "errors": [result.get("error", "Unknown engine calculation failure")],
                "simulation_result": None,
            }

        return {
            "simulation_result": result,
            "active_scenario_id": result.get("scenario_id"),
            "errors": [],
        }

    except Exception as e:
        return {
            "errors": [f"Scenario tool execution failed: {str(e)}"],
            "simulation_result": None,
        }


# --- NODE 7: Validate Results ---
async def validate_results(state: ScenarioWorkflowState, config: RunnableConfig) -> Dict[str, Any]:
    res = state.get("simulation_result")
    if not res:
        return {"errors": ["No simulation result returned from math engine."]}

    if res.get("status") != "success":
        return {"errors": [f"Math engine returned failure status: {res.get('error')}"]}

    impact = res.get("impact", {})
    if "corpus_difference" not in impact or "retirement_age_delay_years" not in impact:
        return {"errors": ["Required impact metrics missing in calculation response."]}

    return {"errors": []}


# --- NODE 8: Interpret Results ---
async def interpret_results(state: ScenarioWorkflowState, config: RunnableConfig) -> Dict[str, Any]:
    res = state.get("simulation_result") or {}
    loan = res.get("loan_metrics") or {}
    impact = res.get("impact") or {}
    baseline = res.get("baseline") or res.get("baseline_metrics") or {}
    scenario = res.get("modified") or res.get("scenario_metrics") or {}

    emi = loan.get("monthly_emi", loan.get("emi", 0.0))
    principal = loan.get("principal", 0.0)
    rate = loan.get("annual_interest_rate", 0.0)
    total_interest = loan.get("total_interest", 0.0)
    tenure_mo = loan.get("tenure_months", 60)
    tenure_yrs = tenure_mo // 12

    corpus_diff = impact.get("corpus_difference", 0.0)
    age_delay = impact.get("retirement_age_difference", impact.get("retirement_age_delay_years", 0.0))
    months_delay = int(round(age_delay * 12))
    surplus_red = impact.get("monthly_surplus_reduction", abs(impact.get("monthly_cash_flow_difference", emi)))

    base_corpus = baseline.get("projected_corpus", baseline.get("projected_corpus_at_retirement", 0.0))
    scen_corpus = scenario.get("projected_corpus", scenario.get("projected_corpus_at_retirement", 0.0))
    base_prob = baseline.get("goal_success_probability", baseline.get("goal_completion_probability", 1.0)) * 100.0
    scen_prob = scenario.get("goal_success_probability", scenario.get("goal_completion_probability", 1.0)) * 100.0

    alternatives = res.get("alternatives", [])
    alt_text = ""
    if alternatives:
        alt_lines = "\n".join(f"- **{a.get('name')}:** {a.get('description', '')}" for a in alternatives)
        alt_text = f"\n\n**Actionable Alternatives to Mitigate Impact:**\n{alt_lines}"

    goal_impacts = res.get("goal_impacts", [])
    goal_impact_text = ""
    if goal_impacts and len(goal_impacts) > 1:
        gi_lines = []
        for gi in goal_impacts:
            status_icon = "🟢" if gi.get("modified_status") == "On Track" else ("🟡" if gi.get("modified_status") == "At Risk" else "🔴")
            gi_lines.append(
                f"- {status_icon} **{gi.get('name')} ({gi.get('priority')}):** "
                f"{gi.get('baseline_funding_ratio')}% → {gi.get('modified_funding_ratio')}% funded. "
                f"{gi.get('trade_off_note')}"
            )
        goal_impact_text = f"\n\n**Multi-Goal Life Impact Analysis:**\n" + "\n".join(gi_lines)

    narrative = (
        f"Under the assumptions selected ({tenure_yrs}-year tenure), "
        f"the ₹{principal/100000:.1f} Lakh vehicle loan at {rate*100:.1f}% interest requires a monthly EMI of ₹{emi:,.2f}, "
        f"reducing your monthly investable surplus by ₹{surplus_red:,.2f}.\n\n"
        f"**Retirement Impact:**\n"
        f"- Projected retirement corpus shifts from ₹{base_corpus/10000000:.2f} Cr to ₹{scen_corpus/10000000:.2f} Cr "
        f"(-₹{abs(corpus_diff)/100000:.2f} Lakh net impact due to diverted compounding).\n"
        f"- Equivalent retirement milestone is delayed by **{age_delay:.2f} years** (~{months_delay} months).\n"
        f"- Goal achievement probability shifts from {base_prob:.1f}% to {scen_prob:.1f}%.\n"
        f"- Total loan interest paid over {tenure_yrs} years will be ₹{total_interest:,.2f}."
        f"{goal_impact_text}"
        f"{alt_text}"
    )

    return {"interpretation": narrative}


# --- NODE 9: Present Scenario ---
async def present_scenario(state: ScenarioWorkflowState, config: RunnableConfig) -> Dict[str, Any]:
    interp = state.get("interpretation") or "Scenario simulation complete."
    res = state.get("simulation_result") or {}

    presentation = {
        "presentation_type": "scenario_comparison",
        "scenario_id": state.get("active_scenario_id"),
        "narrative": interp,
        "data": res,
        "structured_data": res,
        "alternatives": res.get("alternatives", []),
        "trajectory": res.get("trajectory", []),
    }

    return {
        "final_presentation": presentation,
        "messages": [AIMessage(content=interp)],
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
def route_after_understanding(state: ScenarioWorkflowState) -> Literal["retrieve_memory", "execute_confirmation", "interpret_results", "load_financial_context"]:
    intent = state.get("request_intent")
    if intent == "confirm_transaction" and state.get("pending_confirmation"):
        return "execute_confirmation"
    elif intent == "resume_scenario":
        return "retrieve_memory"
    elif intent == "explain_scenario" and state.get("simulation_result"):
        return "interpret_results"
    return "load_financial_context"


def route_after_retrieve(state: ScenarioWorkflowState) -> Literal["resolve_intent_and_freshness", "clarification_node"]:
    if state.get("validation_status") == "needs_clarification":
        return "clarification_node"
    return "resolve_intent_and_freshness"


def route_after_intent(state: ScenarioWorkflowState) -> Literal["clarification_node", "present_scenario", "load_financial_context"]:
    st = state.get("validation_status")
    if st == "needs_clarification":
        return "clarification_node"
    elif st == "cached_result":
        return "present_scenario"
    return "load_financial_context"


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
scenario_workflow.add_node("retrieve_memory", retrieve_memory)
scenario_workflow.add_node("resolve_intent_and_freshness", resolve_intent_and_freshness)
scenario_workflow.add_node("execute_confirmation", execute_confirmation)
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

# Flow Wiring
scenario_workflow.add_edge(START, "understand_request")

scenario_workflow.add_conditional_edges(
    "understand_request",
    route_after_understanding,
    {
        "retrieve_memory": "retrieve_memory",
        "execute_confirmation": "execute_confirmation",
        "interpret_results": "interpret_results",
        "load_financial_context": "load_financial_context",
    },
)

scenario_workflow.add_conditional_edges(
    "retrieve_memory",
    route_after_retrieve,
    {
        "resolve_intent_and_freshness": "resolve_intent_and_freshness",
        "clarification_node": "clarification_node",
    },
)

scenario_workflow.add_conditional_edges(
    "resolve_intent_and_freshness",
    route_after_intent,
    {
        "clarification_node": "clarification_node",
        "present_scenario": "present_scenario",
        "load_financial_context": "load_financial_context",
    },
)

scenario_workflow.add_edge("execute_confirmation", END)
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

# Compile with durable checkpointer
scenario_workflow_app = scenario_workflow.compile(checkpointer=get_checkpointer())
