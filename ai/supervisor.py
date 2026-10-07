import os
import json
from typing import Annotated, Literal, TypedDict, List, Dict, Any
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent

from sqlalchemy.future import select
from models.database import AsyncSessionLocal, User, Asset, Liability, Goal
from engine.tax_calc import calculate_new_regime_tax, calculate_old_regime_tax

class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    user_query: str
    user_id: int
    investment_data: dict
    tax_data: dict
    budget_data: dict
    next: str
    errors: list[str]

# --- TAX TOOLS ---
@tool
def calculate_tax_impact(salary: float, deductions: float = 0.0, agri_income: float = 0.0) -> dict:
    """Calculates income tax based on the new or old regime.
    Use this to calculate exact tax figures."""
    try:
        new_res = calculate_new_regime_tax(salary, agri_income)
        old_res = calculate_old_regime_tax(salary, deductions, agri_income)
        return {
            "new_regime": new_res,
            "old_regime": old_res
        }
    except Exception as e:
        return {"error": str(e)}

from engine.ml_forecaster import run_ml_forecaster
from engine.math_engine import run_scenario

# --- QUANT TOOLS ---
@tool
def simulate_goal_probability_tool(target_amount: float, current_savings: float, monthly_sip: float, years: int) -> str:
    """Runs a probabilistic machine learning forecast using PyTorch LSTM."""
    result = run_ml_forecaster(target_amount, current_savings, monthly_sip, years, num_simulations=500)
    return f"Probability of hitting target: {result['probability_of_success_percent']}%"

@tool
def run_scenario_tool(base_salary: float, monthly_expenses: float, months_unemployed: int) -> str:
    """Calculates financial impact of life events."""
    result = run_scenario(base_salary, monthly_expenses, months_unemployed)
    return f"Annual Surplus/Deficit: {result['annual_surplus']}"

# --- BUDGETING TOOLS ---
@tool
async def get_user_income(config: RunnableConfig) -> dict:
    """Gets the user's base salary/income. This enforces strict user isolation."""
    user_id = config.get("configurable", {}).get("user_id")
    if not user_id:
        return {"error": "Unauthorized: No user_id in context"}
    async with AsyncSessionLocal() as db:
        res = await db.execute(select(User).filter(User.id == user_id))
        user = res.scalars().first()
        if user:
            return {"base_salary": user.base_salary}
        return {"error": "User not found"}

@tool
async def get_user_assets(config: RunnableConfig) -> list:
    """Gets the user's assets. Enforces user isolation."""
    user_id = config.get("configurable", {}).get("user_id")
    if not user_id:
        return [{"error": "Unauthorized"}]
    async with AsyncSessionLocal() as db:
        res = await db.execute(select(Asset).filter(Asset.user_id == user_id))
        assets = res.scalars().all()
        return [{"name": a.name, "type": a.asset_type, "value": a.current_value} for a in assets]

@tool
async def get_user_liabilities(config: RunnableConfig) -> list:
    """Gets the user's liabilities. Enforces user isolation."""
    user_id = config.get("configurable", {}).get("user_id")
    if not user_id:
        return [{"error": "Unauthorized"}]
    async with AsyncSessionLocal() as db:
        res = await db.execute(select(Liability).filter(Liability.user_id == user_id))
        liabilities = res.scalars().all()
        return [{"name": l.name, "principal_remaining": l.principal_remaining, "monthly_payment": l.monthly_payment} for l in liabilities]


# --- AGENTS ---
# Using temperature=0.1 for more deterministic orchestration
llm = ChatGoogleGenerativeAI(model="gemini-2.5-pro", temperature=0.1)

tax_agent = create_react_agent(
    llm, 
    tools=[calculate_tax_impact], 
    prompt="You are a Tax Agent. Calculate tax implications using tools. Do not invent tax rates or assumptions."
)

quant_agent = create_react_agent(
    llm, 
    tools=[simulate_goal_probability_tool, run_scenario_tool], 
    prompt="You are an Quant Analyst. Run deep learning Monte Carlo simulations for the user goals."
)

budgeting_agent = create_react_agent(
    llm, 
    tools=[get_user_income, get_user_assets, get_user_liabilities], 
    prompt="You are a Budgeting Agent. Query the user's financial profile. You ONLY have access to the user's own data (enforced by tools). Never generate arbitrary SQL."
)

# Nodes
async def tax_node(state: AgentState, config: RunnableConfig):
    # Structured state sharing:
    # We pass the investment and budget data to the tax agent if they exist in the state.
    context = ""
    if state.get("investment_data"):
        context += f"Investment Data: {json.dumps(state['investment_data'])}\n"
    if state.get("budget_data"):
        context += f"Budget Data: {json.dumps(state['budget_data'])}\n"
    
    msgs = state["messages"]
    if context:
        msgs = [SystemMessage(content=context)] + msgs
        
    res = await tax_agent.ainvoke({"messages": msgs}, config)
    return {"messages": [res["messages"][-1]]}

async def quant_node(state: AgentState, config: RunnableConfig):
    res = await quant_agent.ainvoke({"messages": state["messages"]}, config)
    return {"messages": [res["messages"][-1]]}

async def budgeting_node(state: AgentState, config: RunnableConfig):
    res = await budgeting_agent.ainvoke({"messages": state["messages"]}, config)
    return {"messages": [res["messages"][-1]]}

# Router
class Router(TypedDict):
    agents: List[Literal["Tax_Agent", "Quant_Agent", "Budgeting_Agent", "FINISH"]]
    reason: str

supervisor_prompt = """You are a Supervisor orchestrating financial agents: Tax_Agent, Quant_Agent, Budgeting_Agent.
User Query: {query}

Decide which agent(s) need to act. 
- If the user asks about taxes, route to Tax_Agent.
- If they ask about stocks or market, route to Quant_Agent.
- If they ask about their income, spending, assets, or liabilities, route to Budgeting_Agent.
- If you need data from multiple domains, route to ONE agent at a time.
- Once you have enough information to answer the user fully, route to FINISH.

Return a structured output with the list of agents (or FINISH) and your reason."""

supervisor_agent = llm.with_structured_output(Router)

async def supervisor_node(state: AgentState, config: RunnableConfig):
    query = state.get("user_query", state["messages"][0].content if state["messages"] else "")
    sys_msg = SystemMessage(content=supervisor_prompt.format(query=query))
    
    # Pass conversation history to supervisor so it knows what was already answered
    routing = await supervisor_agent.ainvoke([sys_msg] + state["messages"])
    
    next_agent = "FINISH"
    if routing.get("agents") and routing["agents"][0] != "FINISH":
        next_agent = routing["agents"][0]
        
    return {"next": next_agent}

# Graph
workflow = StateGraph(AgentState)
workflow.add_node("Tax_Agent", tax_node)
workflow.add_node("Quant_Agent", quant_node)
workflow.add_node("Budgeting_Agent", budgeting_node)
workflow.add_node("supervisor", supervisor_node)

for member in ["Tax_Agent", "Quant_Agent", "Budgeting_Agent"]:
    workflow.add_edge(member, "supervisor")

workflow.add_edge(START, "supervisor")
workflow.add_conditional_edges(
    "supervisor",
    lambda state: state["next"],
    {
        "Tax_Agent": "Tax_Agent",
        "Quant_Agent": "Quant_Agent",
        "Budgeting_Agent": "Budgeting_Agent",
        "FINISH": END
    }
)

supervisor_app = workflow.compile()
