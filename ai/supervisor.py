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
import yfinance as yf
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

# --- INVESTMENT TOOLS ---
@tool
def get_stock_data(ticker: str) -> dict:
    """Fetches real-time stock prices and 3-year performance trends for a given ticker. Distinguishes real-time vs historical."""
    try:
        stock = yf.Ticker(ticker)
        hist = stock.history(period="3y")
        if hist.empty:
            return {"error": f"No data found for {ticker}", "status": "unavailable"}
        current_price = float(hist['Close'].iloc[-1])
        start_price = float(hist['Close'].iloc[0])
        performance = ((current_price - start_price) / start_price) * 100
        return {
            "symbol": ticker,
            "current_price": current_price,
            "performance_3y_percent": performance,
            "status": "real-time"
        }
    except Exception as e:
        return {"error": str(e), "status": "unavailable"}

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
    state_modifier="You are a Tax Agent. Calculate tax implications using tools. Do not invent tax rates or assumptions."
)

investment_agent = create_react_agent(
    llm, 
    tools=[get_stock_data], 
    state_modifier="You are an Investment Analyst. Fetch market data using tools. Distinguish real-time data from calculated metrics. Never fabricate missing market information."
)

budgeting_agent = create_react_agent(
    llm, 
    tools=[get_user_income, get_user_assets, get_user_liabilities], 
    state_modifier="You are a Budgeting Agent. Query the user's financial profile. You ONLY have access to the user's own data (enforced by tools). Never generate arbitrary SQL."
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

async def investment_node(state: AgentState, config: RunnableConfig):
    res = await investment_agent.ainvoke({"messages": state["messages"]}, config)
    return {"messages": [res["messages"][-1]]}

async def budgeting_node(state: AgentState, config: RunnableConfig):
    res = await budgeting_agent.ainvoke({"messages": state["messages"]}, config)
    return {"messages": [res["messages"][-1]]}

# Router
class Router(TypedDict):
    agents: List[Literal["Tax_Agent", "Investment_Analyst", "Budgeting_Agent", "FINISH"]]
    reason: str

supervisor_prompt = """You are a Supervisor orchestrating financial agents: Tax_Agent, Investment_Analyst, Budgeting_Agent.
User Query: {query}

Decide which agent(s) need to act. 
- If the user asks about taxes, route to Tax_Agent.
- If they ask about stocks or market, route to Investment_Analyst.
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
workflow.add_node("Investment_Analyst", investment_node)
workflow.add_node("Budgeting_Agent", budgeting_node)
workflow.add_node("supervisor", supervisor_node)

for member in ["Tax_Agent", "Investment_Analyst", "Budgeting_Agent"]:
    workflow.add_edge(member, "supervisor")

workflow.add_edge(START, "supervisor")
workflow.add_conditional_edges(
    "supervisor",
    lambda state: state["next"],
    {
        "Tax_Agent": "Tax_Agent",
        "Investment_Analyst": "Investment_Analyst",
        "Budgeting_Agent": "Budgeting_Agent",
        "FINISH": END
    }
)

supervisor_app = workflow.compile()
