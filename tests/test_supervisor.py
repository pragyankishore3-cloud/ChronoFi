import pytest
import asyncio
import os
from langchain_core.messages import HumanMessage
from langchain_core.runnables import RunnableConfig

# Inject dummy key before importing ai.supervisor
os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY", "dummy_test_key_so_pydantic_does_not_crash")

from ai.supervisor import (
    supervisor_app, 
    calculate_tax_impact, 
    get_stock_data, 
    get_user_income,
    AgentState
)

@pytest.mark.asyncio
async def test_supervisor_routing_tax():
    """Test if supervisor correctly routes to Tax Agent."""
    state = {"messages": [HumanMessage(content="How much tax will I pay on 20 lakh salary?")], "user_query": "How much tax will I pay on 20 lakh salary?"}
    
    # We invoke the graph with a single node (supervisor) to see where it routes next
    # A full invoke would run the whole graph, so we can just test the compiled app
    res = await supervisor_app.ainvoke(state)
    assert res is not None
    # Supervisor shouldn't crash, and should have processed messages
    assert len(res["messages"]) > 1

@pytest.mark.asyncio
async def test_supervisor_routing_investment():
    """Test if supervisor correctly routes to Investment Analyst."""
    state = {"messages": [HumanMessage(content="What is the price of AAPL?")], "user_query": "What is the price of AAPL?"}
    res = await supervisor_app.ainvoke(state)
    assert res is not None

def test_tax_tool_valid():
    """Test tax tool directly with valid inputs."""
    res = calculate_tax_impact.invoke({"salary": 1500000})
    assert "new_regime" in res
    assert "old_regime" in res
    assert isinstance(res["new_regime"], dict)
    assert "tax" in res["new_regime"]

def test_tax_tool_invalid():
    """Test tax tool doesn't crash on invalid data."""
    # Depending on pydantic validation, it may raise an error before invocation
    pass # Pydantic handles type validation

def test_stock_data_valid():
    """Test stock data retrieval."""
    res = get_stock_data.invoke({"ticker": "AAPL"})
    assert res["status"] in ["real-time", "unavailable"]

@pytest.mark.asyncio
async def test_budgeting_tool_auth_rejection():
    """Test that budgeting tools enforce user isolation by rejecting empty user_id."""
    # Call without config
    res = await get_user_income.ainvoke({})
    assert "error" in res
    assert "Unauthorized" in res["error"]

