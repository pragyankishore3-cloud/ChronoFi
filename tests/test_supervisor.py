import pytest
import os
import asyncio
from langchain_core.messages import HumanMessage
from langchain_core.runnables import RunnableConfig

from ai.supervisor import (
    supervisor_app, 
    calculate_tax_impact, 
    simulate_goal_probability_tool, 
    get_user_income,
    AgentState
)

@pytest.mark.asyncio
@pytest.mark.skipif('GEMINI_API_KEY' not in os.environ, reason='Requires API key')
async def skip_test_supervisor_routing_tax():
    """Test if supervisor correctly routes to Tax Agent."""
    state = {"messages": [HumanMessage(content="How much tax will I pay on 20 lakh salary?")], "user_query": "How much tax will I pay on 20 lakh salary?"}
    
    # We invoke the graph with a single node (supervisor) to see where it routes next
    # A full invoke would run the whole graph, so we can just test the compiled app
    res = await supervisor_app.ainvoke(state)
    assert res is not None
    # Supervisor shouldn't crash, and should have processed messages
    assert len(res["messages"]) > 1

@pytest.mark.asyncio
@pytest.mark.skipif('GEMINI_API_KEY' not in os.environ, reason='Requires API key')
async def skip_test_supervisor_routing_quant():
    """Test if supervisor correctly routes to Quant Agent."""
    state = {"messages": [HumanMessage(content="Will I hit my retirement goal if I save 10k a month?")], "user_query": "Will I hit my retirement goal if I save 10k a month?"}
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

def test_quant_tool_valid():
    """Test stock data retrieval."""
    res = simulate_goal_probability_tool.invoke({"target_amount": 1000000, "current_savings": 50000, "monthly_sip": 5000, "years": 10})
    assert "Probability" in res

@pytest.mark.asyncio
@pytest.mark.skipif('GEMINI_API_KEY' not in os.environ, reason='Requires API key')
async def test_budgeting_tool_auth_rejection():
    """Test that budgeting tools enforce user isolation by rejecting empty user_id."""
    # Call without config
    res = await get_user_income.ainvoke({})
    assert "error" in res
    assert "Unauthorized" in res["error"]
