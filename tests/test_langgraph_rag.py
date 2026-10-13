from unittest.mock import AsyncMock, patch

import pytest
from langchain_core.messages import HumanMessage

from ai.supervisor import supervisor_node


@pytest.fixture
def base_state():
    return {
        "messages": [],
        "user_query": "",
        "user_id": 1,
        "investment_data": {},
        "tax_data": {},
        "budget_data": {},
        "next": "",
        "errors": []
    }

@pytest.mark.asyncio
async def test_supervisor_routes_to_document_agent(base_state):
    """Exit Gate: Supervisor accurately routes to Document_Agent when appropriate."""
    base_state["user_query"] = "What does my uploaded Form 16 say about my salary?"
    base_state["messages"] = [HumanMessage(content=base_state["user_query"])]
    
    # We mock the structured output call to mimic LLM routing
    mock_routing = {"agents": ["Document_Agent"], "reason": "User is asking about an uploaded document."}
    
    mock_agent = AsyncMock()
    mock_agent.ainvoke = AsyncMock(return_value=mock_routing)
    with patch("ai.supervisor.supervisor_agent", mock_agent):
        result = await supervisor_node(base_state, config={"configurable": {"user_id": 1}})
        
    assert result["next"] == "Document_Agent"

@pytest.mark.asyncio
async def test_supervisor_routes_away_from_document_agent(base_state):
    """Exit Gate: Supervisor does not unnecessarily trigger RAG."""
    base_state["user_query"] = "What is the tax rate on $100,000?"
    base_state["messages"] = [HumanMessage(content=base_state["user_query"])]
    
    mock_routing = {"agents": ["Tax_Agent"], "reason": "General tax question."}
    
    mock_agent = AsyncMock()
    mock_agent.ainvoke = AsyncMock(return_value=mock_routing)
    with patch("ai.supervisor.supervisor_agent", mock_agent):
        result = await supervisor_node(base_state, config={"configurable": {"user_id": 1}})
        
    assert result["next"] == "Tax_Agent"

@pytest.mark.asyncio
async def test_supervisor_finishes_when_done(base_state):
    base_state["user_query"] = "Thank you."
    base_state["messages"] = [HumanMessage(content=base_state["user_query"])]
    
    mock_routing = {"agents": ["FINISH"], "reason": "Conversation is over."}
    
    mock_agent = AsyncMock()
    mock_agent.ainvoke = AsyncMock(return_value=mock_routing)
    with patch("ai.supervisor.supervisor_agent", mock_agent):
        result = await supervisor_node(base_state, config={"configurable": {"user_id": 1}})
        
    assert result["next"] == "FINISH"


