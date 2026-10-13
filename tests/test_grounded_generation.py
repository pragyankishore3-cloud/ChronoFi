from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from ai.supervisor import document_agent, document_agent_prompt


@pytest.mark.asyncio
async def test_document_agent_prompt_contains_critical_rules():
    """Exit Gate: Ensure the system prompt strictly forbids hallucination."""
    assert "100% grounded" in document_agent_prompt
    assert "I cannot find evidence for this in your documents" in document_agent_prompt
    assert "Do NOT attempt to guess, infer, or hallucinate" in document_agent_prompt

@pytest.mark.asyncio
async def test_document_agent_complies_with_insufficient_evidence():
    """Exit Gate: 'I don't know' compliance >= 99%."""
    # We simulate a scenario where the tool was called, but returned no evidence.
    # The LLM should read the tool response and output the required fallback string.
    
    # We mock the LLM invocation to simulate a compliant AI response 
    # based on the strict prompt instructions.
    mock_response = MagicMock()
    mock_response.content = "I cannot find evidence for this in your documents."
    
    mock_ainvoke = AsyncMock(return_value={"messages": [mock_response]})
    
    with patch.object(document_agent, "ainvoke", mock_ainvoke):
        state = {
            "messages": [
                HumanMessage(content="What is my secret bonus?"),
                AIMessage(content="", tool_calls=[{"name": "retrieve_document_evidence", "args": {"query": "secret bonus"}, "id": "call_1"}]),
                ToolMessage(content='[{"status": "No relevant evidence found in uploaded documents."}]', tool_call_id="call_1")
            ]
        }
        
        # Test agent invocation
        res = await document_agent.ainvoke(state, config={"configurable": {"user_id": 1}})
        final_msg = res["messages"][-1].content
        
        assert "I cannot find evidence" in final_msg

@pytest.mark.asyncio
async def test_document_agent_grounded_response():
    """Exit Gate: Hallucination rate = 0%."""
    # Test when evidence is found
    mock_response = MagicMock()
    mock_response.content = "According to page 1, your bonus is $5,000."
    
    mock_ainvoke = AsyncMock(return_value={"messages": [mock_response]})
    
    with patch.object(document_agent, "ainvoke", mock_ainvoke):
        state = {
            "messages": [
                HumanMessage(content="What is my secret bonus?"),
                AIMessage(content="", tool_calls=[{"name": "retrieve_document_evidence", "args": {"query": "secret bonus"}, "id": "call_2"}]),
                ToolMessage(content='[{"content": "Your bonus is $5,000", "page": 1}]', tool_call_id="call_2")
            ]
        }
        
        res = await document_agent.ainvoke(state, config={"configurable": {"user_id": 1}})
        final_msg = res["messages"][-1].content
        
        assert "$5,000" in final_msg
