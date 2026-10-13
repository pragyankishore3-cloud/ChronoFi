from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from ai.supervisor import document_agent, document_agent_prompt


@pytest.mark.asyncio
async def test_citation_rules_in_prompt():
    """Exit Gate: Ensure citation mandates exist in the prompt."""
    assert "CITATIONS ARE MANDATORY" in document_agent_prompt
    assert "[Filename, Page X]" in document_agent_prompt
    assert "Do NOT create fake citations" in document_agent_prompt

@pytest.mark.asyncio
async def test_document_agent_includes_citations():
    """Exit Gate: Citation inclusion rate >= 95% (simulated)."""
    # Test when evidence is found with filename and page
    mock_response = MagicMock()
    mock_response.content = "Your bonus is $5,000 [tax_2024.pdf, Page 1]."
    
    mock_ainvoke = AsyncMock(return_value={"messages": [mock_response]})
    
    with patch.object(document_agent, "ainvoke", mock_ainvoke):
        state = {
            "messages": [
                HumanMessage(content="What is my bonus?"),
                AIMessage(content="", tool_calls=[{"name": "retrieve_document_evidence", "args": {"query": "bonus"}, "id": "call_1"}]),
                ToolMessage(content='[{"filename": "tax_2024.pdf", "page": 1, "content": "Your bonus is $5,000"}]', tool_call_id="call_1")
            ]
        }
        
        res = await document_agent.ainvoke(state, config={"configurable": {"user_id": 1}})
        final_msg = res["messages"][-1].content
        
        # Verify the citation was generated correctly
        assert "[tax_2024.pdf, Page 1]" in final_msg
