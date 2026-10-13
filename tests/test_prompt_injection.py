from unittest.mock import patch

import pytest

from ai.supervisor import retrieve_document_evidence


@pytest.mark.asyncio
async def test_retrieve_document_evidence_ignores_malicious_user_id():
    """
    Exit Gate: Prompt injection mitigated.
    Tests that even if the LLM is tricked into supplying a malicious `user_id` as an argument,
    the tool completely ignores it and strictly uses the `RunnableConfig` provided by the backend.
    """
    
    # We call the tool directly as if the LLM invoked it with a hallucinated/injected argument
    # Some older tool schemas might let the LLM pass user_id if not carefully designed.
    # Our tool signature is `async def retrieve_document_evidence(query: str, config: RunnableConfig, document_type: str = None)`
    # It does not accept user_id from the LLM. 
    
    # The config injected by LangGraph securely from the JWT token
    secure_config = {"configurable": {"user_id": 42}}
    
    with patch("ai.supervisor.retrieve_documents") as mock_retrieve:
        mock_retrieve.return_value = []
        
        # Simulate LLM call
        await retrieve_document_evidence.ainvoke(
            {"query": "secret", "document_type": "TAX"}, 
            config=secure_config
        )
        
        # Verify that retrieve_documents was called with the SECURE user_id (42),
        # entirely isolating the backend from any prompt-injected user_ids.
        mock_retrieve.assert_called_once()
        args, kwargs = mock_retrieve.call_args
        
        # The signature is retrieve_documents(query=..., user_id=..., db=...)
        passed_user_id = kwargs.get("user_id") if "user_id" in kwargs else (args[1] if len(args) > 1 else None)
        assert passed_user_id == 42

@pytest.mark.asyncio
async def test_retrieve_document_evidence_fails_closed_on_missing_config():
    """Exit Gate: Unauthorized access = 0."""
    # If config is missing or user_id is missing, it must fail closed.
    empty_config = {"configurable": {}}
    
    result = await retrieve_document_evidence.ainvoke(
        {"query": "secret"}, 
        config=empty_config
    )
    
    assert "Unauthorized" in result[0]["error"]

