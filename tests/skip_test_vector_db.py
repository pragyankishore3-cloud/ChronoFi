import pytest
import os
import numpy as np
from unittest.mock import patch, MagicMock

from models.database import DocumentChunk
from engine.vector_store import embed_chunks, retrieve_documents

@pytest.mark.asyncio
async def test_embed_chunks(db_session):
    """Exit Gate: 100% of successfully processed chunks receive embeddings."""
    chunks = [
        DocumentChunk(document_id=1, user_id=1, page_number=1, chunk_index=0, content="Some financial text"),
        DocumentChunk(document_id=1, user_id=1, page_number=2, chunk_index=1, content="More financial text")
    ]
    
    # We mock the embeddings model so it doesn't hit the API during test
    mock_embeddings = [[0.1] * 768, [0.2] * 768]
    
    with patch("engine.vector_store.embeddings_model.aembed_documents", return_value=mock_embeddings):
        await embed_chunks(chunks, db_session)
        
    # Check that embeddings were saved
    assert chunks[0].embedding == mock_embeddings[0]
    assert chunks[1].embedding == mock_embeddings[1]

@pytest.mark.asyncio
async def test_retrieve_documents_cross_user_isolation(db_session):
    """
    Exit Gate: Cross-user retrievals = 0.
    Exit Gate: 100% of embeddings belong to the correct user.
    """
    # Create chunks for User A
    user_a_chunk = DocumentChunk(document_id=10, user_id=42, page_number=1, chunk_index=0, content="User A Secret Salary", embedding=[0.9, 0.1, 0.1])
    # Create chunks for User B
    user_b_chunk = DocumentChunk(document_id=20, user_id=99, page_number=1, chunk_index=0, content="User B Secret Salary", embedding=[0.9, 0.2, 0.1])
    
    db_session.add(user_a_chunk)
    db_session.add(user_b_chunk)
    await db_session.commit()
    
    query = "What is the secret salary?"
    mock_query_embedding = [0.9, 0.15, 0.1] # Similar to both
    
    # Simulate User B searching
    with patch("engine.vector_store.embeddings_model.aembed_query", return_value=mock_query_embedding):
        results = await retrieve_documents(query, user_id=99, db=db_session, top_k=5)
        
    # Security Rule: Only User B's chunk should be returned, NEVER User A's
    assert len(results) == 1
    assert results[0].user_id == 99
    assert results[0].content == "User B Secret Salary"
    
@pytest.mark.asyncio
async def test_deleted_document_retrieval(db_session):
    """Exit Gate: Deleted documents cannot be retrieved."""
    from models.database import Document, DocumentStatus
    
    # Create Document and Chunk
    doc = Document(user_id=1, filename="temp.pdf", file_path="/tmp", mime_type="pdf", size_bytes=100, status=DocumentStatus.READY)
    db_session.add(doc)
    await db_session.commit()
    await db_session.refresh(doc)
    
    chunk = DocumentChunk(document_id=doc.id, user_id=1, page_number=1, chunk_index=0, content="Secret", embedding=[1.0, 0.0])
    db_session.add(chunk)
    await db_session.commit()
    
    # Delete Document (simulating API delete route causing cascade delete)
    await db_session.delete(doc)
    await db_session.commit()
    
    # Search should return 0 chunks because cascade delete removed them
    with patch("engine.vector_store.embeddings_model.aembed_query", return_value=[1.0, 0.0]):
        results = await retrieve_documents("Secret", user_id=1, db=db_session)
        
    assert len(results) == 0
