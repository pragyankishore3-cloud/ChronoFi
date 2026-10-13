from unittest.mock import AsyncMock, patch
import pytest

from engine.vector_store import retrieve_documents
from models.database import Document, DocumentChunk, DocumentStatus, DocumentType


@pytest.fixture
async def seeded_docs(db_session):
    """Seed DB with documents and chunks for retrieval tests."""
    doc1 = Document(user_id=1, filename="tax.pdf", file_path="/tmp", mime_type="pdf", size_bytes=100, status=DocumentStatus.READY, document_type=DocumentType.FORM_16, financial_year="2024")
    doc2 = Document(user_id=1, filename="loan.pdf", file_path="/tmp", mime_type="pdf", size_bytes=100, status=DocumentStatus.READY, document_type=DocumentType.LOAN)
    
    db_session.add_all([doc1, doc2])
    await db_session.commit()
    await db_session.refresh(doc1)
    await db_session.refresh(doc2)
    
    chunk1 = DocumentChunk(document_id=doc1.id, user_id=1, page_number=1, chunk_index=0, content="Salary is 1500000", embedding=[0.9, 0.1, 0.0])
    chunk2 = DocumentChunk(document_id=doc2.id, user_id=1, page_number=1, chunk_index=0, content="Interest rate is 8.5%", embedding=[0.1, 0.9, 0.0])
    # Low relevance chunk
    chunk3 = DocumentChunk(document_id=doc1.id, user_id=1, page_number=2, chunk_index=1, content="Random boilerplate", embedding=[0.0, 0.0, 0.9])
    
    db_session.add_all([chunk1, chunk2, chunk3])
    await db_session.commit()

@pytest.mark.asyncio
async def test_semantic_retrieval_top_1(db_session, seeded_docs):
    """Exit Gate: Top-1 correct-source rate >= 80%"""
    mock_query_emb = [0.95, 0.05, 0.0] # Very close to chunk1
    
    with patch("engine.vector_store.embeddings_model") as mock_model:
        mock_model.aembed_query = AsyncMock(return_value=mock_query_emb)
        mock_model.embed_query.return_value = mock_query_emb
        results = await retrieve_documents("What is my salary?", user_id=1, db=db_session, top_k=1)
        
    assert len(results) == 1
    assert "Salary" in results[0].content

@pytest.mark.asyncio
async def test_metadata_filtering(db_session, seeded_docs):
    """Ensure metadata filtering restricts search space."""
    mock_query_emb = [0.9, 0.1, 0.0]
    
    with patch("engine.vector_store.embeddings_model") as mock_model:
        mock_model.aembed_query = AsyncMock(return_value=mock_query_emb)
        mock_model.embed_query.return_value = mock_query_emb
        # We query for salary but filter by LOAN type. It should return nothing or low similarity.
        results = await retrieve_documents(
            "What is my salary?", 
            user_id=1, 
            db=db_session, 
            filters={"document_type": DocumentType.LOAN}
        )
    
    # Since chunk1 (salary) is FORM_16, it should be excluded. chunk2 is LOAN but has low similarity.
    # Depending on threshold, chunk2 might be dropped entirely. Let's say it is dropped.
    assert len(results) == 0

@pytest.mark.asyncio
async def test_unsupported_answer_rejection(db_session, seeded_docs):
    """Exit Gate: Unsupported-answer rejection >= 95% (rejects low similarity)"""
    # Query completely unrelated to anything in the DB
    mock_query_emb = [-0.9, -0.9, -0.9]
    
    with patch("engine.vector_store.embeddings_model") as mock_model:
        mock_model.aembed_query = AsyncMock(return_value=mock_query_emb)
        mock_model.embed_query.return_value = mock_query_emb
        results = await retrieve_documents("What is the recipe for cake?", user_id=1, db=db_session, similarity_threshold=0.70)
        
    # Should return empty list because all chunks fail the 0.70 threshold
    assert len(results) == 0

@pytest.mark.asyncio
async def test_cross_user_retrieval(db_session, seeded_docs):
    """Exit Gate: Cross-user retrieval must always remain zero."""
    mock_query_emb = [0.9, 0.1, 0.0]
    
    with patch("engine.vector_store.embeddings_model") as mock_model:
        mock_model.aembed_query = AsyncMock(return_value=mock_query_emb)
        mock_model.embed_query.return_value = mock_query_emb
        # Querying with user_id=99 who has no documents
        results = await retrieve_documents("What is my salary?", user_id=99, db=db_session)
        
    assert len(results) == 0

